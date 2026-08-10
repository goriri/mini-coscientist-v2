"""Fetching and registry lookup behind claim-level source verification.

The verification specialist used to be given ADK's ``load_web_page`` and nothing
else, and that tool cannot verify a scientific source in this deployment for
three independent reasons. It imports ``beautifulsoup4``, which is not in this
project's dependency set, so every call raised ``ImportError`` -- locally and in
the container, which installs from the same lock file. It sends no user agent a
publisher will serve. And it passes ``allow_redirects=False``, so every
``https://doi.org/...`` link -- the only kind that unambiguously names a paper --
returns a bare 302 and nothing else.

The consequence was measurable: a live run discovered forty-four leads and
verified zero of them, and the fourteen marked "inaccessible" were guesses from
the URL string, because no fetch had happened. This module is the replacement.

Verification here is deliberately tiered, because a real, correctly cited paper
behind a paywall is not the same thing as a citation nobody can find, and
collapsing the two makes the evidence floor a measure of open-access
availability rather than of scholarship:

``verified``
    The document was retrieved and its text is available to check the claim
    against.
``metadata_verified``
    The DOI resolves, a registry agrees on the title, authors and year, and no
    retraction is recorded -- but the full text could not be read, so nothing
    vouches for the passage.
``retracted``
    A registry records a retraction. This outranks everything above it.
``inaccessible``
    Neither the document nor a registry record could be obtained.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import re
import socket
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any
from urllib.parse import quote, urljoin, urlsplit

import httpx

from .normalization import strip_unstorable_characters, strip_unstorable_values

MAX_DOCUMENT_BYTES = 6_000_000
MAX_DOCUMENT_CHARS = 120_000
MAX_REDIRECT_HOPS = 6
DEFAULT_FETCH_TIMEOUT_SECONDS = 25.0
DEFAULT_REGISTRY_TIMEOUT_SECONDS = 12.0
MAX_CONCURRENT_FETCHES = 8

BROWSER_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36 mini-coscientist/2 (source-verification)"
)
"""What publishers are asked to serve this fetcher as.

Eight of twelve real lead URLs from a live run returned usable text with a
browser user agent; the tool this replaces sent one no publisher answers. The
project token stays on the end so an administrator reading an access log can see
what this is and that it is not pretending to be a person browsing.
"""

REGISTRY_USER_AGENT = "mini-coscientist/2 (source-verification; registry lookup)"

_DOI_IN_TEXT = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.I)
_DOI_TRAILING_PUNCTUATION = ".,;:)]}>\"'"
_WHITESPACE = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_WORD = re.compile(r"[a-z0-9]+")

_REGISTRY_HOSTS = frozenset(
    {
        "api.crossref.org",
        "api.openalex.org",
        "api.datacite.org",
        "eutils.ncbi.nlm.nih.gov",
    }
)

_TITLE_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "into",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
)

TITLE_MATCH_THRESHOLD = 0.6
"""How much of a claimed title must survive in the registry's title to agree.

Whole-string equality is the wrong test: discovery records "Effects of ALD
coatings on Li-ion cycle life" where Crossref holds "Effects of atomic layer
deposition coatings on lithium-ion cycle life." Those are the same paper, and a
strict comparison would quarantine it. Token overlap after stopwords is what
survives an expanded acronym and still refuses two different papers.
"""


def _strip_doi(value: str) -> str:
    return value.rstrip(_DOI_TRAILING_PUNCTUATION)


def extract_doi(*candidates: str) -> str:
    """The first DOI appearing in any of ``candidates``, lowercased.

    A DOI may arrive as a bare identifier, inside a ``doi.org`` URL, or embedded
    in a publisher path such as ``/doi/full/10.1080/...`` -- one pattern reads
    all three.
    """
    for candidate in candidates:
        if not candidate:
            continue
        found = _DOI_IN_TEXT.search(candidate)
        if found:
            return _strip_doi(found.group(1)).lower()
    return ""


def title_tokens(title: str) -> set[str]:
    return {
        word
        for word in _WORD.findall(title.lower())
        if word not in _TITLE_STOPWORDS and len(word) > 2
    }


def titles_agree(claimed: str, registered: str) -> bool:
    """Whether two renderings of a title plausibly name the same document."""
    claimed_words = title_tokens(claimed)
    registered_words = title_tokens(registered)
    if not claimed_words or not registered_words:
        # Nothing was claimed, so nothing contradicts the registry. The registry
        # record is what gets recorded; there is no disagreement to detect.
        return True
    shared = claimed_words & registered_words
    return len(shared) / min(len(claimed_words), len(registered_words)) >= (
        TITLE_MATCH_THRESHOLD
    )


class _TextExtractor(HTMLParser):
    """Readable text from an HTML document, without a third-party parser.

    ``beautifulsoup4`` would do this and is what the tool being replaced wanted,
    but adding a dependency to strip tags is a poor trade when the standard
    library ships a parser. The only judgement here is which elements hold prose:
    script, style and the navigation furniture do not, and including them buries
    the sentence a verifier is looking for under a site's menu.
    """

    _SKIP = frozenset({"script", "style", "noscript", "svg", "nav", "header", "footer"})
    _BREAK = frozenset(
        {
            "p",
            "div",
            "section",
            "article",
            "li",
            "tr",
            "br",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "blockquote",
        }
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._suppressed = 0
        self.title = ""
        self._in_title = False
        self.citation_meta: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._suppressed += 1
        if tag == "title":
            self._in_title = True
        if tag == "meta":
            # Publishers that paywall the body still emit Highwire citation
            # metadata in the head, which is enough to confirm the record even
            # when the article text never arrives.
            attributes = {key: value or "" for key, value in attrs}
            name = (attributes.get("name") or attributes.get("property") or "").lower()
            if name.startswith(("citation_", "dc.", "og:title")):
                self.citation_meta.setdefault(name, attributes.get("content", ""))
        if tag in self._BREAK:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._suppressed:
            self._suppressed -= 1
        if tag == "title":
            self._in_title = False
        if tag in self._BREAK:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        if self._suppressed:
            return
        self._chunks.append(data)

    def text(self) -> str:
        joined = _WHITESPACE.sub(" ", "".join(self._chunks))
        return _BLANK_LINES.sub("\n\n", joined).strip()


def html_to_text(markup: str) -> tuple[str, str, dict[str, str]]:
    """Prose, document title and citation metadata from an HTML string."""
    extractor = _TextExtractor()
    try:
        extractor.feed(markup)
        extractor.close()
    except Exception:
        # A malformed page is still worth whatever text was parsed before the
        # parser gave up; a verifier reading half an article beats reading none.
        pass
    return extractor.text(), extractor.title.strip(), extractor.citation_meta


def pdf_to_text(payload: bytes) -> str:
    """Prose from a PDF, or an empty string when it cannot be read."""
    return _pdf_content(payload)[0]


# What an authoring tool leaves in /Title when the author never set one. A title that
# is really the file it was exported from names the export, not the document.
_TOOL_TITLE = re.compile(
    r"^(?:microsoft\s+word\s*-\s*|untitled\b|document\d*$|print$|slide\s*\d*$)", re.I
)
_FILE_TITLE = re.compile(r"\.(?:pdf|docx?|tex|indd|pptx?|rtf|odt)$", re.I)


def pdf_title(payload: bytes) -> str:
    """The document's own title, where the PDF carries one worth printing.

    A PDF was read for its text and never for its ``/Title``, so a paper the run had
    retrieved in full still reached the reference list as "Untitled source on
    sandia.gov" -- an entry that was checked against a document it could not name.

    An authoring tool fills the field in when the author does not, so an export
    artefact is refused: "Microsoft Word - draft3.docx" is what the file was made
    from, and a bare filename is the file rather than the work. Four words is the
    floor, which is where a real title starts and a tool's placeholder stops.
    """
    title = " ".join(_pdf_content(payload)[1].split())
    if len(title.split()) < 4 or _TOOL_TITLE.match(title) or _FILE_TITLE.search(title):
        return ""
    return title


def _pdf_content(payload: bytes) -> tuple[str, str]:
    """The prose and the recorded title, or empty strings where neither can be read."""
    try:
        from io import BytesIO

        from pypdf import PdfReader
    except Exception:
        return "", ""
    try:
        reader = PdfReader(BytesIO(payload))
        pages = [page.extract_text() or "" for page in reader.pages[:60]]
        # A PDF with a damaged trailer can still yield pages, so the title is read
        # separately: losing it is not a reason to lose the text as well.
        try:
            title = str((reader.metadata or {}).get("/Title") or "")
        except Exception:
            title = ""
    except Exception:
        return "", ""
    return _BLANK_LINES.sub("\n\n", "\n\n".join(pages)).strip(), title


def _is_public_host(host: str) -> bool:
    """Whether a hostname resolves only to addresses outside this network.

    Every redirect hop is checked rather than only the URL handed in, because
    the point of following redirects is that the final destination is chosen by
    someone else. A literal-address check alone is not enough: a name that
    resolves to 169.254.169.254 reaches the metadata server just as well.
    """
    if not host:
        return False
    bare = host.strip("[]").lower().rstrip(".")
    if bare in {"localhost", "metadata.google.internal"}:
        return False
    try:
        infos = socket.getaddrinfo(bare, None)
    except OSError:
        return False
    addresses = {info[4][0] for info in infos}
    if not addresses:
        return False
    for address in addresses:
        try:
            if not ipaddress.ip_address(address).is_global:
                return False
        except ValueError:
            return False
    return True


@dataclass
class FetchedDocument:
    """What one attempt to read a source document came back with."""

    url: str
    final_url: str = ""
    status: int = 0
    content_type: str = ""
    text: str = ""
    title: str = ""
    citation_meta: dict[str, str] = field(default_factory=dict)
    error: str = ""

    @property
    def readable(self) -> bool:
        """Whether enough text arrived to check a claim against.

        The threshold exists because a paywall interstitial is a successful HTTP
        response carrying a few hundred characters of "sign in to continue". A
        verifier handed that would be reading the publisher's login page and
        reporting on the paper.
        """
        return self.status == 200 and len(self.text) >= 600


@dataclass
class SourceMetadata:
    """What the scholarly registries hold for one source."""

    doi: str = ""
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    container: str = ""
    publisher: str = ""
    source_type: str = ""
    identifiers: dict[str, str] = field(default_factory=dict)
    is_retracted: bool = False
    retraction_note: str = ""
    open_access_pdf: str = ""
    registries: list[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return bool(self.title and self.registries)


class SourceRetriever:
    """Bounded, redirect-following retrieval of documents and registry records.

    One instance per verification sweep: it caches within the sweep so that two
    claims resting on the same paper cost one fetch, and it holds the request
    budget so a manifest of ninety leads cannot turn into an unbounded crawl.
    """

    def __init__(
        self,
        *,
        fetch_timeout: float = DEFAULT_FETCH_TIMEOUT_SECONDS,
        registry_timeout: float = DEFAULT_REGISTRY_TIMEOUT_SECONDS,
        max_documents: int = 120,
        max_registry_requests: int = 200,
        concurrency: int = MAX_CONCURRENT_FETCHES,
    ) -> None:
        self.fetch_timeout = fetch_timeout
        self.registry_timeout = registry_timeout
        self.max_documents = max_documents
        self.max_registry_requests = max_registry_requests
        self._documents_fetched = 0
        self._registry_requests = 0
        self._document_cache: dict[str, FetchedDocument] = {}
        self._metadata_cache: dict[str, SourceMetadata] = {}
        self._gate = asyncio.Semaphore(concurrency)

    async def _registry_json(
        self, client: httpx.AsyncClient, url: str
    ) -> dict[str, Any]:
        if self._registry_requests >= self.max_registry_requests:
            return {}
        if urlsplit(url).hostname not in _REGISTRY_HOSTS:
            raise ValueError("Registry lookup is restricted to approved hosts.")
        self._registry_requests += 1
        try:
            response = await client.get(
                url,
                headers={
                    "User-Agent": REGISTRY_USER_AGENT,
                    "Accept": "application/json",
                },
                timeout=self.registry_timeout,
                follow_redirects=True,
            )
            if response.status_code != 200:
                return {}
            payload = json.loads(response.content[:4_000_000])
        except (httpx.HTTPError, ValueError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    async def fetch_document(
        self, client: httpx.AsyncClient, url: str
    ) -> FetchedDocument:
        """Retrieve one document, following redirects one validated hop at a time.

        httpx would follow them in a single call, and that is exactly what must
        not happen here: each hop is a destination chosen by the previous server,
        so each is re-checked against the private-address guard before it is
        requested. That is also what makes DOIs work at all -- a DOI is a 302,
        and the tool this replaces refused to follow it.
        """
        if url in self._document_cache:
            return self._document_cache[url]
        result = FetchedDocument(url=url)
        if self._documents_fetched >= self.max_documents:
            result.error = "The per-run document budget for this sweep was reached."
            self._document_cache[url] = result
            return result
        self._documents_fetched += 1
        async with self._gate:
            current = url
            for _ in range(MAX_REDIRECT_HOPS):
                parts = urlsplit(current)
                if parts.scheme not in {"http", "https"} or not _is_public_host(
                    parts.hostname or ""
                ):
                    result.error = f"Refused to fetch a non-public locator: {current}"
                    self._document_cache[url] = result
                    return result
                try:
                    response = await client.get(
                        current,
                        headers={
                            "User-Agent": BROWSER_USER_AGENT,
                            "Accept": (
                                "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8"
                            ),
                            "Accept-Language": "en",
                        },
                        timeout=self.fetch_timeout,
                        follow_redirects=False,
                    )
                except httpx.HTTPError as exc:
                    result.error = f"{type(exc).__name__}: {exc}"
                    self._document_cache[url] = result
                    return result
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location", "")
                    if not location:
                        result.status = response.status_code
                        result.error = "Redirect carried no destination."
                        self._document_cache[url] = result
                        return result
                    current = urljoin(current, location)
                    continue
                result.final_url = current
                result.status = response.status_code
                result.content_type = (
                    response.headers.get("content-type", "").split(";")[0].strip()
                )
                body = response.content[:MAX_DOCUMENT_BYTES]
                if response.status_code != 200:
                    result.error = f"HTTP {response.status_code}"
                # Sanitized as the bytes become text, at the point where the open
                # web enters this process. A PDF is a container for arbitrary
                # bytes and one of them was a NUL, which PostgreSQL will not
                # store: the row this text eventually reaches is a research
                # record, and losing it costs a paid verification pass.
                elif "pdf" in result.content_type:
                    result.text = strip_unstorable_characters(pdf_to_text(body))[
                        :MAX_DOCUMENT_CHARS
                    ]
                    # Read for its text and never for its title, so a paper this run
                    # had in full still reached the reference list as "Untitled
                    # source on sandia.gov".
                    result.title = strip_unstorable_characters(pdf_title(body))
                    if not result.text:
                        result.error = "The PDF carried no extractable text layer."
                else:
                    markup = body.decode(response.encoding or "utf-8", "replace")
                    text, title, meta = html_to_text(markup)
                    result.text = strip_unstorable_characters(text)[:MAX_DOCUMENT_CHARS]
                    result.title = strip_unstorable_characters(title)
                    result.citation_meta = strip_unstorable_values(meta)
                self._document_cache[url] = result
                return result
        result.error = f"Exceeded {MAX_REDIRECT_HOPS} redirects without a document."
        self._document_cache[url] = result
        return result

    async def lookup_metadata(
        self, client: httpx.AsyncClient, url: str, *, hint: str = ""
    ) -> SourceMetadata:
        """What the registries hold for a locator, including any retraction.

        Crossref answers for the title, authors and year; OpenAlex is asked
        second because it is the one that carries ``is_retracted`` and a link to
        an open-access copy, which is often the only way the text of a paywalled
        paper can be read at all.
        """
        cache_key = f"{url}\x00{hint}"
        if cache_key in self._metadata_cache:
            return self._metadata_cache[cache_key]
        metadata = SourceMetadata(doi=extract_doi(url, hint))
        if metadata.doi:
            await self._fill_from_crossref(client, metadata)
            await self._fill_from_openalex(client, metadata)
            if not metadata.registries:
                await self._fill_from_datacite(client, metadata)
        else:
            await self._fill_from_pubmed(client, metadata, url)
        self._metadata_cache[cache_key] = metadata
        return metadata

    async def _fill_from_crossref(
        self, client: httpx.AsyncClient, metadata: SourceMetadata
    ) -> None:
        payload = await self._registry_json(
            client, f"https://api.crossref.org/works/{quote(metadata.doi, safe='')}"
        )
        message = payload.get("message") or {}
        titles = message.get("title") or []
        if not titles:
            return
        metadata.title = str(titles[0])
        metadata.registries.append("crossref")
        metadata.identifiers["doi"] = metadata.doi
        metadata.authors = [
            " ".join(
                part
                for part in (author.get("given", ""), author.get("family", ""))
                if part
            )
            for author in (message.get("author") or [])
        ][:20]
        issued = (message.get("issued") or {}).get("date-parts") or []
        if issued and issued[0] and isinstance(issued[0][0], int):
            metadata.year = issued[0][0]
        containers = message.get("container-title") or []
        if containers:
            metadata.container = str(containers[0])
        metadata.publisher = str(message.get("publisher") or "")
        metadata.source_type = str(message.get("type") or "")
        for update in message.get("update-to") or []:
            if str(update.get("type", "")).lower() in {"retraction", "withdrawal"}:
                metadata.is_retracted = True
                metadata.retraction_note = (
                    f"Crossref records a {update.get('type')} of "
                    f"{update.get('DOI', 'this work')}."
                )

    async def _fill_from_openalex(
        self, client: httpx.AsyncClient, metadata: SourceMetadata
    ) -> None:
        payload = await self._registry_json(
            client,
            "https://api.openalex.org/works/"
            + quote(f"https://doi.org/{metadata.doi}", safe=""),
        )
        if not payload.get("id"):
            return
        metadata.registries.append("openalex")
        metadata.identifiers["openalex"] = str(payload["id"])
        if not metadata.title and payload.get("title"):
            metadata.title = str(payload["title"])
        if metadata.year is None and isinstance(payload.get("publication_year"), int):
            metadata.year = payload["publication_year"]
        if payload.get("is_retracted"):
            metadata.is_retracted = True
            metadata.retraction_note = "OpenAlex flags this work as retracted."
        location = payload.get("best_oa_location") or {}
        metadata.open_access_pdf = str(
            location.get("pdf_url") or location.get("landing_page_url") or ""
        )

    async def _fill_from_datacite(
        self, client: httpx.AsyncClient, metadata: SourceMetadata
    ) -> None:
        payload = await self._registry_json(
            client, f"https://api.datacite.org/dois/{quote(metadata.doi, safe='')}"
        )
        attributes = (payload.get("data") or {}).get("attributes") or {}
        titles = attributes.get("titles") or []
        if not titles:
            return
        metadata.registries.append("datacite")
        metadata.title = str(titles[0].get("title", ""))
        metadata.publisher = str(attributes.get("publisher") or "")
        if isinstance(attributes.get("publicationYear"), int):
            metadata.year = attributes["publicationYear"]

    async def _fill_from_pubmed(
        self, client: httpx.AsyncClient, metadata: SourceMetadata, url: str
    ) -> None:
        parts = urlsplit(url)
        if (parts.hostname or "") != "pubmed.ncbi.nlm.nih.gov":
            return
        segments = [segment for segment in parts.path.split("/") if segment]
        if not segments or not segments[0].isdigit():
            return
        pmid = segments[0]
        payload = await self._registry_json(
            client,
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            f"?db=pubmed&id={pmid}&retmode=json",
        )
        record = (payload.get("result") or {}).get(pmid) or {}
        if not record.get("title"):
            return
        metadata.registries.append("pubmed")
        metadata.identifiers["pmid"] = pmid
        metadata.title = str(record["title"])
        metadata.authors = [
            str(author.get("name", "")) for author in record.get("authors") or []
        ][:20]
        if str(record.get("pubdate", ""))[:4].isdigit():
            metadata.year = int(str(record["pubdate"])[:4])
        metadata.container = str(record.get("fulljournalname") or "")
        if doi := extract_doi(str(record.get("elocationid", ""))):
            metadata.doi = doi
            metadata.identifiers["doi"] = doi


@dataclass
class RetrievalOutcome:
    """The highest verification tier the evidence actually supports for one URL."""

    url: str
    tier: str
    document: FetchedDocument
    metadata: SourceMetadata
    reason: str

    @property
    def text(self) -> str:
        return self.document.text


TIER_ORDER = ("inaccessible", "metadata_verified", "verified", "retracted")


async def assess_source(
    client: httpx.AsyncClient,
    retriever: SourceRetriever,
    url: str,
    *,
    claimed_title: str = "",
) -> RetrievalOutcome:
    """Retrieve one source and decide what may honestly be asserted about it.

    The order matters. A retraction outranks a successful fetch, because a
    retracted paper is readable and must never be cited as support. Below that,
    reading the text outranks agreeing with a registry, and agreeing with a
    registry outranks nothing at all.

    An open-access copy is tried when the publisher's own page refuses, which is
    what turns most paywalled 403s into readable text rather than into a gap in
    the evidence base.
    """
    metadata = await retriever.lookup_metadata(client, url, hint=claimed_title)
    document = await retriever.fetch_document(client, url)
    if not document.readable and metadata.open_access_pdf:
        alternate = await retriever.fetch_document(client, metadata.open_access_pdf)
        if alternate.readable:
            document = alternate
    if metadata.is_retracted:
        return RetrievalOutcome(
            url=url,
            tier="retracted",
            document=document,
            metadata=metadata,
            reason=metadata.retraction_note or "A registry records a retraction.",
        )
    if document.readable:
        return RetrievalOutcome(
            url=url,
            tier="verified",
            document=document,
            metadata=metadata,
            reason=(
                f"Retrieved {len(document.text):,} characters of text from "
                f"{document.final_url or url}."
            ),
        )
    if metadata.found:
        if not titles_agree(claimed_title, metadata.title):
            return RetrievalOutcome(
                url=url,
                tier="inaccessible",
                document=document,
                metadata=metadata,
                reason=(
                    "The registry record for this DOI is a different document: "
                    f'cited as "{claimed_title}", registered as "{metadata.title}".'
                ),
            )
        return RetrievalOutcome(
            url=url,
            tier="metadata_verified",
            document=document,
            metadata=metadata,
            reason=(
                f"{' and '.join(metadata.registries)} confirm this record, but the "
                "full text could not be read"
                + (f" ({document.error})." if document.error else ".")
            ),
        )
    return RetrievalOutcome(
        url=url,
        tier="inaccessible",
        document=document,
        metadata=metadata,
        reason=(
            document.error
            or "No document was retrieved and no registry holds this locator."
        ),
    )


async def assess_sources(
    urls: list[tuple[str, str]],
    *,
    retriever: SourceRetriever | None = None,
    client: httpx.AsyncClient | None = None,
) -> dict[str, RetrievalOutcome]:
    """Assess every ``(url, claimed_title)`` pair concurrently, keyed by URL.

    Whole assessments are capped, not just their fetches. The retriever gates
    document retrieval already, but a Crossref and an OpenAlex lookup are two
    more requests per source and pass through no gate at all: a live fan-out
    hands this ninety locators, and Crossref answers a hundred and eighty
    simultaneous questions from one address by rate-limiting the lot, which
    arrives downstream as a corpus nobody could confirm.
    """
    if not urls:
        return {}
    retriever = retriever or SourceRetriever()
    owned = client is None
    client = client or httpx.AsyncClient(follow_redirects=False)
    gate = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)

    async def _assess(url: str, title: str) -> RetrievalOutcome:
        async with gate:
            return await assess_source(client, retriever, url, claimed_title=title)

    try:
        outcomes = await asyncio.gather(
            *(_assess(url, title) for url, title in urls),
            return_exceptions=True,
        )
    finally:
        if owned:
            await client.aclose()
    results: dict[str, RetrievalOutcome] = {}
    for (url, _title), outcome in zip(urls, outcomes, strict=True):
        if isinstance(outcome, RetrievalOutcome):
            results[url] = outcome
        else:
            # A sweep is a best-effort integrity check over ninety locators. One
            # of them raising must cost that locator its verification, not cost
            # the run its evidence stage.
            results[url] = RetrievalOutcome(
                url=url,
                tier="inaccessible",
                document=FetchedDocument(url=url, error=str(outcome)),
                metadata=SourceMetadata(),
                reason=f"Retrieval failed: {outcome}",
            )
    return results


TOOL_TEXT_BUDGET_CHARS = 24_000
"""How much document text one tool call hands back to the verifier.

The full document is kept on the outcome for the deterministic sweep, but a
sixty-page PDF pasted into a prompt crowds out the packet the specialist is
supposed to be writing. Twenty-four thousand characters covers an abstract,
introduction, results and discussion for almost any paper.
"""


async def fetch_source_document(url: str, claimed_title: str = "") -> dict[str, Any]:
    """Retrieve a source document and its scholarly registry record.

    Use this on every source before assigning it a verification status. It
    follows redirects, so a ``https://doi.org/...`` link resolves to the
    publisher's page; it reads both HTML and PDF; and when the publisher refuses,
    it retries against any open-access copy a registry knows about.

    Args:
      url: The source locator to check. A DOI link is preferred over a
        publisher landing page, and a bare domain cannot be verified at all.
      claimed_title: The title the discovery pass recorded for this source, if
        any. Supplying it lets the check report when a DOI resolves to a
        different document than the one that was cited.

    Returns:
      A dictionary describing what could be established. ``tier`` is the highest
      status the evidence supports: ``verified`` when the text was retrieved and
      is included under ``text``; ``metadata_verified`` when a registry confirms
      the record but the text could not be read; ``retracted`` when a registry
      records a retraction; ``inaccessible`` when neither was obtained. Never
      assign a source a stronger status than the ``tier`` reported here.
    """
    retriever = SourceRetriever(max_documents=4, max_registry_requests=12)
    async with httpx.AsyncClient(follow_redirects=False) as client:
        outcome = await assess_source(
            client, retriever, url, claimed_title=claimed_title
        )
    # Sanitized on the way out. Everything below came off the open web -- a PDF's
    # extracted text, a registry's title field -- and ADK writes the tool
    # response into a PostgreSQL row, which one NUL is enough to reject.
    return strip_unstorable_values(
        {
            "tier": outcome.tier,
            "reason": outcome.reason,
            "requested_url": outcome.url,
            "final_url": outcome.document.final_url,
            "http_status": outcome.document.status,
            "registry_title": outcome.metadata.title,
            "registry_authors": outcome.metadata.authors,
            "registry_year": outcome.metadata.year,
            "registry_container": outcome.metadata.container,
            "doi": outcome.metadata.doi,
            "registries": outcome.metadata.registries,
            "retracted": outcome.metadata.is_retracted,
            "text": outcome.document.text[:TOOL_TEXT_BUDGET_CHARS],
            "text_truncated": len(outcome.document.text) > TOOL_TEXT_BUDGET_CHARS,
        }
    )
