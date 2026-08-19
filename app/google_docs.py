"""Publish a finished dossier into the reader's own Google Drive as a Doc.

The service already renders the dossier as a .docx, and until now the button
that said "Google Docs" handed that file over and left the rest to the reader:
download it, find Drive, upload it, wait for the conversion. This is the same
document taking itself there.

Two decisions worth naming, because both are about what this service is allowed
to hold.

*The document is created in the reader's Drive, not in a service account's and
shared out.* A dossier belongs to the researcher who commissioned it. A
service-account copy would be owned by the deployment, would count against the
deployment's quota, would be invisible to the researcher's own sharing controls
and would outlive their interest in it. So the reader authorises this, once, and
the file is theirs from the moment it exists.

*Only ``drive.file``, and only an access token.* ``drive.file`` grants this
service exactly the files it creates itself and nothing else already in the
Drive -- it cannot read, list or touch anything it did not put there, which is
the narrowest scope that can do this job at all. The authorisation is requested
with ``access_type=online``, so Google issues an access token good for about an
hour and no refresh token: there is no long-lived credential for this deployment
to store, leak or have to revoke. The token rides in a signed, HTTP-only cookie
on the reader's own browser, so the deployment holds no user credentials between
requests either -- the price is that the reader reconnects an hour later, which
is the right price for a public service that anybody may use.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from urllib.parse import urlencode

import httpx

AUTHORIZATION_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
UPLOAD_ENDPOINT = "https://www.googleapis.com/upload/drive/v3/files"

# The one scope that can create a file and cannot read the Drive it is created
# in. The broader ``drive`` scope would also work and would ask the reader to
# hand over every document they own to a research tool, which is not a trade
# anybody should be asked to make for an export button.
SCOPE = "https://www.googleapis.com/auth/drive.file"

DOCUMENT_MIME = "application/vnd.google-apps.document"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

TOKEN_COOKIE = "coscientist_gdocs"
STATE_COOKIE = "coscientist_gdocs_state"

# How long the handshake may take. Long enough to read a consent screen and
# choose an account, short enough that a state cookie left in a shared browser
# is not a standing invitation.
STATE_TTL_SECONDS = 600

# Taken off the token's own lifetime before it is trusted, so an export that
# starts with fifty seconds left does not fail halfway up the wire.
EXPIRY_MARGIN_SECONDS = 60


class GoogleDocsError(RuntimeError):
    """Google refused a step of the handshake or the upload."""


class NotConnected(RuntimeError):
    """This browser holds no usable Drive authorisation."""


def configured() -> bool:
    """Whether this deployment has an OAuth client to authorise against.

    A client id and secret are issued per deployment by whoever owns the Cloud
    project, and cannot be created by this code or by the service account it
    runs as. Unset, every route here still answers -- the browser is told the
    feature is unconfigured and shows the plain .docx download instead, which is
    what a local checkout and every test see.
    """
    return bool(_client_id() and _client_secret())


def _client_id() -> str:
    return os.environ.get("GOOGLE_OAUTH_CLIENT_ID", "").strip()


def _client_secret() -> str:
    return os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET", "").strip()


def redirect_uri(app_url: str) -> str:
    """Where Google sends the reader back, which Google must already know.

    Google matches this string exactly against the list registered on the OAuth
    client, so it is derived from one place and reported by the status route --
    a deployment behind a custom domain sets ``GOOGLE_OAUTH_REDIRECT_URI`` and
    registers the same string, rather than discovering the mismatch as a
    ``redirect_uri_mismatch`` on the first reader who clicks the button.
    """
    override = os.environ.get("GOOGLE_OAUTH_REDIRECT_URI", "").strip()
    if override:
        return override
    return f"{app_url.rstrip('/')}/api/research/google/callback"


def authorize_url(app_url: str, state: str) -> str:
    """The consent screen to send the reader to."""
    query = urlencode(
        {
            "client_id": _client_id(),
            "redirect_uri": redirect_uri(app_url),
            "response_type": "code",
            "scope": SCOPE,
            "state": state,
            # No refresh token, and therefore nothing for this service to keep.
            "access_type": "online",
            # Ask every time rather than reusing a grant silently: the reader is
            # about to have a file appear in their Drive and should be told by
            # Google, not only by this page, that they agreed to it.
            "prompt": "consent",
            "include_granted_scopes": "true",
        }
    )
    return f"{AUTHORIZATION_ENDPOINT}?{query}"


def exchange_code(code: str, app_url: str) -> dict:
    """Trade the one-time code for an access token.

    Returns the sealed cookie value and the seconds it is good for, because
    those are the two things the caller has to put on the response and nothing
    else about the token should travel any further than this function.
    """
    response = httpx.post(
        TOKEN_ENDPOINT,
        data={
            "code": code,
            "client_id": _client_id(),
            "client_secret": _client_secret(),
            "redirect_uri": redirect_uri(app_url),
            "grant_type": "authorization_code",
        },
        timeout=30,
    )
    if response.status_code != 200:
        raise GoogleDocsError(
            f"Google refused the authorization code ({response.status_code})."
        )
    payload = response.json()
    token = payload.get("access_token", "")
    if not token:
        raise GoogleDocsError("Google returned no access token.")
    lifetime = int(payload.get("expires_in", 3600))
    return {
        "cookie": seal({"token": token, "expires_at": int(time.time()) + lifetime}),
        "max_age": lifetime,
    }


def access_token(cookie: str | None) -> str:
    """The living token in a sealed cookie, or a refusal.

    An expired token is treated as no token at all rather than sent to Google
    to be rejected: the answer is the same reconnect either way, and one of the
    two costs a round trip and reads to the reader as a failure.
    """
    payload = unseal(cookie)
    if not payload:
        raise NotConnected("This browser is not connected to Google Drive.")
    if int(payload.get("expires_at", 0)) - EXPIRY_MARGIN_SECONDS <= time.time():
        raise NotConnected("The Google Drive authorization has expired.")
    return str(payload["token"])


def create_document(token: str, name: str, docx: bytes) -> dict:
    """Upload the .docx and have Drive convert it to a Google Doc on the way in.

    A ``multipart/related`` body written by hand rather than through httpx's
    ``files=``: that sends ``multipart/form-data``, which this endpoint answers
    with a 400. Asking for ``mimeType`` of a Google Doc while sending Word bytes
    is what makes Drive convert rather than store -- upload it as its own type
    and the reader gets an attachment they still have to convert themselves,
    which is the thing this exists to stop.
    """
    boundary = f"coscientist{secrets.token_hex(16)}"
    metadata = json.dumps({"name": name, "mimeType": DOCUMENT_MIME})
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            b"Content-Type: application/json; charset=UTF-8\r\n\r\n",
            metadata.encode("utf-8"),
            f"\r\n--{boundary}\r\n".encode(),
            f"Content-Type: {DOCX_MIME}\r\n\r\n".encode(),
            docx,
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    response = httpx.post(
        UPLOAD_ENDPOINT,
        params={"uploadType": "multipart", "fields": "id,name,webViewLink"},
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": f"multipart/related; boundary={boundary}",
        },
        content=body,
        timeout=120,
    )
    if response.status_code in (401, 403):
        raise NotConnected("Google declined the stored Drive authorization.")
    if response.status_code >= 300:
        raise GoogleDocsError(
            f"Google Drive refused the document ({response.status_code})."
        )
    created = response.json()
    return {
        "id": created.get("id", ""),
        "name": created.get("name", name),
        "url": created.get("webViewLink")
        or f"https://docs.google.com/document/d/{created.get('id', '')}/edit",
    }


def new_state(session_id: str) -> str:
    """A signed round-trip marker: which run this was, and that we started it."""
    return seal(
        {
            "session": session_id,
            "nonce": secrets.token_urlsafe(16),
            "expires_at": int(time.time()) + STATE_TTL_SECONDS,
        }
    )


def read_state(state: str | None, cookie: str | None) -> str:
    """The session id a callback belongs to, if this browser really asked for it.

    Google hands ``state`` back in a URL, and a URL is something anybody can
    send anybody. The same value is also set as a cookie when the handshake
    starts, so a callback is only honoured when the two agree -- which a forged
    link cannot arrange, because it cannot write a cookie on this origin.
    """
    payload = unseal(state)
    if not payload or payload.get("expires_at", 0) <= time.time():
        raise GoogleDocsError("The Google sign-in took too long. Try again.")
    if not cookie or not hmac.compare_digest(str(state), str(cookie)):
        raise GoogleDocsError("This Google sign-in did not start in this browser.")
    return str(payload.get("session", ""))


def seal(payload: dict) -> str:
    """Sign a small JSON payload for a cookie this service will read back.

    Unpadded base64, so the value carries no ``=``. A cookie value containing
    one is quoted by the server that sets it and unquoted by whatever parses it
    next, and the two do not always agree -- the value also travels back in a
    URL as the OAuth ``state``, where it is compared byte for byte against the
    cookie. Not emitting the character is cheaper than trusting every layer
    between here and Google to round-trip it identically.
    """
    body = _b64(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    return f"{body}.{_signature(body)}"


def unseal(sealed: str | None) -> dict | None:
    """The payload of a cookie this service signed, or ``None`` for anything else."""
    if not sealed or "." not in sealed:
        return None
    body, _, signature = sealed.rpartition(".")
    if not hmac.compare_digest(signature, _signature(body)):
        return None
    try:
        padding = "=" * (-len(body) % 4)
        return json.loads(base64.urlsafe_b64decode(f"{body}{padding}".encode("ascii")))
    except (ValueError, json.JSONDecodeError):
        return None


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _signature(body: str) -> str:
    return _b64(hmac.new(_signing_key(), body.encode("ascii"), hashlib.sha256).digest())


# Regenerated per process when unset, which is deliberate rather than lazy: the
# only thing it protects is a cookie holding a one-hour token, so the worst a
# restart can do is ask a reader to press Connect again. Set
# ``GOOGLE_OAUTH_STATE_SECRET`` on a deployment running more than one instance,
# or a reader whose callback lands on a different container than started the
# handshake is told their sign-in did not start in this browser.
_EPHEMERAL_KEY = secrets.token_bytes(32)


def _signing_key() -> bytes:
    configured_key = os.environ.get("GOOGLE_OAUTH_STATE_SECRET", "").strip()
    return configured_key.encode("utf-8") if configured_key else _EPHEMERAL_KEY
