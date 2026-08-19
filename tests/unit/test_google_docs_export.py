"""Publishing the finished dossier into the reader's own Google Drive.

The export button labelled "Google Docs" produced a .docx and stopped there:
the reader downloaded a file and did the rest themselves. These cover the half
that was missing -- the handshake, the upload, and what each of them refuses.
"""

from __future__ import annotations

import json
import pathlib
import re
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from app import google_docs, research_api


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "client.apps.googleusercontent.com")
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_SECRET", "secret")
    monkeypatch.setenv("GOOGLE_OAUTH_STATE_SECRET", "a-key-for-this-test")
    monkeypatch.setenv("APP_URL", "https://coscientist.example.run.app")
    # Over TLS, because the cookies this sets are Secure whenever the service
    # is deployed behind one -- and a Secure cookie asked for over plain HTTP is
    # dropped by the client without a word, which is what the handshake failing
    # for no visible reason looks like.
    return TestClient(_app(), base_url="https://testserver", follow_redirects=False)


def _app():
    from fastapi import FastAPI

    application = FastAPI()
    application.include_router(research_api.router)
    return application


def _connected_cookie(seconds: int = 3600) -> str:
    return google_docs.seal(
        {"token": "ya29.test-token", "expires_at": int(time.time()) + seconds}
    )


class _Dossier:
    """The one thing the export route wants from a finished run."""

    done = True

    def render_report(self) -> str:
        return "# Coated cathodes\n\nA finding.\n"

    def report_filename(self) -> str:
        return "coated-cathodes.md"


def test_a_deployment_with_no_oauth_client_says_so_rather_than_offering_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Which is every local checkout and every test, and the browser drops the
    button on this answer instead of showing one that fails when pressed. The
    client id is issued per Cloud project by a person; nothing here can make
    one, so "unconfigured" has to be a state this serves rather than a crash."""
    monkeypatch.delenv("GOOGLE_OAUTH_CLIENT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_OAUTH_CLIENT_SECRET", raising=False)
    with TestClient(_app()) as bare:
        status = bare.get("/api/research/google/status").json()
        refused = bare.get("/api/research/google/authorize?session_id=session_1")

    assert status["configured"] is False
    assert status["connected"] is False
    assert refused.status_code == 503


def test_the_status_reports_the_exact_redirect_the_client_must_register(
    client: TestClient,
) -> None:
    """Google matches this string character for character against the list on
    the OAuth client, and a mismatch is a redirect_uri_mismatch on the first
    reader who presses the button rather than anything visible at deploy time.
    So the service says what it will send, and whoever registers it copies."""
    status = client.get("/api/research/google/status").json()

    assert status["configured"] is True
    assert (
        status["redirect_uri"]
        == "https://coscientist.example.run.app/api/research/google/callback"
    )


def test_the_consent_asks_for_one_file_and_no_refresh_token(
    client: TestClient,
) -> None:
    """Two limits on what this deployment can hold, both set at this moment and
    nowhere else. ``drive.file`` is the only scope that can create a document
    without also being able to read every other document the reader owns, and
    ``access_type=online`` means Google issues an hour-long token and no refresh
    token -- so there is no long-lived credential here to leak or revoke."""
    sent = client.get("/api/research/google/authorize?session_id=session_abc")
    destination = httpx.URL(sent.headers["location"])

    assert sent.status_code == 303
    assert destination.host == "accounts.google.com"
    assert destination.params["scope"] == "https://www.googleapis.com/auth/drive.file"
    assert destination.params["access_type"] == "online"
    assert "refresh" not in destination.params.get("response_type", "code")
    # And the browser is given the same state it will have to hand back.
    assert client.cookies.get(google_docs.STATE_COOKIE) == destination.params["state"]


def test_a_callback_that_did_not_start_in_this_browser_is_refused(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``state`` comes back in a URL and a URL is something anybody can send
    anybody, so a link alone must not be able to attach a Drive authorization
    to somebody else's browser. The same value is set as a cookie when the
    handshake starts, which a forged link cannot write on this origin.

    The refusal has to happen before the code is redeemed, not after: a
    handshake that reaches Google and is then discarded has still spent the
    code, and "it failed anyway" is an accident of the network rather than the
    check this is about.
    """

    def _never(*args, **kwargs):
        raise AssertionError("a forged callback reached Google's token endpoint")

    monkeypatch.setattr(google_docs.httpx, "post", _never)
    forged = google_docs.new_state("session_abc")

    landed = client.get(f"/api/research/google/callback?code=x&state={forged}")

    assert landed.status_code == 303
    assert "did+not+start+in+this+browser" in landed.headers["location"]
    assert client.cookies.get(google_docs.TOKEN_COOKIE) is None


def test_declining_at_google_lands_back_on_the_run_being_read(
    client: TestClient,
) -> None:
    """Pressing Cancel is an answer, not an error page. A reader who is an hour
    into a dossier and changes their mind about Drive has to come back to the
    dossier, with the reason said out loud rather than a bare API response."""
    started = client.get("/api/research/google/authorize?session_id=session_abc")
    state = httpx.URL(started.headers["location"]).params["state"]

    landed = client.get(
        f"/api/research/google/callback?error=access_denied&state={state}"
    )

    assert landed.status_code == 303
    assert "session=session_abc" in landed.headers["location"]
    assert (
        "google_error=Google+Drive+access+was+not+granted"
        in (landed.headers["location"])
    )


def test_a_completed_handshake_comes_back_ready_to_finish_the_export(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One press, not two. The reader pressed a button that turned out to need
    consent; coming back to the report and being asked to press it again is the
    application forgetting what it was doing."""
    monkeypatch.setattr(
        google_docs.httpx,
        "post",
        lambda *args, **kwargs: httpx.Response(
            200, json={"access_token": "ya29.fresh", "expires_in": 3599}
        ),
    )
    started = client.get("/api/research/google/authorize?session_id=session_abc")
    state = httpx.URL(started.headers["location"]).params["state"]

    landed = client.get(f"/api/research/google/callback?code=auth-code&state={state}")

    assert landed.status_code == 303
    assert "session=session_abc" in landed.headers["location"]
    assert "google_doc=1" in landed.headers["location"]
    assert google_docs.unseal(client.cookies[google_docs.TOKEN_COOKIE])["token"] == (
        "ya29.fresh"
    )


def test_the_upload_asks_drive_to_convert_rather_than_to_store(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point of the whole path. Uploading Word bytes under the Word type
    puts an attachment in the reader's Drive that they still have to convert by
    hand, which is the thing the old button already did. Naming the Google Doc
    type in the metadata while sending Word bytes is what makes Drive convert
    on the way in -- and the body has to be ``multipart/related``, because
    httpx's ``files=`` sends ``multipart/form-data`` and this endpoint answers
    that with a 400."""
    captured: dict = {}

    def _upload(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return httpx.Response(
            200,
            json={
                "id": "1AbC",
                "name": "coated-cathodes",
                "webViewLink": "https://docs.google.com/document/d/1AbC/edit",
            },
        )

    monkeypatch.setattr(google_docs.httpx, "post", _upload)
    monkeypatch.setattr(research_api, "_load", lambda session_id: _Dossier())
    client.cookies.set(google_docs.TOKEN_COOKIE, _connected_cookie())

    created = client.post("/api/research/sessions/session_abc/report/google-doc")

    assert created.status_code == 200, created.text
    assert created.json()["url"] == "https://docs.google.com/document/d/1AbC/edit"
    assert captured["headers"]["Authorization"] == "Bearer ya29.test-token"
    assert captured["headers"]["Content-Type"].startswith(
        "multipart/related; boundary="
    )
    body = captured["content"]
    metadata = json.loads(body.split(b"\r\n\r\n", 2)[1].split(b"\r\n--", 1)[0])
    assert metadata == {
        "name": "coated-cathodes",
        "mimeType": "application/vnd.google-apps.document",
    }
    # And what was uploaded is the rendered document, not the markdown behind it.
    assert b"word/document.xml" in body
    assert b"# Coated cathodes" not in body


def test_an_hour_later_the_reader_is_asked_to_connect_again_not_told_it_failed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The token is good for about an hour and these reports are read for
    longer than that. An expired one is not sent to Google to be rejected: the
    answer is the same reconnect either way, and one of the two costs a round
    trip and reads to the reader as a fault."""
    monkeypatch.setattr(research_api, "_load", lambda session_id: _Dossier())
    client.cookies.set(google_docs.TOKEN_COOKIE, _connected_cookie(seconds=30))

    refused = client.post("/api/research/sessions/session_abc/report/google-doc")

    assert refused.status_code == 401
    assert "expired" in refused.json()["detail"]


def test_a_cookie_this_service_did_not_sign_is_not_a_connection(
    client: TestClient,
) -> None:
    """Otherwise the cookie is a bearer token anybody can write, and the
    "connected" the browser reads off the status route means nothing."""
    forged = google_docs.seal({"token": "stolen", "expires_at": 2**31})
    tampered = forged[:-4] + ("aaaa" if not forged.endswith("aaaa") else "bbbb")
    client.cookies.set(google_docs.TOKEN_COOKIE, tampered)

    status = client.get("/api/research/google/status").json()

    assert status["connected"] is False


def test_the_word_download_is_labelled_as_a_word_download() -> None:
    """It was labelled "Google Docs (.docx)", which named where a reader might
    take the file rather than what the button produced -- and there is now a
    button beside it that really does mean Google Docs, so the two would have
    carried the same name. Read out of the source rather than off a snapshot,
    because a snapshot with exports in it is a run driven to a finished dossier
    and that is a workflow test, not this one."""
    source = pathlib.Path(research_api.__file__).read_text()

    label = re.search(r'\("docx", "(?P<label>[^"]+)", "\.docx"\)', source)

    assert label, "the report export table is no longer a literal"
    assert label.group("label") == "Word (.docx)"
