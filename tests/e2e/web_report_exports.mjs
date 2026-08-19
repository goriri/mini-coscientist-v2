// What a finished dossier offers, and what it offers when there is nowhere to
// send it.
//
// One button used to say "Google Docs" and download a .docx: it named where a
// reader might take the file rather than what pressing it did, and taking it
// there was still four manual steps. There are two buttons now -- the document
// in their Drive, and the Word file -- and this holds that both are on the page
// and that the second is labelled as itself.
//
// The deployment may have no OAuth client, which is every local checkout, so
// the same page is driven twice against two servers: with a client configured
// and without one. Neither run reaches Google. What happens after the button is
// pressed is the server's half and is covered in tests/unit.
//
//   CHROME_DEBUGGING_PORT=9637 node tests/e2e/web_report_exports.mjs

import { spawn } from "node:child_process";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  assert,
  openPage,
  refuseOccupiedPort,
  refuseOccupiedServer,
  spawnBrowser,
  waitFor,
  waitForDebugging,
  waitForServer,
} from "./browser.mjs";

const baseUrl = process.env.COSCIENTIST_E2E_URL || "http://127.0.0.1:8771";
const chrome =
  process.env.CHROME_BIN ||
  "/tmp/math-witch-playwright/chromium-1187/chrome-linux/chrome";
const debuggingPort = Number(process.env.CHROME_DEBUGGING_PORT || "9228");
const profile = await mkdtemp(join(tmpdir(), "coscientist-chrome-"));
const stateDir = join(profile, "state");
const parsed = new URL(baseUrl);

function run(command, args, env) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: process.cwd(),
      env: { ...process.env, ...env },
      stdio: ["ignore", "pipe", "pipe"],
    });
    let out = "";
    let err = "";
    child.stdout.on("data", (chunk) => (out += chunk));
    child.stderr.on("data", (chunk) => (err += chunk));
    child.on("close", (code) =>
      code === 0
        ? resolve(out.trim())
        : reject(new Error(`${command} failed (${code}): ${err.trim()}`)),
    );
  });
}

function serve(extra) {
  return spawn(
    "./.tools/bin/uv",
    [
      "run",
      "uvicorn",
      "app.fast_api_app:app",
      "--host",
      parsed.hostname,
      "--port",
      parsed.port || "80",
    ],
    {
      cwd: process.cwd(),
      env: {
        ...process.env,
        APP_URL: baseUrl,
        INTEGRATION_TEST: "TRUE",
        COSCIENTIST_DEEP_RESEARCH: "off",
        COSCIENTIST_STATE_DIR: stateDir,
        UV_CACHE_DIR: "/tmp/coscientist-uv-cache",
        ...extra,
      },
      stdio: "ignore",
    },
  );
}

const sessionId = await run(
  ".venv/bin/python3",
  ["tests/e2e/seed_finished_report.py"],
  { COSCIENTIST_STATE_DIR: stateDir },
);
assert(
  /^session_[0-9a-f]+$/.test(sessionId),
  `The seed script did not print a session id (got ${JSON.stringify(sessionId)}).`,
);

const exportRow = `(() => {
  const row = document.querySelector(".report-export-actions");
  if (!row) return null;
  return {
    actions: [...row.children].map((node) => ({
      tag: node.tagName,
      label: node.textContent.trim(),
      download: node.getAttribute("download") || "",
      sessionId: node.dataset.sessionId || "",
    })),
    note: document.querySelector(".google-doc-note")?.textContent.trim() || "",
  };
})()`;

await refuseOccupiedPort(debuggingPort);
const browser = spawnBrowser(chrome, debuggingPort, profile);
let server = null;

async function readTheExportRow(extra) {
  await refuseOccupiedServer(baseUrl);
  server = serve(extra);
  await waitForServer(baseUrl);
  const cdp = await openPage(debuggingPort, baseUrl);
  await waitFor(
    cdp,
    "document.readyState === 'complete' && !!document.querySelector('#composer')",
    "The research workspace did not load.",
  );
  await cdp.evaluate(
    `openResearchSession(${JSON.stringify(sessionId)}, { restore: true })`,
  );
  await waitFor(
    cdp,
    "!!document.querySelector('.report-export-actions')",
    "The finished dossier showed no export actions.",
  );
  // The Google half of the row is settled by a round trip, so the row is read
  // once that has landed rather than in whatever state it was first painted in.
  await waitFor(
    cdp,
    "document.querySelector('.google-doc-note')?.textContent.trim().length > 0",
    "The Google Docs button never resolved whether it was available.",
  );
  const view = await cdp.evaluate(exportRow);
  server.kill("SIGTERM");
  server = null;
  await new Promise((resolve) => setTimeout(resolve, 1500));
  return view;
}

try {
  await waitForDebugging(debuggingPort);

  // With a client: the document and the file are two separate buttons, and the
  // Word one is called Word.
  const configured = await readTheExportRow({
    GOOGLE_OAUTH_CLIENT_ID: "e2e.apps.googleusercontent.com",
    GOOGLE_OAUTH_CLIENT_SECRET: "e2e-secret",
    GOOGLE_OAUTH_STATE_SECRET: "e2e-state-secret",
  });
  assert(
    configured.actions.length === 4,
    `The export row should offer four things (saw ${JSON.stringify(configured.actions)}).`,
  );
  const [drive, ...downloads] = configured.actions;
  assert(
    drive.tag === "BUTTON" && drive.sessionId === sessionId,
    "The Drive action must be a button that knows which run it is exporting.",
  );
  assert(
    drive.label === "Connect Google Drive and open",
    `An unconnected browser is told what pressing it will do (saw ${JSON.stringify(drive.label)}).`,
  );
  assert(
    downloads.map((item) => item.label).join(" · ") ===
      "Word (.docx) · PDF · Markdown",
    `The downloads name the file they produce (saw ${JSON.stringify(downloads.map((item) => item.label))}).`,
  );
  assert(
    downloads.every((item) => item.tag === "A" && item.download),
    "Each download must be a link that downloads rather than navigates.",
  );
  assert(
    /create one file in your Drive/.test(configured.note),
    `The reader is told what they are about to grant (saw ${JSON.stringify(configured.note)}).`,
  );

  // Without one: the button is gone rather than present and broken, and the row
  // still offers all three files.
  const bare = await readTheExportRow({
    GOOGLE_OAUTH_CLIENT_ID: "",
    GOOGLE_OAUTH_CLIENT_SECRET: "",
  });
  assert(
    bare.actions.length === 3 && bare.actions.every((item) => item.tag === "A"),
    `A deployment with no OAuth client offers only the downloads (saw ${JSON.stringify(bare.actions)}).`,
  );
  assert(
    bare.actions[0].label === "Word (.docx)",
    "The Word download keeps its own name whether or not Drive is available.",
  );
  assert(
    /no Google OAuth client/.test(bare.note),
    `An unavailable export says why (saw ${JSON.stringify(bare.note)}).`,
  );

  console.log(
    JSON.stringify(
      {
        status: "passed",
        sessionId,
        configured: configured.actions.map((item) => item.label),
        unconfigured: bare.actions.map((item) => item.label),
      },
      null,
      2,
    ),
  );
} finally {
  browser.kill("SIGTERM");
  if (server) server.kill("SIGTERM");
}
