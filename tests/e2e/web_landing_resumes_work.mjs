// A run outlives the browser that started it. It is hours of Deep Research on
// a server, and any browser may watch it -- so the landing screen showing an
// empty "New inquiry" form while the deployment had four runs mid-evidence,
// one of them blocked on governance, was the page hiding its own work. This
// checks the two halves of that: a first visit with work in progress opens it,
// and a first visit with nothing running still offers the launcher.
import { spawn } from "node:child_process";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  assert,
  delay,
  openPage,
  refuseOccupiedPort,
  refuseOccupiedServer,
  spawnBrowser,
  waitFor,
  waitForDebugging,
  waitForServer,
} from "./browser.mjs";

const baseUrl = process.env.COSCIENTIST_E2E_URL || "http://127.0.0.1:8768";
const chrome =
  process.env.CHROME_BIN ||
  "/tmp/math-witch-playwright/chromium-1187/chrome-linux/chrome";
const debuggingPort = Number(process.env.CHROME_DEBUGGING_PORT || "9225");
const profile = await mkdtemp(join(tmpdir(), "coscientist-landing-chrome-"));
const startServer = process.env.COSCIENTIST_E2E_START_SERVER !== "false";
let server = null;

if (startServer) {
  await refuseOccupiedServer(baseUrl);
  const parsed = new URL(baseUrl);
  server = spawn(
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
        // Subprocess: no pytest fixture strips its credentials, so the switch
        // has to be thrown here or a browser test bills a research pass.
        COSCIENTIST_DEEP_RESEARCH: "off",
        COSCIENTIST_STATE_DIR: join(profile, "state"),
        UV_CACHE_DIR: "/tmp/coscientist-uv-cache",
      },
      stdio: "ignore",
    },
  );
}

await refuseOccupiedPort(debuggingPort);
const browser = spawnBrowser(chrome, debuggingPort, profile);

// What the landing screen settled on, once it had stopped moving.
const landing = async (cdp) => {
  await waitFor(
    cdp,
    "document.readyState === 'complete' && !!document.querySelector('#composer')",
    "The research workspace did not load.",
    30000,
  );
  // The directory poll, the session fetch and the render behind it.
  await delay(2500);
  return JSON.parse(
    await cdp.evaluate(`(() => {
      const wrap = document.getElementById("composerWrap");
      return JSON.stringify({
        launcher: !wrap.hidden && getComputedStyle(wrap).display !== "none",
        title: document.getElementById("sessionTitle").textContent.trim(),
      });
    })()`),
  );
};

// A browser that has never been here: no remembered run, no session in the
// address. That is the visit the launcher used to win unconditionally.
const firstVisit = async () => {
  const cdp = await openPage(debuggingPort, baseUrl);
  await waitFor(
    cdp,
    "!!document.querySelector('#composer')",
    "The workspace did not load before clearing its memory.",
    30000,
  );
  await cdp.evaluate("localStorage.clear()");
  await cdp.call("Page.navigate", { url: baseUrl });
  return cdp;
};

try {
  await waitForServer(baseUrl);
  await waitForDebugging(debuggingPort);

  const empty = await landing(await firstVisit());
  assert(
    empty.launcher,
    "With nothing running, the landing screen must still offer the launcher.",
  );
  assert(
    empty.title === "New inquiry",
    `An idle deployment should head itself "New inquiry", not "${empty.title}".`,
  );

  const question =
    "Which host factors determine severity of respiratory syncytial virus?";
  const created = await fetch(`${baseUrl}/api/research/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, approval_profile: "milestone" }),
  });
  assert(created.status === 201, `Could not start a run: ${created.status}`);
  const { id } = await created.json();

  const running = await landing(await firstVisit());
  assert(
    !running.launcher,
    "A browser landing while a run is in progress must not be offered an empty form.",
  );
  assert(
    running.title !== "New inquiry" && running.title.length > 0,
    `The landing screen should name the run in progress, not "${running.title}".`,
  );

  console.log(JSON.stringify({ status: "passed", id, empty, running }, null, 2));
} finally {
  browser.kill("SIGTERM");
  if (server) server.kill("SIGTERM");
}
