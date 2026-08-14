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
  // The directory poll, the session fetch and the render behind it -- four
  // round trips on the throttled visit, each of them held eight hundred
  // milliseconds, so this waits for the slow one rather than the loopback one.
  await delay(5000);
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
//
// ``latency`` holds the directory answer back. A loopback server replies in
// single-digit milliseconds, which is too fast for the window this test is
// about to exist in -- against the deployment the same answer took between
// three and twelve seconds, and every one of them was spent showing the wrong
// screen. Slowing the request is what makes that window reproducible here.
//
// ``throughput`` separates the stylesheet from the script. Delayed by latency
// alone they arrive together, and the page paints nothing until both are in --
// which skips the frames drawn from styled markup with app.js still in flight,
// the very frames the markup's own ``hidden`` is there for. Metered, the
// smaller stylesheet lands first and those frames happen.
const firstVisit = async ({
  latency = 0,
  throughput = -1,
  url = baseUrl,
  cdp: existing = null,
} = {}) => {
  const cdp = existing || (await openPage(debuggingPort, baseUrl));
  await waitFor(
    cdp,
    "!!document.querySelector('#composer')",
    "The workspace did not load before clearing its memory.",
    30000,
  );
  await cdp.evaluate("localStorage.clear()");
  await cdp.call("Network.enable");
  // A first visit arrives with an empty cache, and that is the visit this
  // models. Left on, the cache handed the second navigation its scripts in
  // forty milliseconds however slow the network was said to be, so the frames
  // this test is about -- the ones drawn from the markup alone, before app.js
  // has been fetched -- never happened.
  await cdp.call("Network.setCacheDisabled", { cacheDisabled: true });
  await cdp.call("Network.emulateNetworkConditions", {
    offline: false,
    latency,
    downloadThroughput: throughput,
    uploadThroughput: -1,
  });
  await cdp.call("Page.navigate", { url });
  return cdp;
};

// Installed before the page's own scripts, in every document this target
// loads, and it samples the launcher every frame from the first one. Polling
// over DevTools cannot do this job: under latency the outgoing page is still
// on screen, so a poll reads the document being left rather than the one being
// tested, and every run of it reported a flash that was not there.
// Eight hundred milliseconds on every request and a hundred kilobytes a
// second through it: the deployment on a phone, near enough.
const SLOW = { latency: 800, throughput: 100 * 1024 };

const RECORD_LAUNCHER = `
  // Cleared here rather than before the navigation, because the page this
  // target is leaving had already resumed the run and was still writing its
  // identifier to storage when the clear went out. The next visit then read a
  // remembered run, took the restore path, and was offered the launcher while
  // it waited -- a flash this test caused and then reported.
  localStorage.clear();
  window.__launcherSeen = false;
  const sample = () => {
    const wrap = document.getElementById("composerWrap");
    if (wrap && !wrap.hidden && getComputedStyle(wrap).display !== "none") {
      if (!window.__launcherSeen) window.__seenAt = Math.round(performance.now());
      window.__launcherSeen = true;
    }
    requestAnimationFrame(sample);
  };
  requestAnimationFrame(sample);
`;

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

  // The launcher must not appear on the way, either. Against the deployment
  // the directory and the run behind it took up to twelve seconds to arrive,
  // and offering "New inquiry" for all of it is the wrong answer held long
  // enough to be typed into.
  const recording = async () => {
    const cdp = await openPage(debuggingPort, baseUrl);
    await cdp.call("Page.enable");
    await cdp.call("Page.addScriptToEvaluateOnNewDocument", {
      source: RECORD_LAUNCHER,
    });
    return cdp;
  };
  const settling = await firstVisit({ ...SLOW, cdp: await recording() });

  // The latency stays on for this read: the run behind the directory is a
  // second request, and the launcher must stay away across both.
  const running = await landing(settling);
  // Every frame this document has drawn, not a sample of some of them. The run
  // in progress keeps the launcher hidden once it lands, so one frame is the
  // flash -- including a frame painted from the markup before app.js has been
  // fetched at all, which is what a slow connection used to get.
  assert(
    !(await settling.evaluate("window.__launcherSeen")),
    `The launcher flashed on screen ${await settling.evaluate("window.__seenAt")}ms into the visit, before the run in progress replaced it.`,
  );
  assert(
    !running.launcher,
    "A browser landing while a run is in progress must not be offered an empty form.",
  );
  assert(
    running.title !== "New inquiry" && running.title.length > 0,
    `The landing screen should name the run in progress, not "${running.title}".`,
  );

  // A link to a run, opened by a browser that has never been here. The same
  // wrong screen by a different route: the run named in the address is a round
  // trip away, and the launcher used to fill the wait -- a second and a half
  // of "New inquiry" for anyone handed a session identifier.
  const linked = await firstVisit({
    ...SLOW,
    url: `${baseUrl}/?session=${id}`,
    cdp: await recording(),
  });
  const shared = await landing(linked);
  assert(
    !(await linked.evaluate("window.__launcherSeen")),
    `A shared link showed the launcher ${await linked.evaluate("window.__seenAt")}ms in, before the run it names arrived.`,
  );
  assert(
    shared.title !== "New inquiry" && shared.title.length > 0,
    `A shared link should open the run it names, not "${shared.title}".`,
  );

  console.log(
    JSON.stringify({ status: "passed", id, empty, running, shared }, null, 2),
  );
} finally {
  browser.kill("SIGTERM");
  if (server) server.kill("SIGTERM");
}
