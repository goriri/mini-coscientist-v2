// The shelf of previous sessions: what order it is in, and how a run is found.
//
// Three things are held here, all of them about the list and none about the
// research in it.
//
//   1. Newest first, by when the research last moved.
//   2. Opening a run is not the run moving. Reading one used to rewrite its
//      rank and send it to the top, so the list reordered itself under the hand
//      that was using it and the run just left was no longer where it was left.
//   3. The drawer shows what fits. With sixty runs on the server, reaching the
//      one you remember by name means being able to ask for it by name.
//
// The clocks are seeded rather than earned -- four runs written in the same
// second cannot demonstrate an ordering -- and the state is offline throughout.
//
//   CHROME_DEBUGGING_PORT=9635 node tests/e2e/web_session_shelf.mjs

import { spawn } from "node:child_process";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  assert,
  openPage,
  refuseOccupiedPort,
  refuseDeploymentTarget,
  refuseOccupiedServer,
  spawnBrowser,
  waitFor,
  waitForDebugging,
  waitForServer,
} from "./browser.mjs";

const baseUrl = process.env.COSCIENTIST_E2E_URL || "http://127.0.0.1:8769";
refuseDeploymentTarget("web_session_shelf", baseUrl);
const chrome =
  process.env.CHROME_BIN ||
  "/tmp/math-witch-playwright/chromium-1187/chrome-linux/chrome";
const debuggingPort = Number(process.env.CHROME_DEBUGGING_PORT || "9226");
const profile = await mkdtemp(join(tmpdir(), "coscientist-chrome-"));
const stateDir = join(profile, "state");

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

const seeded = JSON.parse(
  await run(".venv/bin/python3", ["tests/e2e/seed_session_shelf.py"], {
    COSCIENTIST_STATE_DIR: stateDir,
  }),
);
assert(
  seeded.length === 4 &&
    seeded.every((item) => /^session_[0-9a-f]+$/.test(item.id)),
  `The seed script did not print four session ids (got ${JSON.stringify(seeded)}).`,
);
const expected = seeded.map((item) => item.id);
const battery = seeded.find((item) => item.question.includes("battery")).id;
const perovskite = seeded.find((item) =>
  item.question.includes("perovskite"),
).id;

await refuseOccupiedServer(baseUrl);
const parsed = new URL(baseUrl);
const server = spawn(
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
    },
    stdio: "ignore",
  },
);

await refuseOccupiedPort(debuggingPort);
const browser = spawnBrowser(chrome, debuggingPort, profile);

const drawerOrder = `[...document.querySelectorAll(".session-history-item")].map((node) => node.dataset.sessionId)`;
// Whether the overlay is on the screen, which is not what its `hidden`
// attribute says. `.session-browser` sets a display of its own, and an author
// display beats the one the attribute gets from the browser: the overlay was
// painted over the landing page and every gate from first paint, its close
// button did nothing, and the attribute read correctly throughout.
const overlayShown = `(() => {
  const panel = document.querySelector("#sessionBrowser");
  return (
    getComputedStyle(panel).display !== "none" &&
    panel.getBoundingClientRect().height > 0
  );
})()`;

const browserRows = `(() => {
  const panel = document.querySelector("#sessionBrowser");
  return {
    hidden: panel.hidden,
    count: document.querySelector("#sessionBrowserCount").textContent.trim(),
    ids: [...document.querySelectorAll(".session-browser-item")].map((node) => node.dataset.sessionId),
    empty: !!document.querySelector("#sessionBrowserList .history-empty"),
    heading: document.querySelector("#sessionBrowserList .history-empty strong")?.textContent.trim() || "",
  };
})()`;

const type = (text) => `(() => {
  const input = document.querySelector("#sessionSearch");
  input.value = ${JSON.stringify(text)};
  input.dispatchEvent(new Event("input", { bubbles: true }));
})()`;

try {
  await waitForServer(baseUrl);
  await waitForDebugging(debuggingPort);
  const cdp = await openPage(debuggingPort, baseUrl);
  await waitFor(
    cdp,
    "document.readyState === 'complete' && !!document.querySelector('#composer')",
    "The research workspace did not load.",
  );
  await waitFor(
    cdp,
    "document.querySelectorAll('.session-history-item').length === 4",
    "The four seeded sessions did not reach the history drawer.",
  );
  assert(
    !(await cdp.evaluate(overlayShown)),
    "The session search overlay is on screen before anybody asked for it.",
  );

  // 1. Newest first. The seed wrote them in a fourth order again, so insertion
  // order, identifier order and reverse-of-either all disagree with this.
  let order = await cdp.evaluate(drawerOrder);
  assert(
    JSON.stringify(order) === JSON.stringify(expected),
    `The drawer is not in time-descending order (saw ${JSON.stringify(order)}, wanted ${JSON.stringify(expected)}).`,
  );

  // 2. Opening the oldest run leaves it oldest. February's run does not become
  // today's because somebody read it.
  await cdp.evaluate(
    `document.querySelectorAll(".session-history-item")[3].querySelector(".session-history-open").click()`,
  );
  await waitFor(
    cdp,
    `document.querySelector('.session-history-item.current')?.dataset.sessionId === ${JSON.stringify(perovskite)}`,
    "Clicking the oldest session did not open it.",
  );
  order = await cdp.evaluate(drawerOrder);
  assert(
    JSON.stringify(order) === JSON.stringify(expected),
    `Opening a session reordered the drawer (saw ${JSON.stringify(order)}).`,
  );

  // And it is still not moved after a reload, which is the check that the visit
  // was not written into this browser's stored copy of the list either.
  await cdp.call("Page.reload", { ignoreCache: false });
  await waitFor(
    cdp,
    "document.querySelectorAll('.session-history-item').length === 4",
    "The history drawer did not come back after a reload.",
  );
  order = await cdp.evaluate(drawerOrder);
  assert(
    JSON.stringify(order) === JSON.stringify(expected),
    `A reload put the opened session back on top (saw ${JSON.stringify(order)}).`,
  );

  // 3. Every session, by name. The overlay opens on the same order.
  await cdp.evaluate('document.querySelector("#searchSessionsButton").click()');
  await waitFor(
    cdp,
    "!document.querySelector('#sessionBrowser').hidden",
    "The search button did not open the session browser.",
  );
  assert(
    await cdp.evaluate(overlayShown),
    "The search button set the attribute but drew nothing.",
  );
  let view = await cdp.evaluate(browserRows);
  assert(
    JSON.stringify(view.ids) === JSON.stringify(expected),
    `The browser is not in time-descending order (saw ${JSON.stringify(view.ids)}).`,
  );
  assert(
    view.count === "4 sessions, newest first",
    `The unfiltered count must say what the list is (saw ${JSON.stringify(view.count)}).`,
  );

  // Two words, in the wrong order, from two ends of the question. A single-
  // string match would find this run only if the words were typed adjacent and
  // the right way round, which is not how anybody remembers a question.
  await cdp.evaluate(type("coating battery"));
  view = await cdp.evaluate(browserRows);
  assert(
    JSON.stringify(view.ids) === JSON.stringify([battery]),
    `Searching two words out of order did not find the one run with both (saw ${JSON.stringify(view.ids)}).`,
  );
  assert(
    view.count === "1 of 4 sessions match “coating battery”",
    `A filtered count must say how much of the whole it is (saw ${JSON.stringify(view.count)}).`,
  );

  // The day a run began, which is what is remembered about an old one when its
  // wording is not. February is on one card and no other.
  await cdp.evaluate(type("feb"));
  view = await cdp.evaluate(browserRows);
  assert(
    JSON.stringify(view.ids) === JSON.stringify([perovskite]),
    `Searching by month did not reach the run from February (saw ${JSON.stringify(view.ids)}).`,
  );

  await cdp.evaluate(type("thermoelectric"));
  view = await cdp.evaluate(browserRows);
  assert(
    view.empty && view.heading === "Nothing matches that",
    "A search matching nothing must say so rather than showing an empty box.",
  );

  // Opening a result closes the overlay and shows the run, and the run it shows
  // is the one that was clicked.
  await cdp.evaluate(type("coral"));
  await waitFor(
    cdp,
    "document.querySelectorAll('.session-browser-item').length === 1",
    "The coral run was not reachable by name.",
  );
  await cdp.evaluate('document.querySelector(".session-browser-open").click()');
  await waitFor(
    cdp,
    `document.querySelector('#sessionBrowser').hidden && document.querySelector('.session-history-item.current')?.dataset.sessionId === ${JSON.stringify(expected[0])}`,
    "Opening a search result did not close the overlay onto that session.",
  );

  // Reopened, the query is still there to be corrected rather than retyped, and
  // Escape is the way out.
  await cdp.evaluate('document.querySelector("#searchSessionsButton").click()');
  await waitFor(
    cdp,
    "!document.querySelector('#sessionBrowser').hidden",
    "The session browser did not reopen.",
  );
  await cdp.evaluate(`(() => {
    document.querySelector("#sessionSearch").dispatchEvent(
      new KeyboardEvent("keydown", { key: "Escape", bubbles: true }),
    );
  })()`);
  await waitFor(
    cdp,
    "document.querySelector('#sessionBrowser').hidden",
    "Escape did not close the session browser.",
  );
  assert(
    !(await cdp.evaluate(overlayShown)),
    "Escape set the attribute but left the overlay over the workspace.",
  );

  console.log(
    JSON.stringify(
      {
        status: "passed",
        order: expected,
        searched: ["coating battery", "feb", "coral"],
      },
      null,
      2,
    ),
  );
} finally {
  // SIGTERM, not SIGKILL: the server is a uv wrapper around uvicorn, and a
  // killed wrapper leaves the uvicorn holding the port for the next run.
  browser.kill("SIGTERM");
  server.kill("SIGTERM");
}
