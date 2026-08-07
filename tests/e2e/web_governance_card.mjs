// The governance card, driven the way a researcher drives it.
//
// A run whose reflect stage records more than one fatal flaw is answered one
// finding at a time. Each answer used to retire the whole card and append a new
// one holding the findings that were left: the answered one vanished, every
// part-typed reason in the others was wiped, and the page jumped. This drives
// three findings through the card and holds that none of that happens.
//
// The state is seeded rather than earned. The offline provider never writes a
// fatal flaw, so before this existed the browser's governance path was only
// ever exercised by live production runs.
//
//   CHROME_DEBUGGING_PORT=9531 node tests/e2e/web_governance_card.mjs

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

const baseUrl = process.env.COSCIENTIST_E2E_URL || "http://127.0.0.1:8767";
const chrome =
  process.env.CHROME_BIN ||
  "/tmp/math-witch-playwright/chromium-1187/chrome-linux/chrome";
const debuggingPort = Number(process.env.CHROME_DEBUGGING_PORT || "9224");
const profile = await mkdtemp(join(tmpdir(), "coscientist-chrome-"));
const stateDir = join(profile, "state");

const NAME = "Dr. Ada Lovelace";
const REASON =
  "Confirmed against the binder datasheet; the protocol cannot be run safely.";

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

const sessionId = await run(
  ".venv/bin/python3",
  ["tests/e2e/seed_governance_block.py"],
  { COSCIENTIST_STATE_DIR: stateDir },
);
assert(
  /^session_[0-9a-f]+$/.test(sessionId),
  `The seed script did not print a session id (got ${JSON.stringify(sessionId)}).`,
);

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

const findingsState = `(() => {
  const card = document.querySelector(".approval-card:not(.resolved)");
  if (!card) return null;
  const open = [...card.querySelectorAll(".governance-finding:not(.settled)")];
  const settled = [...card.querySelectorAll(".governance-finding.settled")];
  return {
    stamped: card.dataset.e2eGovernanceCard === "1",
    liveCards: document.querySelectorAll(".approval-card:not(.resolved)").length,
    open: open.map((node) => ({
      reviewId: node.dataset.reviewId,
      title: node.querySelector("strong").textContent.trim(),
      name: node.querySelector(".governance-adjudicator").value,
      reason: node.querySelector(".governance-justification").value,
      hasCopy: !!node.querySelector('[data-decision="copy_reason"]'),
    })),
    settled: settled.map((node) => ({
      reviewId: node.dataset.reviewId,
      verdict: node.querySelector(".governance-verdict").textContent.trim(),
      resolution: node.querySelector(".governance-resolution").textContent.trim(),
      title: node.querySelector(".governance-settled-title").textContent.trim(),
    })),
    acceptDisabled: !!card.querySelector('[data-decision="accept"]:disabled'),
    blockedReason: card.querySelector(".gate-blocked")?.textContent.trim() || "",
  };
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
  await cdp.evaluate(
    `openResearchSession(${JSON.stringify(sessionId)}, { restore: true })`,
  );
  await waitFor(
    cdp,
    "document.querySelectorAll('.governance-finding:not(.settled)').length === 3",
    "The seeded session's three governance findings did not reach the card.",
  );

  // Stamped so every later check can tell "the same card, re-rendered" from "a
  // new card holding whatever was left".
  await cdp.evaluate(
    'document.querySelector(".approval-card:not(.resolved)").dataset.e2eGovernanceCard = "1"',
  );

  let view = await cdp.evaluate(findingsState);
  assert(
    view.acceptDisabled,
    "The primary must stay disabled while a fatal finding is unanswered.",
  );
  assert(
    /3 safety findings unanswered/.test(view.blockedReason),
    `A disabled primary must say why it is disabled (saw ${JSON.stringify(view.blockedReason)}).`,
  );
  assert(
    view.open.every((item) => item.hasCopy),
    "With more than one finding open, each must offer to copy its reason to the rest.",
  );
  assert(
    view.open[0].title.includes("400 C"),
    "A finding must name the hypothesis it is about, not its identifier.",
  );

  // One reason, typed once. Four findings on a live run were four objections to
  // the same hazard and every one had to be retyped in full.
  const firstId = view.open[0].reviewId;
  await cdp.evaluate(`(() => {
    const finding = document.querySelector('[data-review-id="${firstId}"]');
    finding.querySelector(".governance-adjudicator").value = ${JSON.stringify(NAME)};
    finding.querySelector(".governance-justification").value = ${JSON.stringify(REASON)};
    finding.querySelector('[data-decision="copy_reason"]').click();
  })()`);
  view = await cdp.evaluate(findingsState);
  assert(
    view.open.length === 3 &&
      view.open.every(
        (item) => item.reason === REASON && item.name === NAME,
      ),
    "Copying a reason must fill every open finding, ready to edit.",
  );

  // An edit to one copy must not follow into the others: each finding still
  // posts its own reason, which is the whole reason this is a copy.
  const secondId = view.open[1].reviewId;
  const edited = `${REASON} The vent path here is the additional problem.`;
  await cdp.evaluate(
    `document.querySelector('[data-review-id="${secondId}"] .governance-justification').value = ${JSON.stringify(edited)}`,
  );

  // Answer the first, and hold everything the rebuild used to destroy.
  await cdp.evaluate(`(() => {
    const finding = document.querySelector('[data-review-id="${firstId}"]');
    finding.querySelector('[data-decision="withdraw_hypothesis"]').click();
  })()`);
  await waitFor(
    cdp,
    "document.querySelectorAll('.governance-finding.settled').length === 1",
    "Answering a finding did not settle it in place on the card.",
  );
  view = await cdp.evaluate(findingsState);
  assert(
    view.stamped,
    "The card that answered a finding was replaced instead of re-rendered.",
  );
  assert(
    view.liveCards === 1,
    `Answering a finding left ${view.liveCards} live approval cards.`,
  );
  assert(
    view.settled.length === 1 && view.settled[0].reviewId === firstId,
    "The answered finding must stay on the card as a resolved row.",
  );
  assert(
    view.settled[0].verdict === "Withdrawn" &&
      view.settled[0].resolution.includes(NAME) &&
      view.settled[0].resolution.includes(REASON),
    "A resolved row must show the verdict, the adjudicator and the reason.",
  );
  assert(
    view.settled[0].title.includes("400 C"),
    "A withdrawn hypothesis must keep its title after the population is rewritten.",
  );
  assert(
    view.open.length === 2,
    "The findings still to answer must remain on the same card.",
  );
  assert(
    view.open[0].reason === edited,
    "An edited reason must survive answering another finding.",
  );
  assert(
    view.open[1].reason === REASON && view.open[1].name === NAME,
    "A copied reason must survive answering another finding.",
  );
  assert(
    /2 safety findings unanswered/.test(view.blockedReason),
    `The blocked reason must count down (saw ${JSON.stringify(view.blockedReason)}).`,
  );

  // Overriding needs a confirmation CDP cannot answer.
  const remaining = view.open.map((item) => item.reviewId);
  await cdp.evaluate(`(() => {
    const ask = window.confirm;
    window.confirm = () => true;
    document.querySelector('[data-review-id="${remaining[0]}"] [data-decision="override_governance"]').click();
    window.confirm = ask;
  })()`);
  await waitFor(
    cdp,
    "document.querySelectorAll('.governance-finding.settled').length === 2",
    `Finding ${remaining[0]} could not be overridden.`,
  );
  view = await cdp.evaluate(findingsState);
  assert(view.stamped && view.liveCards === 1, "The card was replaced again.");
  assert(
    view.settled[1].verdict === "Override recorded" &&
      view.settled[1].resolution.includes("vent path here"),
    "Each finding posts the reason typed against it, not a shared one.",
  );
  assert(
    view.open.length === 1 && !view.open[0].hasCopy,
    "With one finding left there is nothing to copy a reason into.",
  );

  await cdp.evaluate(`(() => {
    const ask = window.confirm;
    window.confirm = () => true;
    document.querySelector('[data-review-id="${remaining[1]}"] [data-decision="override_governance"]').click();
    window.confirm = ask;
  })()`);

  // The last answer clears the block, and only then does the card give way.
  await waitFor(
    cdp,
    "!document.querySelector('.governance-finding')",
    "Clearing the last finding did not release the gate.",
  );
  const cleared = await cdp.evaluate(`(async () => {
    const workflow = await fetch("/api/research/sessions/${sessionId}").then((r) => r.json());
    return { status: workflow.status, findings: workflow.governance_blockers.length };
  })()`);
  assert(
    cleared.status !== "governance_blocked" && cleared.findings === 0,
    `The session stayed blocked after every finding was answered (${JSON.stringify(cleared)}).`,
  );

  console.log(
    JSON.stringify(
      { status: "passed", sessionId, sessionStatus: cleared.status },
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
