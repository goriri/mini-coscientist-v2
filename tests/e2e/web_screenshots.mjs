// Captures the workspace at the points a researcher actually looks at it: the
// empty landing, a live gate, and the finished dossier, at desktop and phone
// widths. The HITL flow test asserts behaviour; this one exists so the layout
// and the type can be read by eye before a deploy.
import { spawn } from "node:child_process";
import { mkdtemp, writeFile, mkdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  delay,
  openPage,
  waitFor as waitForShared,
  waitForDebugging,
  waitForServer,
} from "./browser.mjs";

const baseUrl = process.env.COSCIENTIST_E2E_URL || "http://127.0.0.1:8768";
const outputDir = process.env.COSCIENTIST_SHOT_DIR || "/tmp/coscientist-shots";
const chrome =
  process.env.CHROME_BIN ||
  "/tmp/math-witch-playwright/chromium-1187/chrome-linux/chrome";
const debuggingPort = Number(process.env.CHROME_DEBUGGING_PORT || "9224");
const profile = await mkdtemp(join(tmpdir(), "coscientist-shots-"));
const startServer = process.env.COSCIENTIST_E2E_START_SERVER !== "false";
// Thirty seconds is right for the offline server this script starts itself.
// Pointed at a deployment with Deep Research on, every stage is a real model
// call, so the waits are stretched by COSCIENTIST_E2E_TIMEOUT_SCALE -- the same
// knob every other suite takes, applied inside the shared wait. This script
// read a knob of its own that nothing else set, so a deployment run configured
// the way the others are kept its local thirty seconds and gave up on the first
// gate, reported as "The first approval gate never opened".
const gateTimeout = 30000;
// Against a deployment the evidence stage is a real Deep Research pass -- seven
// concurrent interactions, most of an hour -- and a fifteen-minute budget on
// gate two reported "Gate 2 never became actionable" over a stage that was
// working exactly as designed. Naming an earlier run of the same question forks
// its corpus and skips the stage, which is what the HITL suite does. The cost
// is the two evidence photographs; the run says which ones it did not take
// rather than printing a shorter list as though it were the whole set.
const seedEvidenceFrom = process.env.COSCIENTIST_E2E_SEED_FROM || "";
let server = null;

await mkdir(outputDir, { recursive: true });

if (startServer) {
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
        COSCIENTIST_DEEP_RESEARCH: "off",
        COSCIENTIST_STATE_DIR: join(profile, "state"),
        UV_CACHE_DIR: "/tmp/coscientist-uv-cache",
      },
      stdio: "ignore",
    },
  );
}

const browser = spawn(
  chrome,
  [
    "--headless=new",
    "--no-sandbox",
    "--disable-gpu",
    "--force-device-scale-factor=2",
    `--remote-debugging-port=${debuggingPort}`,
    `--user-data-dir=${profile}`,
    "about:blank",
  ],
  { stdio: "ignore" },
);

// The gate budget is the only thing this script needs on top of the shared
// wait, which carries the per-call deadline, the retry around a single failed
// evaluation and the narration. A copy of that plumbing lived here and had
// none of the three: one transient DevTools "Internal error", seventeen minutes
// into a deployment run, killed the whole pass.
function waitFor(cdp, expression, note, timeout = gateTimeout) {
  return waitForShared(cdp, expression, note, timeout);
}

async function click(cdp, selector) {
  // The selectors carry their own double quotes, so they are encoded rather
  // than pasted into a quoted string. The card being answered is marked, so
  // settle() can wait on that card rather than on a selector: every gate offers
  // an accept, and offline the next one opens inside a single poll, so "the
  // accept button is gone" is never true for long enough to observe.
  const query = JSON.stringify(`.approval-card:not(.resolved) ${selector}`);
  for (let attempt = 0; attempt < 20; attempt += 1) {
    const stageKey = await cdp.evaluate(
      `(() => {
        const button = document.querySelector(${query});
        if (!button) return null;
        const card = button.closest(".approval-card");
        document
          .querySelectorAll("[data-e2e-answering]")
          .forEach((node) => node.removeAttribute("data-e2e-answering"));
        card.setAttribute("data-e2e-answering", "1");
        button.click();
        return card.dataset.stageKey || "";
      })()`,
    );
    if (stageKey !== null) return stageKey;
    await delay(200);
  }
  throw new Error(`No live gate offered ${selector}.`);
}

async function settle(cdp, stageKey) {
  // A decision reaches the server before the next poll repaints, so the card
  // just answered stays on screen for a beat. Without this the loop re-enters
  // on the stale gate and clicks a button that vanishes mid-retry.
  //
  // The card retires itself the moment the decision is recorded, which is the
  // one signal that means what this wait needs it to mean. Waiting on the
  // buttons instead let the loop run a gate ahead of the page: it read the
  // exploratory offer on a card the server had already accepted, spent a second
  // photographing it, and then found nothing left to click.
  //
  // A refusal also settles, and one gate is designed to produce one: the
  // evidence floor is evaluated when accept is pressed, not when the draft is
  // drawn, so the first press at the evidence gate comes back with the
  // shortfall and the offer of the exploratory route, on a card that stays
  // live. The stage does not move and is not supposed to; the loop reads the
  // offer on its next pass.
  await waitFor(
    cdp,
    `(() => {
      const card = document.querySelector("[data-e2e-answering]");
      if (!card || card.classList.contains("resolved")) return true;
      const fallback = card.querySelector('[data-decision="exploratory_evidence"]');
      return !!fallback && !fallback.disabled;
    })()`,
    `The gate for ${stageKey || "an unidentified stage"} never cleared.`,
  );
}

async function shoot(cdp, name) {
  const { data } = await cdp.call("Page.captureScreenshot", {
    format: "png",
    fromSurface: true,
    captureBeyondViewport: false,
  });
  const path = join(outputDir, `${name}.png`);
  await writeFile(path, Buffer.from(data, "base64"));
  return path;
}

const shots = [];
let rankingShot = false;

try {
  await waitForServer(baseUrl);
  await waitForDebugging(debuggingPort);
  const cdp = await openPage(debuggingPort, baseUrl);
  await cdp.call("Emulation.setDeviceMetricsOverride", {
    width: 1512,
    height: 950,
    deviceScaleFactor: 2,
    mobile: false,
  });
  await waitFor(
    cdp,
    "document.readyState === 'complete' && !!document.querySelector('#composer')",
    "The workspace did not load.",
  );
  await delay(600);
  shots.push(await shoot(cdp, "01-landing-desktop"));

  // The fork is refused unless the question matches the source run's word for
  // word, which the one typed below is.
  if (seedEvidenceFrom) {
    console.log(
      `Forking the evidence base of ${seedEvidenceFrom}: the evidence stage is ` +
        "skipped, so 04-evidence-trust and 04-evidence-integrity-gate are not " +
        "photographed on this pass.",
    );
    await cdp.evaluate(`(() => {
      const field = document.querySelector("#seedEvidenceFrom");
      field.value = ${JSON.stringify(seedEvidenceFrom)};
      field.dispatchEvent(new Event("input", { bubbles: true }));
    })()`);
  }
  await cdp.evaluate(`(() => {
    const input = document.querySelector("#promptInput");
    input.value = "Does a protective coating improve rechargeable battery cycle life compared with an uncoated control?";
    input.dispatchEvent(new Event("input", { bubbles: true }));
    document.querySelector("#composer").requestSubmit();
  })()`);
  await waitFor(
    cdp,
    "!!document.querySelector('.approval-card:not(.resolved) .workflow-progress')",
    "The running-stage card never appeared.",
    8000,
  );
  await delay(400);
  shots.push(await shoot(cdp, "02-stage-running"));

  await waitFor(
    cdp,
    "!!document.querySelector('.approval-card:not(.resolved) [data-decision=\"toggle_edit\"]')",
    "The first approval gate never opened.",
  );
  await delay(400);
  shots.push(await shoot(cdp, "03-approval-gate"));

  // One pass per gate, plus room for the refusal the evidence floor is there to
  // give. Six was the count of gates a run photographed before any of them
  // refused anything, and it ran out halfway.
  for (let gate = 0; gate < 14; gate += 1) {
    if (await cdp.evaluate("!!document.querySelector('.report-completion')")) {
      break;
    }
    await waitFor(
      cdp,
      "!!document.querySelector('.approval-card:not(.resolved) [data-decision=\"accept\"]:not(:disabled)') || !!document.querySelector('.approval-card:not(.resolved) [data-decision=\"exploratory_evidence\"]')",
      `Gate ${gate + 1} never became actionable.`,
    );
    if (
      await cdp.evaluate(
        "!!document.querySelector('.approval-card:not(.resolved) [data-decision=\"exploratory_evidence\"]')",
      )
    ) {
      // The trust assessment sits above the gate it explains, so the gate
      // screenshot alone never showed the thing the researcher is deciding on.
      if (await cdp.evaluate("!!document.querySelector('.evidence-trust')")) {
        await cdp.evaluate(
          "document.querySelector('.evidence-trust').scrollIntoView({block: 'start'})",
        );
        await delay(300);
        shots.push(await shoot(cdp, "04-evidence-trust"));
        await cdp.evaluate(
          "document.querySelector('.approval-card:not(.resolved)').scrollIntoView({block: 'center'})",
        );
      }
      await delay(300);
      shots.push(await shoot(cdp, "04-evidence-integrity-gate"));
      await settle(
        cdp,
        await click(cdp, '[data-decision="exploratory_evidence"]'),
      );
      continue;
    }
    // Once, at the gate that produces it. The table stays in the transcript
    // afterwards, so every later gate re-took this shot over the top of the one
    // before it and the file ended up being the meta-review screen under the
    // ranking screen's name.
    if (
      !rankingShot &&
      (await cdp.evaluate("!!document.querySelector('.ranking-table')"))
    ) {
      rankingShot = true;
      await cdp.evaluate(
        "document.querySelector('.ranking-table').scrollIntoView({block: 'center'})",
      );
      await delay(300);
      shots.push(await shoot(cdp, "05-ranking-presentation"));
    }
    await settle(
      cdp,
      await click(cdp, '[data-decision="accept"]:not(:disabled)'),
    );
    await delay(400);
  }

  await waitFor(
    cdp,
    "!!document.querySelector('.report-completion')",
    "The dossier never completed.",
    Math.max(90000, gateTimeout),
  );
  await delay(600);
  shots.push(await shoot(cdp, "06-dossier-complete"));
  await cdp.evaluate(
    "document.querySelector('.conversation').scrollTop = document.querySelector('.conversation').scrollHeight * 0.4",
  );
  await delay(400);
  shots.push(await shoot(cdp, "07-dossier-prose"));

  await cdp.call("Emulation.setDeviceMetricsOverride", {
    width: 390,
    height: 844,
    deviceScaleFactor: 3,
    mobile: true,
  });
  // With mobile emulation on, Chrome applies a visual-viewport zoom of its own,
  // and the first capture after this switch came out as the top-left corner of
  // the phone layout blown up past every edge -- an artefact of the capture, not
  // a layout bug on the page. Pinning the page scale to 1 is what makes the shot
  // show what a phone shows; the layout viewport is already 390 by this point.
  await cdp.call("Emulation.setPageScaleFactor", { pageScaleFactor: 1 });
  await waitFor(
    cdp,
    "window.innerWidth === 390 && document.documentElement.scrollWidth <= 390",
    "The phone viewport never settled at 390px.",
  );
  await delay(600);
  shots.push(await shoot(cdp, "08-dossier-mobile"));
  await cdp.evaluate("document.querySelector('#historyButton').click()");
  await delay(500);
  shots.push(await shoot(cdp, "09-history-mobile"));

  console.log(
    JSON.stringify(
      {
        status: "captured",
        shots,
        ...(seedEvidenceFrom
          ? {
              notPhotographed:
                "the evidence gate: this pass forked the corpus of " +
                `${seedEvidenceFrom} rather than searching for one`,
            }
          : {}),
      },
      null,
      2,
    ),
  );
} finally {
  browser.kill("SIGTERM");
  if (server) server.kill("SIGTERM");
}
