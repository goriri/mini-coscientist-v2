// Captures the workspace at the points a researcher actually looks at it: the
// empty landing, a live gate, and the finished dossier, at desktop and phone
// widths. The HITL flow test asserts behaviour; this one exists so the layout
// and the type can be read by eye before a deploy.
import { spawn } from "node:child_process";
import { mkdtemp, writeFile, mkdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

const baseUrl = process.env.COSCIENTIST_E2E_URL || "http://127.0.0.1:8768";
const outputDir = process.env.COSCIENTIST_SHOT_DIR || "/tmp/coscientist-shots";
const chrome =
  process.env.CHROME_BIN ||
  "/tmp/math-witch-playwright/chromium-1187/chrome-linux/chrome";
const debuggingPort = Number(process.env.CHROME_DEBUGGING_PORT || "9224");
const profile = await mkdtemp(join(tmpdir(), "coscientist-shots-"));
const startServer = process.env.COSCIENTIST_E2E_START_SERVER !== "false";
// Thirty seconds is right for the offline server this script starts itself.
// Pointed at a deployment with Deep Research on, the evidence stage is seven
// concurrent research interactions and takes minutes, so the budget is a knob
// rather than a constant -- otherwise the only way to photograph a real run is
// to edit the harness.
const gateTimeout = Number(
  process.env.COSCIENTIST_E2E_GATE_TIMEOUT_MS || "30000",
);
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

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function waitForEndpoint(url, attempts, note) {
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      if ((await fetch(url)).ok) return;
    } catch {
      // Still starting.
    }
    await delay(100);
  }
  throw new Error(note);
}

class Cdp {
  constructor(url) {
    this.socket = new WebSocket(url);
    this.nextId = 1;
    this.pending = new Map();
    this.socket.addEventListener("message", (event) => {
      const message = JSON.parse(event.data);
      if (!message.id) return;
      const handler = this.pending.get(message.id);
      if (!handler) return;
      this.pending.delete(message.id);
      if (message.error) handler.reject(new Error(message.error.message));
      else handler.resolve(message.result);
    });
  }

  async ready() {
    if (this.socket.readyState === WebSocket.OPEN) return;
    await new Promise((resolve, reject) => {
      this.socket.addEventListener("open", resolve, { once: true });
      this.socket.addEventListener("error", reject, { once: true });
    });
  }

  call(method, params = {}) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.socket.send(JSON.stringify({ id, method, params }));
    });
  }

  async evaluate(expression) {
    const result = await this.call("Runtime.evaluate", {
      expression,
      awaitPromise: true,
      returnByValue: true,
    });
    if (result.exceptionDetails) {
      throw new Error(result.exceptionDetails.text || "Evaluation failed.");
    }
    return result.result.value;
  }
}

async function waitFor(cdp, expression, note, timeout = gateTimeout) {
  const started = Date.now();
  while (Date.now() - started < timeout) {
    if (await cdp.evaluate(expression)) return;
    await delay(150);
  }
  throw new Error(note);
}

async function click(cdp, selector) {
  // The selectors carry their own double quotes, so they are encoded rather
  // than pasted into a quoted string.
  const query = JSON.stringify(`.approval-card:not(.resolved) ${selector}`);
  for (let attempt = 0; attempt < 20; attempt += 1) {
    const landed = await cdp.evaluate(
      `(() => {
        const button = document.querySelector(${query});
        if (!button) return false;
        button.click();
        return true;
      })()`,
    );
    if (landed) return;
    await delay(200);
  }
  throw new Error(`No live gate offered ${selector}.`);
}

async function settle(cdp, selector) {
  // A decision reaches the server before the next poll repaints, so the card
  // just answered stays on screen for a beat. Without this the loop re-enters
  // on the stale gate and clicks a button that vanishes mid-retry.
  const query = JSON.stringify(`.approval-card:not(.resolved) ${selector}`);
  await waitFor(
    cdp,
    `!document.querySelector(${query})`,
    `The gate offering ${selector} never cleared.`,
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

try {
  await waitForEndpoint(baseUrl, 300, "The server did not become ready.");
  await waitForEndpoint(
    `http://127.0.0.1:${debuggingPort}/json/version`,
    80,
    "Chrome DevTools did not become ready.",
  );
  const page = await (
    await fetch(
      `http://127.0.0.1:${debuggingPort}/json/new?${encodeURIComponent(baseUrl)}`,
      { method: "PUT" },
    )
  ).json();
  const cdp = new Cdp(page.webSocketDebuggerUrl);
  await cdp.ready();
  await cdp.call("Runtime.enable");
  await cdp.call("Page.enable");
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

  for (let gate = 0; gate < 6; gate += 1) {
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
      await click(cdp, '[data-decision="exploratory_evidence"]');
      await settle(cdp, '[data-decision="exploratory_evidence"]');
      continue;
    }
    if (await cdp.evaluate("!!document.querySelector('.ranking-table')")) {
      await cdp.evaluate(
        "document.querySelector('.ranking-table').scrollIntoView({block: 'center'})",
      );
      await delay(300);
      shots.push(await shoot(cdp, "05-ranking-presentation"));
    }
    await click(cdp, '[data-decision="accept"]:not(:disabled)');
    await settle(cdp, '[data-decision="accept"]');
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

  console.log(JSON.stringify({ status: "captured", shots }, null, 2));
} finally {
  browser.kill("SIGTERM");
  if (server) server.kill("SIGTERM");
}
