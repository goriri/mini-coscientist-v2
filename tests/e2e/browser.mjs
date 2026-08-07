// The browser plumbing every end-to-end script needs: a DevTools client, the
// waits around it, and the guards that stop two runs sharing one Chrome.
//
// Lifted out of web_hitl_flow.mjs when a second script needed it. Copying it
// would have meant two copies of the port guard and the call deadline, and both
// exist because of a specific hour-long failure -- the kind that only gets
// fixed in whichever copy was in front of somebody at the time.

import { spawn } from "node:child_process";

export const delay = (milliseconds) =>
  new Promise((resolve) => setTimeout(resolve, milliseconds));

export function assert(condition, message) {
  if (!condition) throw new Error(message);
}

// Against the deployed service every stage is a real model call rather than
// the integration stub, so the local budgets are an order of magnitude short.
// The waits scale together; the assertions they guard do not change.
const timeoutScale = Number(process.env.COSCIENTIST_E2E_TIMEOUT_SCALE || "1");

// A DevTools call that never comes back used to wedge the whole run: the
// promise had no deadline, so waitFor sat inside it and its own timeout could
// never fire. One evaluation whose awaited fetch never settled left the run
// idle for forty minutes -- no output, no failure, no CPU -- while the gate it
// was waiting on had been on screen the whole time.
const CALL_TIMEOUT_MILLISECONDS = 30000;

export class Cdp {
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
      clearTimeout(handler.timer);
      if (message.error) handler.reject(new Error(message.error.message));
      else handler.resolve(message.result);
    });
    const abandon = (reason) => {
      for (const [id, handler] of this.pending) {
        clearTimeout(handler.timer);
        handler.reject(new Error(reason));
        this.pending.delete(id);
      }
    };
    this.socket.addEventListener("close", () =>
      abandon("The DevTools connection closed mid-run."),
    );
    this.socket.addEventListener("error", () =>
      abandon("The DevTools connection failed mid-run."),
    );
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
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`${method} did not answer within 30 s.`));
      }, CALL_TIMEOUT_MILLISECONDS);
      this.pending.set(id, { resolve, reject, timer });
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
      throw new Error(
        result.exceptionDetails.text || "Browser evaluation failed.",
      );
    }
    return result.result.value;
  }
}

export async function waitFor(cdp, expression, message, timeout = 15000) {
  timeout *= timeoutScale;
  const started = Date.now();
  let lastFailure = null;
  while (Date.now() - started < timeout) {
    try {
      if (await cdp.evaluate(expression)) return;
    } catch (error) {
      // A single unanswered call is a hiccup to retry, not a verdict; the
      // deadline above is what ends the wait either way. The last one is kept
      // so a wait that only ever failed says why.
      lastFailure = error;
    }
    await delay(100);
  }
  throw new Error(lastFailure ? `${message} (${lastFailure.message})` : message);
}

// A second Chrome cannot bind a port the first one holds, and it exits
// quietly when it tries. The run then attaches to whichever browser is
// already there and drives someone else's tabs: two live runs against the
// same service, one browser, and when the first run ended its browser went
// with it and the second hung until its timeout. Ninety minutes and a
// billed research session were lost to that before it was noticed.
export async function refuseOccupiedPort(debuggingPort) {
  try {
    const response = await fetch(
      `http://127.0.0.1:${debuggingPort}/json/version`,
    );
    if (response.ok) {
      throw new Error(
        `A browser is already listening on the DevTools port ${debuggingPort}. ` +
          "Another end-to-end run is probably in flight; stop it, or set " +
          "CHROME_DEBUGGING_PORT to a free port for this one.",
      );
    }
  } catch (error) {
    if (!/ECONNREFUSED|fetch failed/i.test(String(error))) throw error;
  }
}

// The same failure one layer down. A server left behind by an earlier run keeps
// the port, the new one exits quietly when it cannot bind, and waitForServer is
// satisfied by the orphan -- which is serving a different state directory. Two
// runs in a row then failed to find a session that had just been seeded, and
// the seeding looked like the bug.
export async function refuseOccupiedServer(baseUrl) {
  try {
    const response = await fetch(baseUrl);
    if (response.ok) {
      throw new Error(
        `Something is already serving ${baseUrl}, and it is not this run. ` +
          "Stop it (a previous end-to-end server is the usual answer), or set " +
          "COSCIENTIST_E2E_URL to a free port.",
      );
    }
  } catch (error) {
    if (!/ECONNREFUSED|fetch failed/i.test(String(error))) throw error;
  }
}

export function spawnBrowser(chrome, debuggingPort, profile) {
  return spawn(
    chrome,
    [
      "--headless=new",
      "--no-sandbox",
      "--disable-gpu",
      `--remote-debugging-port=${debuggingPort}`,
      `--user-data-dir=${profile}`,
      "about:blank",
    ],
    { stdio: "ignore" },
  );
}

export async function waitForServer(baseUrl) {
  for (let attempt = 0; attempt < 300; attempt += 1) {
    try {
      const response = await fetch(baseUrl);
      if (response.ok) return;
    } catch {
      // The ADK application is still attaching its A2A routes.
    }
    await delay(100);
  }
  throw new Error("The local Co-Scientist server did not become ready.");
}

export async function waitForDebugging(debuggingPort) {
  for (let attempt = 0; attempt < 80; attempt += 1) {
    try {
      const response = await fetch(
        `http://127.0.0.1:${debuggingPort}/json/version`,
      );
      if (response.ok) return;
    } catch {
      // Chrome is still starting.
    }
    await delay(100);
  }
  throw new Error("Chrome DevTools endpoint did not become ready.");
}

export async function openPage(debuggingPort, baseUrl) {
  const response = await fetch(
    `http://127.0.0.1:${debuggingPort}/json/new?${encodeURIComponent(baseUrl)}`,
    { method: "PUT" },
  );
  const page = await response.json();
  const cdp = new Cdp(page.webSocketDebuggerUrl);
  await cdp.ready();
  await cdp.call("Runtime.enable");
  await cdp.call("Page.enable");
  return cdp;
}
