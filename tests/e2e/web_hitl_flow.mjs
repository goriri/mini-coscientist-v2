import { spawn } from "node:child_process";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

const baseUrl = process.env.COSCIENTIST_E2E_URL || "http://127.0.0.1:8766";
const chrome =
  process.env.CHROME_BIN ||
  "/tmp/math-witch-playwright/chromium-1187/chrome-linux/chrome";
const debuggingPort = Number(process.env.CHROME_DEBUGGING_PORT || "9223");
const profile = await mkdtemp(join(tmpdir(), "coscientist-chrome-"));
const startServer = process.env.COSCIENTIST_E2E_START_SERVER !== "false";
let server = null;

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
        // Subprocess: no pytest fixture strips its credentials, so the Deep
        // Research switch has to be thrown here or a browser test bills a
        // real research pass.
        COSCIENTIST_DEEP_RESEARCH: "off",
        COSCIENTIST_STATE_DIR: join(profile, "state"),
        UV_CACHE_DIR: "/tmp/coscientist-uv-cache",
      },
      stdio: "ignore",
    },
  );
}

// A second Chrome cannot bind a port the first one holds, and it exits
// quietly when it tries. The run then attaches to whichever browser is
// already there and drives someone else's tabs: two live runs against the
// same service, one browser, and when the first run ended its browser went
// with it and the second hung until its timeout. Ninety minutes and a
// billed research session were lost to that before it was noticed.
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

const browser = spawn(
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

const delay = (milliseconds) =>
  new Promise((resolve) => setTimeout(resolve, milliseconds));

async function waitForServer() {
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

async function waitForDebugging() {
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

// A DevTools call that never comes back used to wedge the whole run: the
// promise had no deadline, so waitFor sat inside it and its own timeout could
// never fire. One evaluation whose awaited fetch never settled left the run
// idle for forty minutes -- no output, no failure, no CPU -- while the gate it
// was waiting on had been on screen the whole time.
const CALL_TIMEOUT_MILLISECONDS = 30000;

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

// Against the deployed service every stage is a real model call rather than
// the integration stub, so the local budgets are an order of magnitude short.
// The waits scale together; the assertions they guard do not change.
const timeoutScale = Number(process.env.COSCIENTIST_E2E_TIMEOUT_SCALE || "1");

async function waitFor(cdp, expression, message, timeout = 15000) {
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
  throw new Error(
    lastFailure ? `${message} (${lastFailure.message})` : message,
  );
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

try {
  await waitForServer();
  await waitForDebugging();
  const pageResponse = await fetch(
    `http://127.0.0.1:${debuggingPort}/json/new?${encodeURIComponent(baseUrl)}`,
    { method: "PUT" },
  );
  const page = await pageResponse.json();
  const cdp = new Cdp(page.webSocketDebuggerUrl);
  await cdp.ready();
  await cdp.call("Runtime.enable");
  await cdp.call("Page.enable");
  await waitFor(
    cdp,
    "document.readyState === 'complete' && !!document.querySelector('#composer')",
    "The research workspace did not load.",
  );

  const desktopLayout = await cdp.evaluate(`(() => {
    const shell = document.querySelector(".app-shell");
    const workspace = document.querySelector(".workspace");
    const conversation = document.querySelector(".conversation");
    return {
      viewport: innerHeight,
      shell: shell.clientHeight,
      workspace: workspace.clientHeight,
      conversation: conversation.clientHeight,
      guided: document.querySelector('[data-mode="guided"]').classList.contains("active"),
    };
  })()`);
  assert(desktopLayout.guided, "Guided HITL should be the default mode.");
  assert(
    Math.abs(desktopLayout.shell - desktopLayout.viewport) <= 1,
    "The desktop shell must be bounded to the viewport.",
  );
  assert(
    desktopLayout.conversation < desktopLayout.workspace,
    "The conversation must be an independently scrollable grid row.",
  );

  const samplePrompts = await cdp.evaluate(`(() => {
    const cards = [...document.querySelectorAll("[data-prompt]")];
    return cards.map((card) => {
      card.click();
      return document.querySelector("#promptInput").value;
    });
  })()`);
  assert(
    samplePrompts.length === 3,
    "Exactly three drug-research samples are required.",
  );
  assert(
    samplePrompts[0].includes("KRAS inhibitors") &&
      samplePrompts[1].includes("GLP-1-based") &&
      samplePrompts[2].includes("multidrug-resistant Gram-negative"),
    "The center panel did not expose the three complete drug-research questions.",
  );

  await cdp.evaluate(`(() => {
    const input = document.querySelector("#promptInput");
    input.value = "Does a protective coating improve rechargeable battery cycle life compared with an uncoated control?";
    input.dispatchEvent(new Event("input", { bubbles: true }));
    document.querySelector("#composer").requestSubmit();
  })()`);
  await waitFor(
    cdp,
    "!!document.querySelector('.approval-card:not(.resolved) .workflow-progress')",
    "Initial generation did not immediately show asynchronous progress.",
    3000,
  );
  const stablePolling = await cdp.evaluate(`(() => {
    const card = document.querySelector(".approval-card:not(.resolved)");
    const input = document.querySelector("#promptInput");
    input.disabled = false;
    input.value = "draft question preserved during polling";
    input.focus();
    input.setSelectionRange(6, 14);
    const area = document.querySelector(".conversation");
    const beforeTop = area.scrollTop;
    for (let index = 0; index < 10; index += 1) renderWorkflow(state.workflow);
    const current = document.querySelector(".approval-card:not(.resolved)");
    return {
      sameNode: card === current,
      value: input.value,
      focused: document.activeElement === input,
      selection: [input.selectionStart, input.selectionEnd],
      beforeTop,
      afterTop: area.scrollTop,
    };
  })()`);
  assert(
    stablePolling.sameNode,
    "Polling replaced the Specialists working card.",
  );
  assert(
    stablePolling.value === "draft question preserved during polling" &&
      stablePolling.focused &&
      stablePolling.selection[0] === 6 &&
      stablePolling.selection[1] === 14,
    "Polling did not preserve the research editor state.",
  );
  assert(
    stablePolling.beforeTop === stablePolling.afterTop,
    "Polling changed the conversation scroll position.",
  );
  await cdp.evaluate(`(() => {
    const input = document.querySelector("#promptInput");
    input.value = "";
    input.blur();
  })()`);
  await waitFor(
    cdp,
    "!!document.querySelector('.approval-card:not(.resolved) [data-decision=\"toggle_edit\"]')",
    "The first human approval gate did not become ready.",
    30000,
  );

  const workflowId = await cdp.evaluate(
    "document.querySelector('.approval-card:not(.resolved)').dataset.workflowId",
  );
  assert(workflowId, "The approval card must reference a durable workflow.");

  const structuredPreview = await cdp.evaluate(
    `formatText('### Goal manager\\n\\n{"success_criteria":["A measurable endpoint"],"blocking":false}')`,
  );
  assert(
    structuredPreview.includes("Success Criteria") &&
      structuredPreview.includes("<ul>") &&
      !structuredPreview.includes('{"success_criteria"'),
    "Structured research artifacts were not rendered as readable fields.",
  );

  const beforeRevision = await cdp.evaluate(
    "document.querySelectorAll('.message.assistant').length",
  );
  await cdp.evaluate(`(() => {
    const card = document.querySelector(".approval-card:not(.resolved)");
    card.querySelector('[data-decision="toggle_edit"]').click();
    card.querySelector(".direct-edit-field").value += "\\n\\nResearcher edit: add a prespecified endpoint and matched control.";
    card.querySelector('[data-decision="edit"]').click();
  })()`);
  await waitFor(
    cdp,
    `document.querySelectorAll(".message.assistant").length > ${beforeRevision}`,
    "Direct editing did not create a new visible artifact version.",
    30000,
  );
  const editedDraftSaved = await cdp.evaluate(
    `fetch("/api/research/sessions/${workflowId}").then(r => r.json()).then(w => w.pending_draft.content.includes("Researcher edit:"))`,
  );
  assert(editedDraftSaved, "The directly edited draft was not persisted.");

  // How many gates a run asks for is a property of the run, not a constant: the
  // evidence-integrity card and any governance finding appear only where the
  // stage produced one, so the deployed service asked five times and the
  // integration stub four. Looping a fixed five meant the last pass waited for a
  // gate that was never coming -- against the stub it failed outright, and
  // against the service it would have burned its whole budget after the dossier
  // was already built. The gates are driven until the run says it has a report,
  // and the ceiling is here to end a loop that is not converging, not to bound
  // the run: reaching it is a failure and says so.
  const GATE_CEILING = 8;
  let gatesAccepted = 0;
  let rankChecked = false;
  for (let gate = 0; ; gate += 1) {
    assert(
      gate < GATE_CEILING,
      `The workflow was still asking for approval after ${GATE_CEILING} gates.`,
    );
    await waitFor(
      cdp,
      `(async () => {
        if (document.querySelector('.approval-card:not(.resolved)')) return true;
        const workflow = await fetch("/api/research/sessions/${workflowId}").then((response) => response.json());
        return workflow.status === "ready_for_report";
      })()`,
      `Approval or evidence-integrity gate ${gate + 1} was unavailable.`,
      30000,
    );
    if (
      !(await cdp.evaluate(
        "!!document.querySelector('.approval-card:not(.resolved)')",
      ))
    ) {
      break;
    }
    const needsEvidenceFallback = await cdp.evaluate(
      "!!document.querySelector('.approval-card:not(.resolved) [data-decision=\"exploratory_evidence\"]')",
    );
    if (needsEvidenceFallback) {
      await cdp.evaluate(
        "document.querySelector('.approval-card:not(.resolved) [data-decision=\"exploratory_evidence\"]').click()",
      );
    }
    // A fatal safety finding is a state the reflect stage is designed to reach,
    // and it used to be the end of the run in the browser. Overriding rather
    // than withdrawing, so the population the later gates assert on is the one
    // the tournament ranked; confirm() is answered because CDP cannot.
    //
    // Waited for against the accept button rather than tested once ahead of it.
    // The card is rendered when the stage produces its draft, and a governance
    // finding lands on a later poll: a run whose reflect stage blocked on a real
    // safety flaw was checked before the flaw arrived, skipped this loop, and
    // then spent its whole budget waiting for an accept button that governance
    // had already disabled. The two are one gate, so they are one wait.
    for (;;) {
      await waitFor(
        cdp,
        `!!document.querySelector('.approval-card:not(.resolved) .governance-finding')
          || !!document.querySelector('.approval-card:not(.resolved) [data-decision="accept"]:not(:disabled)')`,
        `Approval gate ${gate + 1} was unavailable.`,
        30000,
      );
      if (
        !(await cdp.evaluate(
          "!!document.querySelector('.approval-card:not(.resolved) .governance-finding')",
        ))
      ) {
        break;
      }
      const reviewId = await cdp.evaluate(`(() => {
        const ask = window.confirm;
        window.confirm = () => true;
        const finding = document.querySelector(".approval-card:not(.resolved) .governance-finding");
        finding.querySelector(".governance-adjudicator").value = "E2E Safety Officer";
        finding.querySelector(".governance-justification").value = "Automated end-to-end check; no bench work follows this run.";
        finding.querySelector('[data-decision="override_governance"]').click();
        window.confirm = ask;
        return finding.dataset.reviewId;
      })()`);
      // Per finding, not per block: the workflow deliberately answers one at a
      // time, so a session with two flaws stays blocked after the first.
      await waitFor(
        cdp,
        `fetch("/api/research/sessions/${workflowId}").then(r => r.json()).then(w => !w.governance_blockers.some(item => item.review_id === ${JSON.stringify(reviewId)}))`,
        `Governance finding ${reviewId} at gate ${gate + 1} could not be adjudicated.`,
        30000,
      );
    }
    // Keyed to the tournament being on screen rather than to a gate number,
    // which only held while the count of gates was assumed to be fixed.
    if (
      !rankChecked &&
      (await cdp.evaluate(
        "!!document.querySelector('[data-presentation-stage=\"rank\"] .ranking-table')",
      ))
    ) {
      rankChecked = true;
      const rankingPresentation = await cdp.evaluate(`(() => ({
        rankingRows: document.querySelectorAll('[data-presentation-stage="rank"] .ranking-table > div').length,
        candidateCards: document.querySelectorAll('[data-presentation-stage="rank"] .candidate-card').length,
        shortlistCards: document.querySelectorAll('[data-presentation-stage="rank"] .candidate-card.shortlisted').length,
        technicalClosed: !document.querySelector('[data-presentation-stage="rank"] .technical-details').open,
        rawJsonVisible: document.querySelector('[data-presentation-stage="rank"] .message-copy pre:not([hidden])') !== null,
      }))()`);
      assert(
        rankingPresentation.rankingRows === 8,
        "Rank must show all candidates.",
      );
      assert(
        rankingPresentation.candidateCards === 8 &&
          rankingPresentation.shortlistCards === 4,
        "Rank candidate cards or shortlist markers are incomplete.",
      );
      assert(
        rankingPresentation.technicalClosed,
        "Technical JSON must be collapsed by default.",
      );
    }
    await cdp.evaluate(
      "document.querySelector('.approval-card:not(.resolved) [data-decision=\"accept\"]').click()",
    );
    gatesAccepted += 1;
    await waitFor(
      cdp,
      "!!document.querySelector('.workflow-progress') || !!document.querySelector('.approval-card:not(.resolved) [data-decision=\"accept\"]:not(:disabled)') || !document.querySelector('.approval-card:not(.resolved)')",
      "Accept did not immediately show progress or the next gate.",
      3000,
    );
  }
  // A loop that can stop on its own can also stop on the first pass and take
  // the whole gate chapter with it. Milestone approval puts a gate on scope,
  // evidence, ranking and meta-review at the least.
  assert(
    gatesAccepted >= 4,
    `Only ${gatesAccepted} approval gates were driven; milestone mode has at least four.`,
  );
  assert(
    rankChecked,
    "The tournament was never presented, so its table was never checked.",
  );

  await waitFor(
    cdp,
    `fetch("/api/research/sessions/${workflowId}").then(r => r.json()).then(w => w.status === "ready_for_report")`,
    "The milestone workflow did not reach its final dossier.",
    45000,
  );
  await waitFor(
    cdp,
    "document.querySelectorAll('.report-completion .report-export-actions a').length === 3",
    "The completed workflow did not expose all report exports.",
  );
  const reportExports = await cdp.evaluate(`(async () => {
    const links = [...document.querySelectorAll(".report-export-actions a")];
    const responses = await Promise.all(
      links.map(async (link) => {
        const response = await fetch(link.href);
        const bytes = new Uint8Array(await response.arrayBuffer());
        return {
          label: link.textContent.trim(),
          status: response.status,
          signature: [...bytes.slice(0, 5)],
          type: response.headers.get("content-type"),
        };
      }),
    );
    return responses;
  })()`);
  // Named, not counted. A run that spent the better part of an hour reaching
  // this line failed it with "One or more dossier exports failed." and nothing
  // else -- the server log was the only place the 500 and its cause were
  // written down, and this is the last assertion of the flow.
  const failedExports = reportExports.filter((item) => item.status !== 200);
  assert(
    failedExports.length === 0,
    `Dossier exports failed: ${failedExports
      .map((item) => `${item.label} returned ${item.status} (${item.type})`)
      .join("; ")}.`,
  );
  const signatures = reportExports.map((item) =>
    item.signature.map((byte) => String.fromCharCode(byte)).join(""),
  );
  assert(
    signatures[0].startsWith("PK") &&
      signatures[1] === "%PDF-" &&
      reportExports[2].type.includes("text/markdown"),
    `DOCX, PDF, or Markdown export signatures were invalid: ${reportExports
      .map((item, index) => `${item.label} ${JSON.stringify(signatures[index])}`)
      .join("; ")}.`,
  );
  await waitFor(
    cdp,
    "document.querySelector('.conversation').scrollHeight > document.querySelector('.conversation').clientHeight",
    "The completed dossier did not produce a scrollable conversation.",
  );

  const scrollMetrics = await cdp.evaluate(`(async () => {
    const area = document.querySelector(".conversation");
    area.scrollTop = area.scrollHeight;
    await new Promise(requestAnimationFrame);
    const bottom = area.scrollTop;
    const max = area.scrollHeight - area.clientHeight;
    area.scrollTop = 0;
    area.dispatchEvent(new Event("scroll"));
    await new Promise(requestAnimationFrame);
    return {
      top: area.scrollTop,
      bottom,
      max,
      jumpVisible: !document.querySelector("#jumpLatest").hidden,
    };
  })()`);
  assert(
    scrollMetrics.top === 0,
    "The conversation could not scroll to the top.",
  );
  assert(
    scrollMetrics.bottom > 0,
    "The conversation could not scroll to the bottom.",
  );
  assert(
    Math.abs(scrollMetrics.bottom - scrollMetrics.max) <= 2,
    "The conversation did not reach its actual bottom.",
  );
  assert(
    scrollMetrics.jumpVisible,
    "The Latest update affordance should appear.",
  );

  await cdp.evaluate("document.querySelector('#jumpLatest').click()");
  await waitFor(
    cdp,
    `(() => {
      const area = document.querySelector(".conversation");
      return area.scrollHeight - area.scrollTop - area.clientHeight < 3;
    })()`,
    "Latest update did not return to the bottom.",
    3000,
  );
  // Against the deployed service the finished record runs to fifty thousand
  // pixels, and the animated scroll this used to do stopped two thousand pixels
  // in: the arrival check above still had a whole document below it, and the
  // affordance came back because the handler saw a reader who had stopped
  // following. Arriving and staying arrived are two claims, and only one of
  // them was being made.
  assert(
    await cdp.evaluate("document.querySelector('#jumpLatest').hidden"),
    "Latest update reappeared after it said it had taken the reader to the end.",
  );

  const derivedTitle = await cdp.evaluate(
    `deriveSessionName("A deliberately long scientific research question about whether protective coatings improve rechargeable battery cycle life under demanding conditions.")`,
  );
  assert(
    derivedTitle.length <= 52 && derivedTitle.endsWith("…"),
    "Session naming must truncate deterministically at a word boundary.",
  );

  // The topbar used to hold a per-browser counter, so it named nothing anyone
  // else could recognise. It carries the session name now, and the name the
  // opening question produces is only a guess -- a researcher must be able to
  // replace it, and the replacement must reach every place the name is shown.
  const topbarName = await cdp.evaluate(
    "document.querySelector('#sessionTitle').textContent",
  );
  assert(
    topbarName.includes("protective coating"),
    `The topbar must name the session, not a counter. Saw "${topbarName}".`,
  );

  await cdp.evaluate(`(() => {
    document.querySelector("#renameSession").click();
    const field = document.querySelector("#sessionTitleInput");
    field.value = "Coating durability study";
    field.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
  })()`);
  await waitFor(
    cdp,
    `document.querySelector("#sessionTitle").textContent === "Coating durability study"
      && document.querySelector('.session-history-item[data-session-id="${workflowId}"] .session-history-open strong').textContent === "Coating durability study"
      && document.querySelector("#currentSessionName").textContent === "Coating durability study"`,
    "A renamed session did not carry its name to the topbar, history and rail.",
    5000,
  );

  // Emptying the field is how the derived name is asked for back.
  await cdp.evaluate(`(() => {
    document.querySelector("#renameSession").click();
    const field = document.querySelector("#sessionTitleInput");
    field.value = "   ";
    field.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
  })()`);
  await waitFor(
    cdp,
    `document.querySelector("#sessionTitle").textContent.includes("protective coating")`,
    "Clearing a custom name did not restore the derived one.",
    5000,
  );

  await cdp.evaluate("document.querySelector('#newInquiry').click()");
  await cdp.evaluate(`(() => {
    const input = document.querySelector("#promptInput");
    input.value = "Can electrolyte concentration change dendrite formation?";
    input.dispatchEvent(new Event("input", { bubbles: true }));
    document.querySelector("#composer").requestSubmit();
  })()`);
  await waitFor(
    cdp,
    "document.querySelectorAll('.session-history-item').length === 2 && !!document.querySelector('.approval-card:not(.resolved) [data-decision=\"toggle_edit\"]')",
    "A second resumable research session was not added to browser history.",
    30000,
  );
  const secondWorkflowId = await cdp.evaluate("state.workflowId");
  const historyOrder = await cdp.evaluate(
    "[...document.querySelectorAll('.session-history-item')].map(item => item.dataset.sessionId)",
  );
  assert(
    historyOrder[0] === secondWorkflowId && historyOrder[1] === workflowId,
    "Recent sessions must be ordered by most recently opened or updated.",
  );

  await cdp.evaluate(
    `document.querySelector('.session-history-item[data-session-id="${workflowId}"] .session-history-open').click()`,
  );
  await waitFor(
    cdp,
    `state.workflowId === "${workflowId}" && document.querySelector("#currentSessionName").textContent.includes("protective coating")`,
    "Selecting a history card did not restore the requested workflow and title.",
  );

  const decisionsBeforePreview = await cdp.evaluate(
    `fetch("/api/research/sessions/${workflowId}").then(r => r.json()).then(w => w.decisions.length)`,
  );
  await cdp.evaluate(
    `document.querySelector('.stage-nav li[data-stage="scope"] > button').click()`,
  );
  await waitFor(
    cdp,
    "!!document.querySelector('.historical-preview-banner') && document.body.classList.contains('history-preview')",
    "Clicking a completed stage did not open its read-only output.",
  );
  const previewState = await cdp.evaluate(`(() => ({
    title: document.querySelector(".historical-preview-banner h2").textContent,
    composerDisabled: document.querySelector("#promptInput").disabled,
    scopeViewing: document.querySelector('.stage-nav li[data-stage="scope"]').classList.contains("viewing"),
  }))()`);
  assert(
    previewState.title.includes("Scope"),
    "The stage preview has the wrong title.",
  );
  assert(
    previewState.composerDisabled,
    "Historical output must not be editable.",
  );
  assert(
    previewState.scopeViewing,
    "The previewed stage must be visually distinct.",
  );
  await cdp.evaluate("document.querySelector('[data-return-current]').click()");
  await waitFor(
    cdp,
    "!!document.querySelector('.message.assistant') && !document.body.classList.contains('history-preview')",
    "Return to current gate did not restore the live workflow.",
  );
  const decisionsAfterPreview = await cdp.evaluate(
    `fetch("/api/research/sessions/${workflowId}").then(r => r.json()).then(w => w.decisions.length)`,
  );
  assert(
    decisionsAfterPreview === decisionsBeforePreview,
    "Browsing stage history must not create workflow decisions.",
  );

  await cdp.call("Page.reload", { ignoreCache: true });
  await waitFor(
    cdp,
    `document.readyState === "complete" && state.workflowId === "${workflowId}" && document.querySelector("#currentSessionCard").hidden === false`,
    "Refresh did not restore the most recently opened research session.",
    15000,
  );

  await cdp.call("Emulation.setDeviceMetricsOverride", {
    width: 390,
    height: 844,
    deviceScaleFactor: 1,
    mobile: true,
  });
  await delay(300);
  const mobileLayout = await cdp.evaluate(`(() => ({
    viewport: innerHeight,
    workspace: document.querySelector(".workspace").clientHeight,
    pageScroll: scrollY,
    topbarTop: document.querySelector(".topbar").getBoundingClientRect().top,
    topbarBottom: document.querySelector(".topbar").getBoundingClientRect().bottom,
    scrollable:
      document.querySelector(".conversation").scrollHeight >
      document.querySelector(".conversation").clientHeight,
  }))()`);
  assert(
    Math.abs(mobileLayout.workspace - mobileLayout.viewport) <= 1,
    "The mobile workspace must remain bounded to the dynamic viewport.",
  );
  assert(
    mobileLayout.scrollable,
    "The conversation must remain scrollable on mobile.",
  );
  assert(
    mobileLayout.pageScroll === 0 && Math.abs(mobileLayout.topbarTop) <= 1,
    "The mobile page shell must not scroll away from its fixed workspace.",
  );

  // The workspace sets overflow:hidden, so anything too wide for a phone is
  // silently sliced off at the right edge rather than producing a scrollbar
  // anyone would notice. The topbar heading and the composer selects were both
  // running past it. Nothing a reader has to read may leave the viewport.
  const mobileOverflow = await cdp.evaluate(`(() => {
    const scrolls = (node) => {
      for (let at = node.parentElement; at; at = at.parentElement) {
        // A code block that scrolls sideways is offering its width, not losing it.
        if (at.scrollWidth > at.clientWidth + 1) return true;
      }
      return false;
    };
    const escapes = [];
    for (const node of document.querySelectorAll(".workspace *")) {
      if (!node.getClientRects().length) continue;
      const box = node.getBoundingClientRect();
      if (box.width === 0 || box.right <= innerWidth + 1) continue;
      if (scrolls(node)) continue;
      escapes.push({
        selector: node.tagName.toLowerCase() + "." + (node.className || "").toString().split(" ")[0],
        right: Math.round(box.right),
        width: Math.round(box.width),
        parent: Math.round(node.parentElement.getBoundingClientRect().width),
      });
    }
    return escapes.slice(0, 12);
  })()`);
  assert(
    mobileOverflow.length === 0,
    `Content ran off the phone viewport: ${JSON.stringify(mobileOverflow)}`,
  );

  await cdp.evaluate("document.querySelector('#historyButton').click()");
  await waitFor(
    cdp,
    "document.body.classList.contains('history-open') && document.querySelector('.context-panel').getBoundingClientRect().right <= innerWidth + 1",
    "The mobile research-history drawer did not open.",
  );
  const mobileHistory = await cdp.evaluate(`(() => {
    const panel = document.querySelector(".context-panel").getBoundingClientRect();
    return {
      cards: document.querySelectorAll(".session-history-item").length,
      left: panel.left,
      right: panel.right,
      viewport: innerWidth,
    };
  })()`);
  assert(
    mobileHistory.cards === 2,
    "The mobile drawer lost browser session history.",
  );
  assert(
    mobileHistory.left >= 0 &&
      mobileHistory.right <= mobileHistory.viewport + 1,
    "The mobile history drawer must fit inside the viewport.",
  );
  await cdp.evaluate("document.querySelector('#closeHistory').click()");

  await cdp
    .call("Page.captureScreenshot", {
      format: "png",
      fromSurface: true,
      captureBeyondViewport: false,
    })
    .then(async ({ data }) => {
      const { writeFile } = await import("node:fs/promises");
      await writeFile(
        "/tmp/coscientist-hitl-e2e.png",
        Buffer.from(data, "base64"),
      );
    });

  await cdp.evaluate(`(() => {
    window.confirm = () => true;
    document.querySelector('.session-history-item[data-session-id="${secondWorkflowId}"] .session-delete-cloud').click();
  })()`);
  await waitFor(
    cdp,
    `document.querySelectorAll(".session-history-item").length === 1 && fetch("/api/research/sessions/${secondWorkflowId}").then(response => response.status === 404)`,
    "Permanent cloud deletion did not remove the session and its browser reference.",
    10000,
  );

  console.log(
    JSON.stringify(
      {
        status: "passed",
        workflowId,
        secondWorkflowId,
        desktopLayout,
        scrollMetrics,
        mobileLayout,
        mobileHistory,
        screenshot: "/tmp/coscientist-hitl-e2e.png",
      },
      null,
      2,
    ),
  );
} finally {
  browser.kill("SIGTERM");
  if (server) server.kill("SIGTERM");
}
