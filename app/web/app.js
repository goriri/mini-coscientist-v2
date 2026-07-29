const APP_NAME = "app";
const SESSION_HISTORY_KEY = "coscientist.researchSessions.v1";
const CURRENT_WORKFLOW_KEY = "coscientist.currentWorkflowId";
const RESEARCH_STAGES = [
  "scope",
  "evidence",
  "generate",
  "reflect",
  "rank",
  "evolve",
  "proximity",
  "meta_review",
];

function loadSessionHistory() {
  try {
    const saved = JSON.parse(localStorage.getItem(SESSION_HISTORY_KEY) || "[]");
    return Array.isArray(saved) ? saved.slice(0, 20) : [];
  } catch {
    return [];
  }
}

const state = {
  userId: localStorage.getItem("coscientist.userId") || crypto.randomUUID(),
  sessionId: null,
  workflowId: null,
  workflow: null,
  mode: localStorage.getItem("coscientist.mode") || "guided",
  busy: false,
  autoFollow: true,
  lastDraftId: null,
  pollTimer: null,
  inquiry: Number(localStorage.getItem("coscientist.inquiry") || "1"),
  viewingStage: null,
  recentSessions: loadSessionHistory(),
};

localStorage.setItem("coscientist.userId", state.userId);

const elements = {
  composer: document.querySelector("#composer"),
  input: document.querySelector("#promptInput"),
  send: document.querySelector("#sendButton"),
  welcome: document.querySelector("#welcome"),
  messages: document.querySelector("#messages"),
  newInquiry: document.querySelector("#newInquiry"),
  copySession: document.querySelector("#copySession"),
  connection: document.querySelector(".composer-note"),
  connectionText: document.querySelector("#connectionText"),
  inquiryNumber: document.querySelector("#inquiryNumber"),
  toast: document.querySelector("#toast"),
  mobileMenu: document.querySelector("#mobileMenu"),
  conversation: document.querySelector(".conversation"),
  jumpLatest: document.querySelector("#jumpLatest"),
  approvalProfile: document.querySelector("#approvalProfile"),
  approvalIndicator: document.querySelector("#approvalIndicator"),
  currentSessionCard: document.querySelector("#currentSessionCard"),
  currentSessionName: document.querySelector("#currentSessionName"),
  currentSessionState: document.querySelector("#currentSessionState"),
  sessionHistory: document.querySelector("#sessionHistory"),
  sessionCount: document.querySelector("#sessionCount"),
  historyButton: document.querySelector("#historyButton"),
  closeHistory: document.querySelector("#closeHistory"),
};

const agentLabels = {
  co_scientist_supervisor: "Co-Scientist",
  goal_manager: "Goal manager",
  evidence_discovery: "Evidence discovery",
  deep_research_discovery: "Deep Research discovery",
  source_verification: "Source verification",
  generation: "Hypothesis generation",
  reflection: "Correctness review",
  novelty_review: "Novelty review",
  methods_statistics: "Methods & statistics",
  ethics_safety_governance: "Ethics & governance",
  impact_review: "Impact review",
  ranking: "Tournament ranking",
  evolution: "Candidate evolution",
  proximity: "Proximity landscape",
  meta_reviewer: "Meta reviewer",
};

function updateInquiryNumber() {
  elements.inquiryNumber.textContent = String(state.inquiry).padStart(3, "0");
}

function setConnection(mode, text) {
  elements.connection.classList.remove("ready", "error");
  if (mode) elements.connection.classList.add(mode);
  elements.connectionText.textContent = text;
}

function toast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("visible");
  window.setTimeout(() => elements.toast.classList.remove("visible"), 2200);
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatInline(value) {
  return escapeHtml(value)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}

function formatPlainText(value) {
  const chunks = value.split(/```(?:[a-zA-Z0-9_-]+)?\n?/);
  return chunks
    .map((chunk, index) => {
      if (index % 2 === 1) {
        return `<pre><code>${escapeHtml(chunk.trim())}</code></pre>`;
      }
      return chunk
        .trim()
        .split(/\n{2,}/)
        .filter(Boolean)
        .map((paragraph) => `<p>${formatInline(paragraph).replaceAll("\n", "<br>")}</p>`)
        .join("");
    })
    .join("");
}

function humanizeKey(value) {
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function formatStructuredValue(value) {
  if (Array.isArray(value)) {
    if (!value.length) return '<p class="empty-value">None recorded</p>';
    return `<ul>${value
      .map((item) =>
        typeof item === "object" && item !== null
          ? `<li class="nested-record">${formatStructuredValue(item)}</li>`
          : `<li>${formatInline(String(item))}</li>`,
      )
      .join("")}</ul>`;
  }
  if (value && typeof value === "object") {
    return `<dl class="structured-fields">${Object.entries(value)
      .filter(([key]) => key !== "schema_version")
      .map(
        ([key, item]) =>
          `<div><dt>${escapeHtml(humanizeKey(key))}</dt><dd>${formatStructuredValue(item)}</dd></div>`,
      )
      .join("")}</dl>`;
  }
  if (typeof value === "boolean") return `<p>${value ? "Yes" : "No"}</p>`;
  if (value === null || value === "") return '<p class="empty-value">Not specified</p>';
  return `<p>${formatInline(String(value))}</p>`;
}

function formatArtifactSection(section) {
  const headingMatch = section.match(/^###\s+([^\n]+)\n*/);
  const heading = headingMatch ? headingMatch[1].trim() : "";
  const body = headingMatch ? section.slice(headingMatch[0].length).trim() : section.trim();
  let structured = null;
  let remainder = "";

  if (body.startsWith("{")) {
    for (let end = body.length; end > 1; end = body.lastIndexOf("}", end - 1)) {
      if (end < 1) break;
      try {
        structured = JSON.parse(body.slice(0, end + 1));
        remainder = body.slice(end + 1).trim();
        break;
      } catch {
        // Try the preceding closing brace.
      }
    }
  }

  return `
    <section class="artifact-section">
      ${heading ? `<h3>${escapeHtml(heading)}</h3>` : ""}
      ${
        structured
          ? `<div class="structured-artifact">${formatStructuredValue(structured)}</div>`
          : formatPlainText(body)
      }
      ${remainder ? `<div class="artifact-notes">${formatPlainText(remainder)}</div>` : ""}
    </section>
  `;
}

function formatText(value) {
  const sections = value.trim().split(/(?=^###\s+)/m).filter(Boolean);
  return sections.map(formatArtifactSection).join("");
}

function renderDisplayValue(value) {
  if (Array.isArray(value)) {
    if (!value.length) return '<p class="empty-value">None recorded</p>';
    return `<ul>${value
      .map((item) => `<li>${renderDisplayValue(item)}</li>`)
      .join("")}</ul>`;
  }
  if (value && typeof value === "object") {
    return `<dl class="compact-fields">${Object.entries(value)
      .map(
        ([key, item]) =>
          `<div><dt>${escapeHtml(humanizeKey(key))}</dt><dd>${renderDisplayValue(item)}</dd></div>`,
      )
      .join("")}</dl>`;
  }
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (value === null || value === undefined || value === "") return "Not specified";
  return formatInline(String(value));
}

function renderCandidateCard(candidate) {
  const reviewSummary = (candidate.reviews || [])
    .map(
      (review) => `
        <article class="review-summary ${review.fatal_flaws?.length ? "has-flaw" : ""}">
          <div><strong>${escapeHtml(humanizeKey(review.criterion))}</strong><span>${Math.round((review.confidence || 0) * 100)}% confidence</span></div>
          <p>${escapeHtml(review.recommendation.replaceAll("_", " "))}</p>
          ${
            review.findings?.length
              ? `<ul>${review.findings.map((item) => `<li>${formatInline(item)}</li>`).join("")}</ul>`
              : ""
          }
          ${
            review.fatal_flaws?.length
              ? `<div class="fatal-flaw"><strong>Fatal flaws</strong>${renderDisplayValue(review.fatal_flaws)}</div>`
              : ""
          }
        </article>`,
    )
    .join("");
  return `
    <article class="candidate-card ${candidate.shortlisted ? "shortlisted" : ""}">
      <header>
        <div>
          <span class="candidate-label">${escapeHtml(candidate.label || candidate.candidate_id)}</span>
          <h4>${formatInline(candidate.claim)}</h4>
        </div>
        <div class="candidate-badges">
          ${candidate.rank ? `<span>#${candidate.rank}</span>` : ""}
          ${candidate.elo ? `<span>${candidate.elo} Elo</span>` : ""}
          ${candidate.shortlisted ? "<span>Shortlist</span>" : ""}
        </div>
      </header>
      <p class="candidate-rationale">${formatInline(candidate.rationale || "")}</p>
      <div class="candidate-key-points">
        <div><strong>Key prediction</strong>${renderDisplayValue(candidate.predictions?.[0])}</div>
        <div><strong>Falsifier</strong>${renderDisplayValue(candidate.falsifier)}</div>
      </div>
      ${
        candidate.fatal_flaws?.length
          ? `<div class="fatal-flaw"><strong>Unresolved fatal flaws</strong>${renderDisplayValue(candidate.fatal_flaws)}</div>`
          : ""
      }
      <details class="candidate-details">
        <summary>Show full details</summary>
        <div class="candidate-detail-grid">
          <div><strong>Strategy</strong>${renderDisplayValue(candidate.strategy)}</div>
          <div><strong>Predictions</strong>${renderDisplayValue(candidate.predictions)}</div>
          <div><strong>Competing explanations</strong>${renderDisplayValue(candidate.alternatives)}</div>
          <div><strong>Dependencies</strong>${renderDisplayValue(candidate.dependencies)}</div>
          <div><strong>Risks</strong>${renderDisplayValue(candidate.risks)}</div>
          <div><strong>Go/no-go tests</strong>${renderDisplayValue(candidate.go_no_go_tests)}</div>
        </div>
        ${reviewSummary ? `<section class="candidate-reviews"><h5>Independent reviews</h5>${reviewSummary}</section>` : ""}
        <p class="technical-id">Artifact identity: ${escapeHtml(candidate.candidate_id)}</p>
      </details>
    </article>`;
}

function renderStagePresentation(presentation, rawText = "") {
  if (!presentation) return formatText(rawText);
  const metrics = (presentation.metrics || [])
    .map(
      (metric) =>
        `<div><strong>${renderDisplayValue(metric.value)}</strong><span>${escapeHtml(metric.label)}</span></div>`,
    )
    .join("");
  const details = (presentation.details || [])
    .map(
      (detail) =>
        `<section><h4>${escapeHtml(detail.label)}</h4>${renderDisplayValue(detail.value)}</section>`,
    )
    .join("");
  const ranking = presentation.ranking?.length
    ? `<div class="ranking-table" role="table" aria-label="Candidate ranking">
        ${presentation.ranking
          .map(
            (item) => `
              <div role="row">
                <strong role="cell">#${item.rank}</strong>
                <span role="cell"><b>${escapeHtml(item.label)}</b>${formatInline(item.claim)}</span>
                <span role="cell">${item.elo} Elo${item.shortlisted ? " · shortlist" : ""}</span>
              </div>`,
          )
          .join("")}
      </div>`
    : "";
  const comparisons = (presentation.comparison_rounds || [])
    .map(
      (round) => `
        <details class="comparison-round">
          <summary>Round ${round.round} · ${round.comparisons.length} comparisons</summary>
          ${round.comparisons
            .map(
              (item) =>
                `<p><strong>${escapeHtml(item.candidate_a_label)}</strong> vs <strong>${escapeHtml(item.candidate_b_label)}</strong> · winner ${escapeHtml(item.winner_label || "draw")}<br><span>${formatInline(item.rationale)}</span></p>`,
            )
            .join("")}
        </details>`,
    )
    .join("");
  const evolution = (presentation.evolution || [])
    .map(
      (record) => `
        <section class="evolution-card">
          <span>Evolution round ${record.round}</span>
          ${renderCandidateCard(record.candidate)}
          <div class="evolution-notes"><strong>Changes</strong>${renderDisplayValue(record.changes)}<strong>Critiques addressed</strong>${renderDisplayValue(record.critiques_addressed)}<strong>New prediction</strong>${renderDisplayValue(record.new_prediction)}</div>
        </section>`,
    )
    .join("");
  const clusters = (presentation.clusters || [])
    .map(
      (cluster) => `
        <article class="cluster-card">
          <h4>${escapeHtml(cluster.name)}</h4>
          <p><strong>Shared mechanism:</strong> ${formatInline(cluster.shared_mechanism)}</p>
          <p><strong>Shared outcome:</strong> ${formatInline(cluster.shared_outcome)}</p>
          <ul>${cluster.candidates
            .map(
              (candidate) =>
                `<li><strong>${escapeHtml(candidate.label)}</strong>${formatInline(candidate.claim)}</li>`,
            )
            .join("")}</ul>
          <details><summary>Required data and evidence overlap</summary>${renderDisplayValue(cluster.required_data)}${renderDisplayValue(cluster.evidence_overlap)}</details>
        </article>`,
    )
    .join("");
  const recommendations = (presentation.recommendations || [])
    .map(
      (item) => `
        <article class="recommendation-card ${item.excluded_for_fatal_flaw ? "excluded" : ""}">
          <span>${item.recommended ? "Recommended" : "Excluded"}</span>
          <h4>${escapeHtml(item.label)}</h4>
          <p>${formatInline(item.claim)}</p>
          ${item.fatal_flaws?.length ? `<div class="fatal-flaw">${renderDisplayValue(item.fatal_flaws)}</div>` : ""}
        </article>`,
    )
    .join("");
  return `
    <section class="stage-presentation" data-presentation-stage="${escapeHtml(presentation.stage)}">
      <header class="presentation-head">
        <p class="eyebrow">${escapeHtml(stageLabel(presentation.stage))} · structured result</p>
        <h3>${formatInline(presentation.summary || "")}</h3>
      </header>
      ${metrics ? `<div class="presentation-metrics">${metrics}</div>` : ""}
      ${ranking}
      ${presentation.candidates?.length ? `<div class="candidate-grid">${presentation.candidates.map(renderCandidateCard).join("")}</div>` : ""}
      ${comparisons ? `<section class="comparison-list"><h4>Tournament details</h4>${comparisons}</section>` : ""}
      ${evolution ? `<div class="evolution-list">${evolution}</div>` : ""}
      ${clusters ? `<div class="cluster-grid">${clusters}</div>` : ""}
      ${recommendations ? `<div class="recommendation-grid">${recommendations}</div>` : ""}
      ${details ? `<div class="presentation-details">${details}</div>` : ""}
      ${
        rawText
          ? `<details class="technical-details"><summary>Technical details and source artifact</summary><pre><code>${escapeHtml(rawText)}</code></pre></details>`
          : ""
      }
    </section>`;
}

function initials(author) {
  const label = agentLabels[author] || author || "Co-Scientist";
  return label
    .split(/[\s&-]+/)
    .slice(0, 2)
    .map((part) => part[0])
    .join("")
    .toUpperCase();
}

function ensureConversation() {
  elements.welcome.hidden = true;
  elements.messages.hidden = false;
}

function appendMessage(role, text, author = "", presentation = null) {
  ensureConversation();
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const label = role === "user" ? "Researcher" : agentLabels[author] || "Co-Scientist";
  article.innerHTML = `
    <div class="message-avatar">${role === "user" ? "You" : initials(author)}</div>
    <div>
      <div class="message-meta">
        <strong>${escapeHtml(label)}</strong>
        <span>${role === "user" ? "Inquiry" : "Agent response"}</span>
      </div>
      <div class="message-copy">${renderStagePresentation(presentation, text)}</div>
    </div>
  `;
  elements.messages.append(article);
  scrollToBottom(role === "user");
  return article;
}

function appendPending() {
  ensureConversation();
  const article = document.createElement("article");
  article.className = "message assistant pending";
  article.innerHTML = `
    <div class="message-avatar">CS</div>
    <div>
      <div class="message-meta">
        <strong>Co-Scientist</strong>
        <span>Reasoning</span>
      </div>
      <div class="message-copy">
        <div class="thinking" aria-label="Co-Scientist is reasoning">
          <span></span><span></span><span></span>
        </div>
      </div>
    </div>
  `;
  elements.messages.append(article);
  scrollToBottom(true);
  return article;
}

function updateAssistant(article, text, author, presentation = null) {
  article.classList.remove("pending");
  article.querySelector(".message-avatar").textContent = initials(author);
  article.querySelector(".message-meta strong").textContent =
    agentLabels[author] || "Co-Scientist";
  article.querySelector(".message-meta span").textContent = "Agent response";
  article.querySelector(".message-copy").innerHTML = renderStagePresentation(
    presentation,
    text,
  );
  scrollToBottom();
}

function showError(article, message) {
  article.classList.remove("pending");
  article.querySelector(".message-meta span").textContent = "Connection error";
  article.querySelector(".message-copy").innerHTML =
    `<p class="message-error">${escapeHtml(message)}</p>`;
  scrollToBottom();
}

function scrollToBottom(force = false) {
  if (!force && !state.autoFollow) return;
  window.requestAnimationFrame(() => {
    elements.conversation.scrollTo({
      top: elements.conversation.scrollHeight,
      behavior: force ? "smooth" : "auto",
    });
  });
}

function nearConversationBottom() {
  const remaining =
    elements.conversation.scrollHeight -
    elements.conversation.scrollTop -
    elements.conversation.clientHeight;
  return remaining < 120;
}

function setActiveAgent(author) {
  document
    .querySelectorAll("#specialistList > div")
    .forEach((item) => item.classList.toggle("active", item.dataset.agent === author));
}

function deriveSessionName(question, maxLength = 52) {
  const normalized = String(question || "").replace(/\s+/g, " ").trim();
  if (!normalized) return "Untitled inquiry";
  const clause =
    normalized
      .split(/(?:[.!?;:\n]|\s[—–]\s|,\s+(?:and|but|while|whereas)\b)/i)
      .find((part) => part.trim().length >= 3)
      ?.trim() || normalized;
  if (clause.length <= maxLength) return clause;
  const candidate = clause.slice(0, maxLength - 1);
  const boundary = candidate.lastIndexOf(" ");
  const clipped = boundary >= Math.floor(maxLength * 0.55) ? candidate.slice(0, boundary) : candidate;
  return `${clipped.trim()}…`;
}

function stageLabel(stage) {
  return stage === "meta_review" ? "Meta-review" : humanizeKey(stage);
}

function relativeTime(value) {
  const timestamp = Date.parse(value || "");
  if (!Number.isFinite(timestamp)) return "recently";
  const seconds = Math.max(0, Math.round((Date.now() - timestamp) / 1000));
  if (seconds < 45) return "just now";
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h ago`;
  if (seconds < 604800) return `${Math.round(seconds / 86400)}d ago`;
  return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" }).format(
    new Date(timestamp),
  );
}

function saveSessionHistory() {
  state.recentSessions = state.recentSessions.slice(0, 20);
  localStorage.setItem(SESSION_HISTORY_KEY, JSON.stringify(state.recentSessions));
}

function upsertRecentSession(workflow, touch = false) {
  const existing = state.recentSessions.find((item) => item.id === workflow.id);
  const entry = {
    id: workflow.id,
    title: deriveSessionName(workflow.question),
    query: workflow.question,
    status: workflow.status,
    stage: workflow.stage,
    createdAt: workflow.created_at || existing?.createdAt || new Date().toISOString(),
    updatedAt: workflow.updated_at || existing?.updatedAt || new Date().toISOString(),
    lastOpenedAt: touch
      ? new Date().toISOString()
      : existing?.lastOpenedAt || workflow.updated_at || new Date().toISOString(),
    unavailable: false,
    deleteToken: workflow.deletion_token || existing?.deleteToken || null,
  };
  state.recentSessions = [
    entry,
    ...state.recentSessions.filter((item) => item.id !== workflow.id),
  ].sort((left, right) => Date.parse(right.lastOpenedAt) - Date.parse(left.lastOpenedAt));
  saveSessionHistory();
  renderSessionHistory();
}

function markSessionUnavailable(sessionId) {
  const session = state.recentSessions.find((item) => item.id === sessionId);
  if (session) session.unavailable = true;
  saveSessionHistory();
  renderSessionHistory();
}

function removeRecentSession(sessionId) {
  state.recentSessions = state.recentSessions.filter((item) => item.id !== sessionId);
  saveSessionHistory();
  if (localStorage.getItem(CURRENT_WORKFLOW_KEY) === sessionId) {
    localStorage.removeItem(CURRENT_WORKFLOW_KEY);
  }
  renderSessionHistory();
}

async function deleteCloudSession(sessionId) {
  const session = state.recentSessions.find((item) => item.id === sessionId);
  if (!session?.deleteToken) {
    toast("This browser does not hold the deletion credential");
    return;
  }
  const confirmed = window.confirm(
    `Permanently delete “${session.title}” and its cloud research record? This cannot be undone.`,
  );
  if (!confirmed) return;
  const response = await fetch(
    `/api/research/sessions/${encodeURIComponent(sessionId)}`,
    {
      method: "DELETE",
      headers: { "X-Session-Delete-Token": session.deleteToken },
    },
  );
  if (!response.ok) {
    let detail = `Deletion failed (${response.status})`;
    try {
      detail = (await response.json()).detail || detail;
    } catch {
      // Keep the HTTP status if a proxy returns no JSON body.
    }
    throw new Error(detail);
  }
  const wasCurrent = sessionId === state.workflowId;
  removeRecentSession(sessionId);
  if (wasCurrent) await newInquiry();
  toast("Cloud session permanently deleted");
}

function renderSessionHistory() {
  elements.sessionCount.textContent = `${state.recentSessions.length} saved`;
  if (!state.recentSessions.length) {
    elements.sessionHistory.innerHTML = `
      <div class="history-empty">
        <span>∅</span>
        <strong>No saved research yet</strong>
        <p>Guided inquiries created in this browser will appear here.</p>
      </div>`;
    return;
  }
  elements.sessionHistory.innerHTML = state.recentSessions
    .map(
      (item) => `
        <article class="session-history-item ${item.id === state.workflowId ? "current" : ""} ${
          item.unavailable ? "unavailable" : ""
        }" data-session-id="${escapeHtml(item.id)}">
          <button class="session-history-open" type="button" ${
            item.unavailable ? "disabled" : ""
          } title="${escapeHtml(item.query || item.title)}">
            <strong>${escapeHtml(item.title)}</strong>
            <span>${item.unavailable ? "Unavailable on this instance" : `${escapeHtml(stageLabel(item.stage))} · ${escapeHtml(relativeTime(item.updatedAt))}`}</span>
          </button>
          <div class="session-history-actions">
            ${
              item.deleteToken
                ? `<button class="session-delete-cloud" type="button" aria-label="Permanently delete ${escapeHtml(item.title)} from Google Cloud" title="Permanently delete from Google Cloud">⌫</button>`
                : ""
            }
            <button class="session-remove" type="button" aria-label="Remove ${escapeHtml(item.title)} from this browser" title="Remove from this browser">×</button>
          </div>
        </article>`,
    )
    .join("");
}

function updateSessionIdentity(workflow) {
  if (!workflow) {
    elements.currentSessionCard.hidden = true;
    return;
  }
  const title = deriveSessionName(workflow.question);
  elements.currentSessionCard.hidden = false;
  elements.currentSessionName.textContent = title;
  elements.currentSessionName.title = workflow.question;
  elements.currentSessionName.setAttribute("aria-label", `Current query: ${workflow.question}`);
  elements.currentSessionState.textContent = `${stageLabel(workflow.stage)} · ${workflowStatusCopy(workflow)}`;
}

function updateStageNavigation(workflow = null) {
  const previews = new Map(
    (workflow?.stage_previews || []).map((preview) => [preview.stage, preview]),
  );
  document.querySelectorAll(".stage-nav li").forEach((item) => {
    const stage = item.dataset.stage;
    const button = item.querySelector("button");
    const report = stage === "report";
    const metadata = previews.get(stage);
    const available = report ? Boolean(workflow?.report_available) : Boolean(metadata?.available);
    item.hidden = report && !available;
    button.disabled = !available;
    item.classList.toggle("active", !report && workflow?.stage === stage);
    item.classList.toggle("done", Boolean(metadata?.is_completed));
    item.classList.toggle("viewing", state.viewingStage === stage);
    if (available) {
      button.title = report
        ? "View the complete research dossier"
        : `View saved ${stageLabel(stage)} output`;
    }
  });
}

async function createSession() {
  setConnection("", "Preparing research session");
  const response = await fetch(
    `/apps/${APP_NAME}/users/${encodeURIComponent(state.userId)}/sessions`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        state: {
          client: "coscientist-web",
          research_integrity_notice: true,
        },
      }),
    },
  );
  if (!response.ok) {
    throw new Error(`Session service returned ${response.status}`);
  }
  const session = await response.json();
  state.sessionId = session.id;
  setConnection("ready", "Session ready · global reasoning");
}

async function ensureSession() {
  if (!state.sessionId) await createSession();
}

function extractEventText(event) {
  const parts = event?.content?.parts || [];
  return parts
    .map((part) => part?.text || "")
    .filter(Boolean)
    .join("");
}

async function streamResearch(prompt, pending) {
  const response = await fetch("/run_sse", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      app_name: APP_NAME,
      user_id: state.userId,
      session_id: state.sessionId,
      new_message: { role: "user", parts: [{ text: prompt }] },
      streaming: true,
    }),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`Research service returned ${response.status}: ${detail.slice(0, 180)}`);
  }
  if (!response.body) throw new Error("Streaming is unavailable in this browser.");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let accumulated = "";
  let activeAuthor = "co_scientist_supervisor";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (!line.startsWith("data:")) continue;
      const payload = line.slice(5).trim();
      if (!payload) continue;
      try {
        const event = JSON.parse(payload);
        if (event.author) {
          activeAuthor = event.author;
          setActiveAgent(activeAuthor);
        }
        const text = extractEventText(event);
        if (text) {
          accumulated += text;
          updateAssistant(pending, accumulated, activeAuthor);
        }
      } catch {
        // Ignore keep-alive or non-JSON SSE frames.
      }
    }
  }

  if (!accumulated) {
    updateAssistant(
      pending,
      "The workflow completed without a textual artifact. Try restating the requested deliverable.",
      activeAuthor,
    );
  }
}

async function researchApi(path, options = {}) {
  const response = await fetch(`/api/research${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  if (!response.ok) {
    let detail = `Research workflow returned ${response.status}`;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch {
      // Keep the HTTP status when a proxy returns a non-JSON error.
    }
    throw new Error(detail);
  }
  return response.json();
}

function workflowStatusCopy(workflow) {
  if (workflow.status === "input_required") return "Scientific input required";
  if (workflow.status === "evidence_required") return "Verified evidence required";
  if (workflow.status === "governance_blocked") return "Governance review blocked";
  if (workflow.status === "stopped_by_researcher") return "Stopped by researcher";
  if (workflow.status === "ready_for_report") return "Dossier ready";
  if (workflow.requires_human_approval) return "Human decision required";
  return "Workflow active";
}

function updateApprovalIndicator() {
  const label = elements.approvalIndicator.querySelector("strong");
  const detail = elements.approvalIndicator.querySelector("span");
  const profile = elements.approvalProfile.value;
  if (state.mode === "conversation") {
    label.textContent = "Conversational mode";
    detail.textContent = "No stage promotion controls";
  } else if (profile === "auto") {
    label.textContent = "Automatic workflow";
    detail.textContent = "Mandatory integrity gates still apply";
  } else {
    label.textContent = "Human review enabled";
    detail.textContent =
      profile === "milestone"
        ? "Scope, ranking, evolution, and synthesis are audited"
        : profile === "stage"
          ? "Every stage promotion requires a decision"
          : "Every specialist artifact requires approval";
  }
}

function unresolvedRequirements(workflow) {
  return workflow.input_requirements.filter(
    (item) => item.blocking && !["provided", "fallback_accepted"].includes(item.status),
  );
}

function evidenceProgressCopy(workflow) {
  const presentation = workflow.evidence_progress;
  if (!presentation) return "";
  const metric = Object.fromEntries(
    (presentation.metrics || []).map((item) => [item.label, item.value]),
  );
  return `Deep Research pass ${metric["Deep Research passes"] || 1} of 3 · ${
    metric.Coverage || 0
  }% coverage · ${metric["Source leads"] || 0} source leads · discovered, not yet verified`;
}

function renderApprovalCard(workflow) {
  const existing = document.querySelector(".approval-card:not(.resolved)");
  if (
    workflow.status === "ready_for_report" ||
    workflow.status === "stopped_by_researcher"
  ) {
    if (existing) existing.remove();
    return;
  }

  const requirements = unresolvedRequirements(workflow);
  const artifactCount = workflow.pending_artifacts.length;
  const decisionRequired = workflow.requires_human_approval;
  const operation = workflow.operation || { status: "idle", detail: "" };
  const operationActive = ["queued", "running"].includes(operation.status);
  const operationFailed =
    operation.status === "failed" ||
    (operation.status === "idle" &&
      workflow.status === "active" &&
      !workflow.pending_draft);
  const gateKey = `${workflow.id}:${workflow.stage}:${
    workflow.pending_draft?.id || operation.kind || "preparing"
  }`;
  if (
    existing &&
    existing.dataset.gateKey === gateKey &&
    operationActive &&
    existing.querySelector(".workflow-progress")
  ) {
    existing.querySelector(".approval-stage-progress").textContent =
      `Stage ${workflow.stage_number} of ${workflow.stage_count} · ${workflow.task_summary.completed}/${workflow.task_summary.total} specialist tasks complete`;
    existing.querySelector(".workflow-progress p").textContent =
      operation.detail || "Preparing the next review gate…";
    const evidenceLine = existing.querySelector(".evidence-live-progress");
    if (evidenceLine) evidenceLine.textContent = evidenceProgressCopy(workflow);
    existing.querySelector(".approval-badge").textContent = "In progress";
    return;
  }
  if (existing) existing.remove();

  const card = document.createElement("section");
  card.className = "approval-card";
  card.dataset.workflowId = workflow.id;
  card.dataset.gateKey = gateKey;
  card.innerHTML = `
    <div class="approval-card-head">
      <div>
        <p class="eyebrow">Supervisor gate · ${escapeHtml(workflow.approval_profile)}</p>
        <h3>${escapeHtml(workflow.stage.replaceAll("_", " "))} · ${
          operationActive
            ? "Specialists working"
            : operationFailed
              ? "Stage preparation interrupted"
              : escapeHtml(workflowStatusCopy(workflow))
        }</h3>
        <p class="approval-stage-progress">Stage ${workflow.stage_number} of ${workflow.stage_count} · ${workflow.task_summary.completed}/${workflow.task_summary.total} specialist tasks complete</p>
      </div>
      <span class="approval-badge">${
        operationActive ? "In progress" : decisionRequired ? "Your decision" : "Integrity gate"
      }</span>
    </div>
    ${
      operationActive
        ? `<div class="workflow-progress"><span></span><p>${escapeHtml(operation.detail || "Preparing the next review gate…")}</p></div>
           ${
             workflow.stage === "evidence"
               ? `<p class="evidence-live-progress">${escapeHtml(evidenceProgressCopy(workflow))}</p>
                  <p class="stored-interaction-notice">Deep Research uses stored Gemini interactions for background execution.</p>`
               : ""
           }`
        : ""
    }
    ${
      operationFailed
        ? `<p class="operation-error">${escapeHtml(operation.detail || "The next stage could not be prepared.")}</p>
           <div class="approval-actions"><button class="primary" type="button" data-decision="continue">Retry stage preparation</button></div>`
        : ""
    }
    <div class="requirement-list"></div>
    ${
      workflow.status === "evidence_required"
        ? `<div class="input-requirement evidence-warning">
            <strong>Discovery could not satisfy the verification gate</strong>
            <p>Retry Deep Research, or explicitly continue in a limited exploratory mode. Downstream output will remain hypotheses—not evidence-backed findings.</p>
            <div class="input-actions">
              <button type="button" data-decision="continue">Retry evidence stage</button>
              <button type="button" data-decision="exploratory_evidence">Continue as exploratory</button>
            </div>
          </div>`
        : ""
    }
    <div class="artifact-list"></div>
    ${
      operationActive || operationFailed
        ? ""
        : `
          <div class="direct-editor" hidden>
            <label>Directly edit this stage draft</label>
            <textarea class="direct-edit-field" rows="14"></textarea>
            <p>Your saved edit becomes a new auditable artifact version.</p>
            <button class="save-edit" type="button" data-decision="edit">Save edited version</button>
          </div>
          <textarea class="revision-field" rows="2" placeholder="Or describe what the agent should revise…"></textarea>
          <div class="approval-actions">
            <button class="primary" type="button" data-decision="accept" ${
              requirements.length || artifactCount ? "disabled" : ""
            }>Accept &amp; continue</button>
            <button type="button" data-decision="toggle_edit">Edit draft directly</button>
            <button type="button" data-decision="revise">Ask agent to revise</button>
            <button class="danger" type="button" data-decision="stop">Stop workflow</button>
          </div>
        `
    }
  `;

  const requirementList = card.querySelector(".requirement-list");
  requirements.forEach((item) => {
    const block = document.createElement("div");
    block.className = "input-requirement";
    block.innerHTML = `
      <strong>${escapeHtml(item.description)}</strong>
      <p>${escapeHtml(item.reason)}</p>
      <input class="input-reference" type="text" placeholder="Dataset, sequence, file, DOI, or other input reference" />
      <div class="input-actions">
        <button type="button" data-decision="provide_input" data-input-type="${escapeHtml(item.input_type)}">Provide reference</button>
        ${
          item.permitted_fallback === "literature_only"
            ? '<button type="button" data-decision="literature_only">Use literature-only mode</button>'
            : ""
        }
      </div>
    `;
    requirementList.append(block);
  });

  const artifactList = card.querySelector(".artifact-list");
  workflow.pending_artifacts.forEach((item) => {
    const block = document.createElement("div");
    block.className = "artifact-review";
    block.innerHTML = `
      <strong>${escapeHtml(item.agent.replaceAll("_", " "))}</strong>
      <div class="artifact-presentation">${renderStagePresentation(item.presentation, item.content)}</div>
      <div class="artifact-actions">
        <button type="button" data-decision="approve_artifact" data-artifact-id="${escapeHtml(item.id)}">Approve artifact</button>
      </div>
    `;
    artifactList.append(block);
  });

  const directEditField = card.querySelector(".direct-edit-field");
  if (directEditField && workflow.pending_draft) {
    directEditField.value = workflow.pending_draft.content;
  }
  card.addEventListener("click", handleDecisionClick);
  elements.messages.append(card);
}

function stopWorkflowPolling() {
  if (state.pollTimer) window.clearTimeout(state.pollTimer);
  state.pollTimer = null;
}

function pollWorkflow() {
  stopWorkflowPolling();
  state.pollTimer = window.setTimeout(async () => {
    if (!state.workflowId) return;
    try {
      const workflow = await researchApi(
        `/sessions/${encodeURIComponent(state.workflowId)}`,
      );
      renderWorkflow(workflow);
    } catch (error) {
      setConnection("error", "Could not refresh workflow progress");
      toast(error instanceof Error ? error.message : "Progress check failed");
    }
  }, 1400);
}

function clearWorkflowDisplay() {
  elements.messages.replaceChildren();
  elements.messages.hidden = true;
  elements.welcome.hidden = false;
  state.lastDraftId = null;
  state.autoFollow = true;
  elements.jumpLatest.hidden = true;
}

function setPreviewMode(enabled) {
  document.body.classList.toggle("history-preview", enabled);
  elements.input.disabled = enabled || state.busy;
  elements.send.disabled = enabled || state.busy;
}

function renderStagePreview(preview) {
  state.viewingStage = preview.stage;
  setPreviewMode(true);
  updateStageNavigation(state.workflow);
  elements.messages.replaceChildren();
  elements.messages.hidden = false;
  elements.welcome.hidden = true;

  const banner = document.createElement("section");
  banner.className = "historical-preview-banner";
  banner.innerHTML = `
    <div>
      <p class="eyebrow">Read-only research history</p>
      <h2>${escapeHtml(preview.stage === "report" ? "Research dossier" : `${stageLabel(preview.stage)} output`)}</h2>
      <p>${
        preview.stage === "report"
          ? "Completed dossier · preserved workflow record"
          : `Version ${escapeHtml(String(preview.version))} · ${escapeHtml(String(preview.status))} · ${escapeHtml(agentLabels[preview.producer] || humanizeKey(preview.producer || "Supervisor"))} · ${escapeHtml(new Date(preview.created_at).toLocaleString())}`
      }</p>
    </div>
    <button type="button" data-return-current>Return to current gate</button>
  `;
  elements.messages.append(banner);
  appendMessage(
    "assistant",
    preview.content,
    preview.producer || (preview.stage === "report" ? "meta_reviewer" : "co_scientist_supervisor"),
    preview.presentation,
  );
  if (preview.stage === "report" && state.workflow) {
    renderReportCompletion(state.workflow);
  }
  setConnection("ready", `Viewing saved ${stageLabel(preview.stage)} output`);
}

function renderReportCompletion(workflow) {
  const existing = document.querySelector(".report-completion");
  if (existing) existing.remove();
  const exports = workflow.report_exports || [];
  const panel = document.createElement("section");
  panel.className = "report-completion";
  panel.innerHTML = `
    <div class="report-completion-mark" aria-hidden="true">✓</div>
    <div class="report-completion-copy">
      <p class="eyebrow">Research workflow complete</p>
      <h2>Your dossier is ready</h2>
      <p>Download an editable Word document for Google Docs, a publication-ready PDF, or the complete Markdown research record.</p>
      <div class="report-export-actions">
        ${exports
          .map(
            (item, index) => `
              <a class="${index === 0 ? "primary" : ""}" href="${escapeHtml(item.url)}" download="${escapeHtml(item.filename)}">
                ${escapeHtml(item.label)}
              </a>`,
          )
          .join("")}
      </div>
      <small>The DOCX file opens directly in Google Docs and remains editable. Direct Drive creation requires user OAuth and is intentionally not requested by this public service.</small>
    </div>
  `;
  elements.messages.append(panel);
}

function returnToCurrentGate() {
  if (!state.workflow) return;
  state.viewingStage = null;
  setPreviewMode(false);
  clearWorkflowDisplay();
  renderWorkflow(state.workflow);
}

async function viewStagePreview(stage) {
  if (!state.workflowId || state.busy) return;
  try {
    let preview;
    if (stage === "report") {
      if (!state.workflow?.report) throw new Error("The dossier is not available yet.");
      preview = {
        stage: "report",
        content: state.workflow.report,
        presentation: state.workflow.report_presentation,
        producer: "meta_reviewer",
        created_at: state.workflow.updated_at,
        version: 1,
        status: "accepted",
      };
    } else {
      preview = await researchApi(
        `/sessions/${encodeURIComponent(state.workflowId)}/stages/${encodeURIComponent(stage)}`,
      );
    }
    renderStagePreview(preview);
    document.body.classList.remove("menu-open");
  } catch (error) {
    toast(error instanceof Error ? error.message : "Stage output could not be loaded");
  }
}

function renderWorkflow(workflow, pending = null, touchHistory = false) {
  state.workflow = workflow;
  state.workflowId = workflow.id;
  state.sessionId = workflow.id;
  updateSessionIdentity(workflow);
  updateStageNavigation(workflow);
  upsertRecentSession(workflow, touchHistory);
  localStorage.setItem(CURRENT_WORKFLOW_KEY, workflow.id);
  const operationActive = ["queued", "running"].includes(
    workflow.operation?.status,
  );

  if (state.viewingStage) {
    if (pending) pending.remove();
    if (operationActive) pollWorkflow();
    else stopWorkflowPolling();
    return;
  }

  const draft = workflow.pending_draft;
  if (draft && draft.id !== state.lastDraftId) {
    if (pending) {
      updateAssistant(
        pending,
        draft.content,
        "co_scientist_supervisor",
        draft.presentation,
      );
    } else {
      appendMessage(
        "assistant",
        draft.content,
        "co_scientist_supervisor",
        draft.presentation,
      );
    }
    state.lastDraftId = draft.id;
  } else if (pending) {
    pending.remove();
  }

  if (workflow.report && workflow.status === "ready_for_report") {
    stopWorkflowPolling();
    appendMessage(
      "assistant",
      workflow.report,
      "meta_reviewer",
      workflow.report_presentation,
    );
    renderReportCompletion(workflow);
    state.lastDraftId = null;
    setConnection("ready", "Dossier ready · decisions preserved");
  } else {
    renderApprovalCard(workflow);
    setConnection(
      workflow.status.includes("blocked") || workflow.status === "input_required"
        ? "error"
        : "ready",
      operationActive ? "Specialists are preparing the next gate" : workflowStatusCopy(workflow),
    );
    if (operationActive) pollWorkflow();
    else stopWorkflowPolling();
  }
}

async function createGuidedWorkflow(prompt, pending) {
  const workflow = await researchApi("/sessions", {
    method: "POST",
    body: JSON.stringify({
      question: prompt,
      approval_profile: elements.approvalProfile.value,
    }),
  });
  renderWorkflow(workflow, pending, true);
}

function selectMode(mode) {
  state.mode = mode;
  localStorage.setItem("coscientist.mode", state.mode);
  document
    .querySelectorAll("[data-mode]")
    .forEach((item) => item.classList.toggle("active", item.dataset.mode === mode));
  elements.approvalProfile.hidden = mode === "conversation";
  elements.approvalProfile.closest(".profile-control").hidden =
    mode === "conversation";
  updateApprovalIndicator();
}

async function openResearchSession(sessionId, { restore = false } = {}) {
  if (state.busy) return;
  stopWorkflowPolling();
  state.viewingStage = null;
  setPreviewMode(false);
  clearWorkflowDisplay();
  selectMode("guided");
  setConnection("", "Restoring governed research session");
  try {
    const workflow = await researchApi(`/sessions/${encodeURIComponent(sessionId)}`);
    renderWorkflow(workflow, null, true);
    document.body.classList.remove("history-open");
    if (!restore) toast("Research session restored");
  } catch (error) {
    markSessionUnavailable(sessionId);
    if (localStorage.getItem(CURRENT_WORKFLOW_KEY) === sessionId) {
      localStorage.removeItem(CURRENT_WORKFLOW_KEY);
    }
    state.workflowId = null;
    state.workflow = null;
    updateSessionIdentity(null);
    updateStageNavigation(null);
    setConnection("error", "Saved session is unavailable on this instance");
    if (!restore) toast(error instanceof Error ? error.message : "Session unavailable");
  }
}

async function handleDecisionClick(event) {
  const button = event.target.closest("[data-decision]");
  if (!button || state.busy || !state.workflowId) return;
  const card = button.closest(".approval-card");
  const action = button.dataset.decision;
  if (action === "toggle_edit") {
    const editor = card.querySelector(".direct-editor");
    editor.hidden = !editor.hidden;
    if (!editor.hidden) {
      editor.querySelector(".direct-edit-field").focus();
      editor.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
    return;
  }
  const payload = { action };
  if (action === "revise") {
    payload.feedback = card.querySelector(".revision-field").value.trim();
    if (!payload.feedback) {
      toast("Describe the required revision first");
      card.querySelector(".revision-field").focus();
      return;
    }
  }
  if (action === "approve_artifact") {
    payload.artifact_id = button.dataset.artifactId;
  }
  if (action === "edit") {
    payload.content = card.querySelector(".direct-edit-field").value.trim();
    if (!payload.content) {
      toast("The edited draft cannot be empty");
      return;
    }
  }
  if (action === "provide_input") {
    payload.input_type = button.dataset.inputType;
    payload.input_reference = button
      .closest(".input-requirement")
      .querySelector(".input-reference")
      .value.trim();
    if (!payload.input_reference) {
      toast("Add an input reference first");
      return;
    }
  }

  state.busy = true;
  card.querySelectorAll("button").forEach((item) => (item.disabled = true));
  setConnection("", action === "accept" ? "Advancing to the next gate" : "Recording decision");
  try {
    const workflow = await researchApi(
      `/sessions/${encodeURIComponent(state.workflowId)}/decisions`,
      { method: "POST", body: JSON.stringify(payload) },
    );
    card.classList.add("resolved");
    card.querySelector(".approval-badge").textContent = "Recorded";
    renderWorkflow(workflow);
  } catch (error) {
    toast(error instanceof Error ? error.message : "Decision could not be recorded");
    try {
      const workflow = await researchApi(
        `/sessions/${encodeURIComponent(state.workflowId)}`,
      );
      renderWorkflow(workflow);
    } catch {
      setConnection("error", "Could not refresh workflow");
    }
  } finally {
    state.busy = false;
  }
}

async function submitPrompt(prompt) {
  const cleaned = prompt.trim();
  if (!cleaned || state.busy) return;

  state.busy = true;
  elements.send.disabled = true;
  elements.input.disabled = true;
  appendMessage("user", cleaned);
  const pending = appendPending();
  setConnection("", "Specialists are reasoning");

  try {
    if (state.mode === "guided") {
      await createGuidedWorkflow(cleaned, pending);
    } else {
      await ensureSession();
      await streamResearch(cleaned, pending);
      setConnection("ready", "Session ready · global reasoning");
    }
  } catch (error) {
    showError(pending, error instanceof Error ? error.message : "Unexpected error");
    setConnection("error", "Research service unavailable");
  } finally {
    state.busy = false;
    elements.send.disabled = Boolean(state.viewingStage);
    elements.input.disabled = Boolean(state.viewingStage);
    elements.input.value = "";
    resizeInput();
    elements.input.focus();
  }
}

function resizeInput() {
  elements.input.style.height = "auto";
  elements.input.style.height = `${Math.min(elements.input.scrollHeight, 180)}px`;
}

async function newInquiry() {
  if (state.busy) return;
  stopWorkflowPolling();
  state.sessionId = null;
  state.workflowId = null;
  state.workflow = null;
  state.lastDraftId = null;
  state.viewingStage = null;
  localStorage.removeItem(CURRENT_WORKFLOW_KEY);
  setPreviewMode(false);
  state.inquiry += 1;
  localStorage.setItem("coscientist.inquiry", String(state.inquiry));
  updateInquiryNumber();
  clearWorkflowDisplay();
  updateSessionIdentity(null);
  updateStageNavigation(null);
  renderSessionHistory();
  setActiveAgent("co_scientist_supervisor");
  if (state.mode === "conversation") {
    try {
      await createSession();
      toast("New research session created");
    } catch {
      setConnection("error", "Could not create session");
    }
  } else {
    setConnection("ready", "Ready for a governed inquiry");
    toast("New inquiry ready");
  }
}

elements.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  submitPrompt(elements.input.value);
});

elements.input.addEventListener("input", resizeInput);
elements.input.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    elements.composer.requestSubmit();
  }
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    elements.input.value = button.dataset.prompt || "";
    resizeInput();
    elements.input.focus();
  });
});

elements.newInquiry.addEventListener("click", newInquiry);
elements.copySession.addEventListener("click", async () => {
  if (!state.sessionId) {
    toast("Session is still being prepared");
    return;
  }
  await navigator.clipboard.writeText(state.sessionId);
  toast("Session identifier copied");
});

elements.mobileMenu.addEventListener("click", () => {
  document.body.classList.toggle("menu-open");
});

elements.conversation.addEventListener("scroll", () => {
  state.autoFollow = nearConversationBottom();
  elements.jumpLatest.hidden = state.autoFollow;
});
elements.jumpLatest.addEventListener("click", () => {
  state.autoFollow = true;
  elements.jumpLatest.hidden = true;
  scrollToBottom(true);
});

document.querySelectorAll("[data-mode]").forEach((button) => {
  button.classList.toggle("active", button.dataset.mode === state.mode);
  button.addEventListener("click", () => {
    if (state.busy) return;
    selectMode(button.dataset.mode);
    newInquiry();
  });
});
document.querySelector(".stage-nav").addEventListener("click", (event) => {
  const button = event.target.closest("li[data-stage] > button");
  if (!button || button.disabled) return;
  viewStagePreview(button.closest("li").dataset.stage);
});
elements.messages.addEventListener("click", (event) => {
  if (event.target.closest("[data-return-current]")) returnToCurrentGate();
});
elements.sessionHistory.addEventListener("click", (event) => {
  const item = event.target.closest(".session-history-item");
  if (!item) return;
  if (event.target.closest(".session-remove")) {
    removeRecentSession(item.dataset.sessionId);
    return;
  }
  if (event.target.closest(".session-delete-cloud")) {
    deleteCloudSession(item.dataset.sessionId).catch((error) =>
      toast(error instanceof Error ? error.message : "Cloud deletion failed"),
    );
    return;
  }
  if (event.target.closest(".session-history-open")) {
    openResearchSession(item.dataset.sessionId);
  }
});
elements.historyButton.addEventListener("click", () => {
  document.body.classList.add("history-open");
});
elements.closeHistory.addEventListener("click", () => {
  document.body.classList.remove("history-open");
});
elements.approvalProfile.addEventListener("change", updateApprovalIndicator);
document.addEventListener("click", (event) => {
  if (
    document.body.classList.contains("menu-open") &&
    !event.target.closest(".navigation") &&
    !event.target.closest("#mobileMenu")
  ) {
    document.body.classList.remove("menu-open");
  }
  if (
    document.body.classList.contains("history-open") &&
    !event.target.closest(".context-panel") &&
    !event.target.closest("#historyButton")
  ) {
    document.body.classList.remove("history-open");
  }
});

updateInquiryNumber();
setActiveAgent("co_scientist_supervisor");
selectMode(state.mode);
renderSessionHistory();
updateStageNavigation(null);
if (state.mode === "conversation") {
  createSession().catch(() => setConnection("error", "Could not create session"));
} else {
  const currentWorkflowId = localStorage.getItem(CURRENT_WORKFLOW_KEY);
  if (currentWorkflowId) {
    openResearchSession(currentWorkflowId, { restore: true });
  } else {
    setConnection("ready", "Ready for a governed inquiry");
  }
}
