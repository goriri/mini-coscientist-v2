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

// Remembered only so a second finding in the same block does not have to be
// signed again from scratch. The name is still shown in an editable field on
// every finding, because whoever answers this one may not be who answered the
// last one.
const ADJUDICATOR_KEY = "coscientist.adjudicator";

function rememberedAdjudicator() {
  return localStorage.getItem(ADJUDICATOR_KEY) || "";
}

const NOTIFY_KEY = "coscientist.stageAlerts";
// Below this, the stage finished while the researcher was still looking at the
// tab they switched to, and a notification is an interruption rather than a
// recall. Deep Research and the tournament both run far longer than this.
const LONG_STAGE_MILLISECONDS = 30000;

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
  viewingStage: null,
  recentSessions: loadSessionHistory(),
  notifyStageAlerts: localStorage.getItem(NOTIFY_KEY) !== "off",
  operationStartedAt: null,
  notifiedGateKey: null,
  pendingAttention: false,
  optionsReady: null,
  renamingId: null,
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
  sessionTitle: document.querySelector("#sessionTitle"),
  sessionTitleInput: document.querySelector("#sessionTitleInput"),
  renameSession: document.querySelector("#renameSession"),
  toast: document.querySelector("#toast"),
  mobileMenu: document.querySelector("#mobileMenu"),
  conversation: document.querySelector(".conversation"),
  jumpLatest: document.querySelector("#jumpLatest"),
  approvalProfile: document.querySelector("#approvalProfile"),
  modelChoice: document.querySelector("#modelChoice"),
  languageChoice: document.querySelector("#languageChoice"),
  evidenceReview: document.querySelector("#evidenceReview"),
  evidenceReviewNote: document.querySelector("#evidenceReviewNote"),
  currentSessionCard: document.querySelector("#currentSessionCard"),
  currentSessionName: document.querySelector("#currentSessionName"),
  currentSessionState: document.querySelector("#currentSessionState"),
  sessionHistory: document.querySelector("#sessionHistory"),
  sessionCount: document.querySelector("#sessionCount"),
  historyButton: document.querySelector("#historyButton"),
  closeHistory: document.querySelector("#closeHistory"),
  notifySection: document.querySelector("#notifySection"),
  notifyEnabled: document.querySelector("#notifyEnabled"),
  notifyDetail: document.querySelector("#notifyDetail"),
};

const DOCUMENT_TITLE = document.title;

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
        .map(
          (paragraph) =>
            `<p>${formatInline(paragraph).replaceAll("\n", "<br>")}</p>`,
        )
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
  if (value === null || value === "")
    return '<p class="empty-value">Not specified</p>';
  return `<p>${formatInline(String(value))}</p>`;
}

function formatArtifactSection(section) {
  const headingMatch = section.match(/^###\s+([^\n]+)\n*/);
  const heading = headingMatch ? headingMatch[1].trim() : "";
  const body = headingMatch
    ? section.slice(headingMatch[0].length).trim()
    : section.trim();
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
  const sections = value
    .trim()
    .split(/(?=^###\s+)/m)
    .filter(Boolean);
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
  if (value === null || value === undefined || value === "")
    return "Not specified";
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

function renderEvidenceSource(source) {
  const claims = (source.claims || [])
    .map(
      (claim) => `
        <li class="evidence-claim relation-${escapeHtml(claim.relation || "neutral")}">
          <span class="claim-relation">${escapeHtml(claim.relation_label || "")}</span>
          <span class="claim-text">${formatInline(claim.text || "")}</span>
          ${claim.location ? `<span class="claim-location">${escapeHtml(claim.location)}</span>` : ""}
        </li>`,
    )
    .join("");
  return `
    <li class="evidence-source tone-${escapeHtml(source.status_tone || "quarantined")}">
      <div class="evidence-source-head">
        <span class="status-chip tone-${escapeHtml(source.status_tone || "quarantined")}" title="${escapeHtml(source.status_meaning || "")}">${escapeHtml(source.status_label || "")}</span>
        <a href="${escapeHtml(source.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(source.title || source.url)}</a>
      </div>
      ${source.citation ? `<p class="evidence-citation">${escapeHtml(source.citation)}</p>` : ""}
      ${claims ? `<ul class="evidence-claims">${claims}</ul>` : '<p class="evidence-no-claim">No claim was attributed to this source.</p>'}
      ${source.verification_note ? `<p class="evidence-note">${formatInline(source.verification_note)}</p>` : ""}
    </li>`;
}

// The evidence panel is the one place a reader decides whether to trust the
// run at all, and the generic detail renderer flattened it into titles and
// URLs: forty-four rows, no facet, no claim, no verification outcome. This
// renders the three things that decision needs -- what kind of evidence exists,
// what each source was cited for, and how far each one has actually been
// checked -- and keeps the empty facets visible, because an absent line of
// evidence is a finding.
function renderEvidenceTrust(evidence) {
  if (!evidence) return "";
  const floorChecks = (evidence.floor_details || [])
    .map(
      (check) => `
        <li class="${check.met ? "met" : "unmet"}">
          <span>${escapeHtml(check.label)}</span>
          <strong>${escapeHtml(check.value)}</strong>
        </li>`,
    )
    .join("");
  const shortfalls = (evidence.shortfalls || [])
    .map((item) => `<li>${formatInline(item)}</li>`)
    .join("");
  const floor = floorChecks
    ? `<div class="evidence-floor ${evidence.floor?.met ? "met" : "unmet"}">
        <p class="evidence-floor-headline">${escapeHtml(evidence.headline || "")}</p>
        <ul class="evidence-floor-checks">${floorChecks}</ul>
        ${shortfalls ? `<ul class="evidence-shortfalls">${shortfalls}</ul>` : ""}
      </div>`
    : `<p class="evidence-pending">Sources have been discovered; verification has not run yet, so nothing here is confirmed.</p>`;
  const facets = (evidence.facets || [])
    .map((facet) => {
      const count = (facet.sources || []).length;
      const gaps = (facet.gaps || [])
        .map(
          (gap) =>
            `<li><span class="gap-impact impact-${escapeHtml(gap.impact || "medium")}">${escapeHtml(gap.impact || "medium")} impact</span> ${formatInline(gap.description || "")}</li>`,
        )
        .join("");
      return `
        <article class="evidence-facet ${count ? "" : "empty"}">
          <header>
            <h5>${escapeHtml(facet.label)}</h5>
            <span>${count ? `${count} usable source${count === 1 ? "" : "s"}` : "nothing usable"}</span>
          </header>
          ${
            count
              ? `<ul class="evidence-source-list">${facet.sources.map(renderEvidenceSource).join("")}</ul>`
              : `<p class="evidence-empty-facet">No source that survived verification covers this.</p>`
          }
          ${gaps ? `<ul class="evidence-gaps">${gaps}</ul>` : ""}
        </article>`;
    })
    .join("");
  const quarantine = (evidence.quarantine || []).length
    ? `<details class="evidence-quarantine">
        <summary>${evidence.quarantine.length} source${evidence.quarantine.length === 1 ? "" : "s"} nothing may rest on</summary>
        <ul class="evidence-source-list">${evidence.quarantine.map(renderEvidenceSource).join("")}</ul>
      </details>`
    : "";
  const legend = (evidence.legend || [])
    .map(
      (item) =>
        `<div><dt><span class="status-chip tone-${escapeHtml(item.tone)}">${escapeHtml(item.label)}</span></dt><dd>${escapeHtml(item.meaning)}</dd></div>`,
    )
    .join("");
  return `
    <section class="evidence-trust">
      ${floor}
      <div class="evidence-facet-list">${facets}</div>
      ${quarantine}
      ${legend ? `<details class="evidence-legend"><summary>What these verification labels mean</summary><dl class="compact-fields">${legend}</dl></details>` : ""}
    </section>`;
}

function renderStagePresentation(presentation, rawText = "") {
  if (!presentation) return formatText(rawText);
  const metrics = (presentation.metrics || [])
    .map(
      (metric) =>
        `<div><strong>${renderDisplayValue(metric.value)}${metric.unit ? `<em>${escapeHtml(metric.unit)}</em>` : ""}</strong><span>${escapeHtml(metric.label)}</span></div>`,
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
  // The tournament in prose, above the matches rather than after them. A reader
  // who opened "Tournament details" met eighteen collapsed rounds and had to
  // read all of them to learn what the ranking had decided. When the judge did
  // not write one the fallback is arithmetic over the same matches, and it says
  // so: it is not a reading of the hypotheses and must not pass for one.
  const briefing = presentation.briefing
    ? `<div class="tournament-briefing">
         ${presentation.briefing_author === "judge" ? "" : '<p class="briefing-source">Computed from the match record — the judge did not write a summary for this run.</p>'}
         ${formatPlainText(presentation.briefing)}
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
      ${comparisons || briefing ? `<section class="comparison-list"><h4>Tournament details</h4>${briefing}${comparisons}</section>` : ""}
      ${evolution ? `<div class="evolution-list">${evolution}</div>` : ""}
      ${clusters ? `<div class="cluster-grid">${clusters}</div>` : ""}
      ${recommendations ? `<div class="recommendation-grid">${recommendations}</div>` : ""}
      ${renderEvidenceTrust(presentation.evidence)}
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
  const label =
    role === "user" ? "Researcher" : agentLabels[author] || "Co-Scientist";
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

const SMOOTH_SCROLL_CEILING = 2400;
/* How far Latest update will animate. Past this it jumps: a finished dossier
   ran to 51,372 px of conversation in a 146 px pane on the live service, and
   the animated scroll did not arrive -- it stopped 2,046 px down and stayed
   there, so the scroll handler decided the reader was no longer following and
   put the button back. The short hop is worth animating because it shows the
   reader which way the new material lies; the long one is a jump to the end of
   a document, and a jump is what it should be. */

function scrollToBottom(force = false) {
  if (!force && !state.autoFollow) return;
  window.requestAnimationFrame(() => {
    const area = elements.conversation;
    const distance = area.scrollHeight - area.scrollTop - area.clientHeight;
    // A behavior passed here beats the stylesheet's scroll-behavior, so the
    // reduced-motion rule in styles.css never reached this scroll at all.
    const stillMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    area.scrollTo({
      top: area.scrollHeight,
      behavior:
        force && !stillMotion && distance <= SMOOTH_SCROLL_CEILING
          ? "smooth"
          : "auto",
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

function deriveSessionName(question, maxLength = 52) {
  const normalized = String(question || "")
    .replace(/\s+/g, " ")
    .trim();
  if (!normalized) return "Untitled inquiry";
  const clause =
    normalized
      .split(/(?:[.!?;:\n]|\s[—–]\s|,\s+(?:and|but|while|whereas)\b)/i)
      .find((part) => part.trim().length >= 3)
      ?.trim() || normalized;
  if (clause.length <= maxLength) return clause;
  const candidate = clause.slice(0, maxLength - 1);
  const boundary = candidate.lastIndexOf(" ");
  const clipped =
    boundary >= Math.floor(maxLength * 0.55)
      ? candidate.slice(0, boundary)
      : candidate;
  return `${clipped.trim()}…`;
}

// A session has one name everywhere it is shown: the topbar, the sidebar card,
// the history list, and the notification body. It starts as the first clause of
// the opening question and stays that way until someone renames it.
function sessionTitle(workflow) {
  if (!workflow) return "New inquiry";
  const saved = state.recentSessions.find((item) => item.id === workflow.id);
  return (
    saved?.customTitle || saved?.title || deriveSessionName(workflow.question)
  );
}

function historyTitle(entry) {
  return entry.customTitle || entry.title;
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
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
  }).format(new Date(timestamp));
}

function saveSessionHistory() {
  state.recentSessions = state.recentSessions.slice(0, 20);
  localStorage.setItem(
    SESSION_HISTORY_KEY,
    JSON.stringify(state.recentSessions),
  );
}

function upsertRecentSession(workflow, touch = false) {
  const existing = state.recentSessions.find((item) => item.id === workflow.id);
  const entry = {
    id: workflow.id,
    title: deriveSessionName(workflow.question),
    // Re-derived on every poll; a name the researcher typed is not.
    customTitle: existing?.customTitle || null,
    query: workflow.question,
    status: workflow.status,
    stage: workflow.stage,
    createdAt:
      workflow.created_at || existing?.createdAt || new Date().toISOString(),
    updatedAt:
      workflow.updated_at || existing?.updatedAt || new Date().toISOString(),
    lastOpenedAt: touch
      ? new Date().toISOString()
      : existing?.lastOpenedAt ||
        workflow.updated_at ||
        new Date().toISOString(),
    unavailable: false,
    deleteToken: workflow.deletion_token || existing?.deleteToken || null,
  };
  state.recentSessions = [
    entry,
    ...state.recentSessions.filter((item) => item.id !== workflow.id),
  ].sort(
    (left, right) =>
      Date.parse(right.lastOpenedAt) - Date.parse(left.lastOpenedAt),
  );
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
  state.recentSessions = state.recentSessions.filter(
    (item) => item.id !== sessionId,
  );
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
            <strong>${escapeHtml(historyTitle(item))}</strong>
            <span>${item.unavailable ? "Unavailable on this instance" : `${escapeHtml(stageLabel(item.stage))} · ${escapeHtml(relativeTime(item.updatedAt))}`}</span>
          </button>
          <div class="session-history-actions">
            <button class="session-rename" type="button" aria-label="Rename ${escapeHtml(historyTitle(item))}" title="Rename this session">✎</button>
            ${
              item.deleteToken
                ? `<button class="session-delete-cloud" type="button" aria-label="Permanently delete ${escapeHtml(historyTitle(item))} from Google Cloud" title="Permanently delete from Google Cloud">⌫</button>`
                : ""
            }
            <button class="session-remove" type="button" aria-label="Remove ${escapeHtml(historyTitle(item))} from this browser" title="Remove from this browser">×</button>
          </div>
        </article>`,
    )
    .join("");
}

function updateSessionIdentity(workflow) {
  if (!workflow) {
    elements.currentSessionCard.hidden = true;
    renderSessionTitle(null);
    return;
  }
  const title = sessionTitle(workflow);
  elements.currentSessionCard.hidden = false;
  elements.currentSessionName.textContent = title;
  elements.currentSessionName.title = workflow.question;
  elements.currentSessionName.setAttribute(
    "aria-label",
    `Current query: ${workflow.question}`,
  );
  elements.currentSessionState.textContent = `${stageLabel(workflow.stage)} · ${workflowStatusCopy(workflow)}`;
  renderSessionTitle(workflow);
}

function renderSessionTitle(workflow = state.workflow) {
  const named = Boolean(workflow);
  const title = sessionTitle(workflow);
  elements.sessionTitle.textContent = title;
  elements.sessionTitle.title = workflow?.question || "";
  elements.renameSession.hidden = !named;
  document.title = state.pendingAttention
    ? `\u25cf ${title} · Co—Scientist`
    : named
      ? `${title} · Co—Scientist`
      : DOCUMENT_TITLE;
}

function beginRename(sessionId = state.workflowId) {
  const entry = state.recentSessions.find((item) => item.id === sessionId);
  if (!entry) return;
  state.renamingId = sessionId;
  elements.sessionTitleInput.value = historyTitle(entry);
  elements.sessionTitleInput.hidden = false;
  elements.sessionTitle.hidden = true;
  elements.renameSession.hidden = true;
  elements.sessionTitleInput.focus();
  elements.sessionTitleInput.select();
}

function endRename(commit) {
  if (!state.renamingId) return;
  const entry = state.recentSessions.find(
    (item) => item.id === state.renamingId,
  );
  if (commit && entry) {
    const typed = elements.sessionTitleInput.value.trim();
    // Clearing the field is how a researcher asks for the derived name back,
    // rather than pinning an empty heading over the workspace.
    entry.customTitle = typed && typed !== entry.title ? typed : null;
    saveSessionHistory();
    renderSessionHistory();
  }
  state.renamingId = null;
  elements.sessionTitleInput.hidden = true;
  elements.sessionTitle.hidden = false;
  renderSessionTitle();
  if (state.workflow) updateSessionIdentity(state.workflow);
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
    const available = report
      ? Boolean(workflow?.report_available)
      : Boolean(metadata?.available);
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
    throw new Error(
      `Research service returned ${response.status}: ${detail.slice(0, 180)}`,
    );
  }
  if (!response.body)
    throw new Error("Streaming is unavailable in this browser.");

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
        if (event.author) activeAuthor = event.author;
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

async function loadResearchOptions() {
  // Both selects stay empty until this resolves. Seeding them with a guess and
  // correcting it later would let a fast typist submit a model the server does
  // not serve, and the run would fail three stages in rather than at the form.
  let options;
  try {
    options = await researchApi("/options");
  } catch {
    setConnection("error", "Could not load model and language options");
    return;
  }
  fillChoices(elements.modelChoice, options.models, (item) => ({
    value: item.id,
    text: item.label,
    title: item.note,
    selected: item.default,
  }));
  fillChoices(elements.languageChoice, options.languages, (item) => ({
    // The endonym alongside the English name, because the person choosing to
    // run in Chinese is the person most likely to be reading for the Chinese.
    value: item.code,
    text:
      item.label === item.endonym
        ? item.label
        : `${item.label} · ${item.endonym}`,
    title: "",
    selected: item.default,
  }));
}

function notificationsAvailable() {
  return typeof Notification !== "undefined" && window.isSecureContext;
}

function renderNotifyControl() {
  if (!notificationsAvailable()) return;
  elements.notifySection.hidden = false;
  const denied = Notification.permission === "denied";
  elements.notifyEnabled.disabled = denied;
  elements.notifyEnabled.checked = !denied && state.notifyStageAlerts;
  elements.notifyDetail.textContent = denied
    ? "Blocked for this site. Re-enable notifications in your browser's site settings."
    : "A browser notification when a long stage finishes while this tab is in the background.";
}

async function requestStageAlerts() {
  // Asked from inside the submit gesture rather than at page load: a prompt
  // that arrives before the visitor has started anything is the one people
  // dismiss permanently, and a denial here cannot be taken back from script.
  if (!notificationsAvailable()) return;
  if (!state.notifyStageAlerts) return;
  if (Notification.permission !== "default") return;
  try {
    await Notification.requestPermission();
  } catch {
    // Older Safari rejects rather than resolving "denied".
  }
  renderNotifyControl();
}

function setAttention(pending) {
  state.pendingAttention = pending;
  renderSessionTitle();
}

function researcherIsWatching() {
  return document.visibilityState === "visible" && document.hasFocus();
}

function raiseStageAlert(workflow, headline) {
  setAttention(true);
  if (!state.notifyStageAlerts) return;
  if (!notificationsAvailable() || Notification.permission !== "granted")
    return;
  const notification = new Notification(headline, {
    body: `${stageLabel(workflow.stage)} · ${sessionTitle(workflow)}`,
    // One live notification per run: a poll that fires twice must replace its
    // own banner rather than stack a second one behind it.
    tag: `coscientist:${workflow.id}`,
    icon: "/assets/favicon.svg",
    requireInteraction: false,
  });
  notification.addEventListener("click", () => {
    window.focus();
    notification.close();
  });
}

function trackStageAlerts(workflow, operationActive) {
  if (operationActive) {
    if (state.operationStartedAt === null)
      state.operationStartedAt = Date.now();
    return;
  }
  const elapsed =
    state.operationStartedAt === null
      ? 0
      : Date.now() - state.operationStartedAt;
  state.operationStartedAt = null;

  const finished = workflow.status === "ready_for_report";
  const waiting =
    workflow.requires_human_approval ||
    ["input_required", "evidence_required", "governance_blocked"].includes(
      workflow.status,
    );
  if (!finished && !waiting) return;
  // The dossier is the payoff and is always worth an alert; an intermediate
  // gate is only worth one if the researcher had time to walk away from it.
  if (!finished && elapsed < LONG_STAGE_MILLISECONDS) return;
  if (researcherIsWatching()) return;

  const gateKey = `${workflow.id}:${workflow.stage}:${workflow.status}:${
    workflow.pending_draft?.id || ""
  }`;
  if (state.notifiedGateKey === gateKey) return;
  state.notifiedGateKey = gateKey;
  raiseStageAlert(
    workflow,
    finished
      ? "Your research dossier is ready"
      : `Co-Scientist needs you · ${workflowStatusCopy(workflow)}`,
  );
}

function fillChoices(select, items, describe) {
  select.replaceChildren(
    ...items.map((item) => {
      const described = describe(item);
      const option = document.createElement("option");
      option.value = described.value;
      option.textContent = described.text;
      if (described.title) option.title = described.title;
      option.selected = described.selected;
      return option;
    }),
  );
}

function runConfigCopy(workflow) {
  // Appended to the gate line rather than given a row of its own: on a resumed
  // session these two are the only way to tell which configuration produced
  // what is on screen, and they never change once the run has started.
  const parts = [];
  if (workflow.model) parts.push(escapeHtml(workflow.model));
  const language = (
    elements.languageChoice.querySelector(
      `option[value="${workflow.language}"]`,
    ) || {}
  ).textContent;
  if (language && workflow.language !== "en") parts.push(escapeHtml(language));
  return parts.length ? ` · ${parts.join(" · ")}` : "";
}

function workflowStatusCopy(workflow) {
  if (workflow.status === "input_required") return "Scientific input required";
  if (workflow.status === "evidence_required")
    return "Verified evidence required";
  if (workflow.status === "governance_blocked")
    return "Governance review blocked";
  if (workflow.status === "stopped_by_researcher")
    return "Stopped by researcher";
  if (workflow.status === "ready_for_report") return "Dossier ready";
  if (workflow.requires_human_approval) return "Human decision required";
  return "Workflow active";
}

function unresolvedRequirements(workflow) {
  return workflow.input_requirements.filter(
    (item) =>
      item.blocking && !["provided", "fallback_accepted"].includes(item.status),
  );
}

function evidenceProgressCopy(workflow) {
  const presentation = workflow.evidence_progress;
  if (!presentation) return "";
  const metric = Object.fromEntries(
    (presentation.metrics || []).map((item) => [item.label, item.value]),
  );
  const verified = metric["Verified"];
  const trust =
    verified === undefined
      ? "discovered, not yet verified"
      : `${verified} verified · ${metric["Registry-confirmed"] || 0} registry-confirmed · ${metric["Quarantined"] || 0} quarantined`;
  return `Deep Research pass ${metric["Deep Research passes"] || 1} of 8 · ${
    metric.Coverage || 0
  }% coverage · ${metric["Source leads"] || 0} source leads · ${trust}`;
}

// What each gate is deciding about, and what pressing the button spends. A gate
// whose primary button said "Accept & continue" told the researcher neither what
// they were accepting nor that accepting the scope draft starts a billed
// Deep Research wave that cannot be cancelled once it is running.
const GATE_WORK = {
  scope: {
    draft: "research plan",
    next: "Evidence",
    starts:
      "up to 8 Deep Research passes against the live literature. Billed, tens of minutes, and a pass cannot be cancelled once it starts.",
    revises: "rewrites the plan. About a minute.",
  },
  evidence: {
    draft: "evidence base",
    next: "Generate",
    starts: "four generator strategies write hypotheses. A few minutes.",
    // Up to six web searches aimed at what you write in the box and at the
    // gaps the coverage audit named -- not a second Deep Research wave. Saying
    // "it may spend another pass" was true of the stage that re-ran discovery
    // from nothing, and it made the one button that improves the evidence base
    // read like the expensive one.
    revises:
      "searches the gaps you name, and keeps everything already found. About a minute, and no Deep Research pass.",
  },
  generate: {
    draft: "candidate hypotheses",
    next: "Reflect",
    starts: "seven reviewers read every candidate. A few minutes.",
    revises: "writes a fresh set of candidates. A few minutes.",
  },
  reflect: {
    draft: "review panel",
    next: "Rank",
    starts: "a Swiss tournament and a debated final round. A few minutes.",
    revises: "re-runs the review panel. A few minutes.",
  },
  rank: {
    draft: "ranking",
    next: "Evolve",
    starts:
      "the shortlist is evolved and independently re-reviewed. A few minutes.",
    revises: "re-runs the tournament. A few minutes.",
  },
  evolve: {
    draft: "evolved shortlist",
    next: "Proximity",
    starts: "the surviving hypotheses are clustered. Under a minute.",
    revises: "evolves the shortlist again. A few minutes.",
  },
  proximity: {
    draft: "proximity map",
    next: "Meta-review",
    starts: "the meta-review synthesises the whole run. A few minutes.",
    revises: "rebuilds the proximity map. Under a minute.",
  },
  meta_review: {
    draft: "meta-review",
    next: "Report",
    starts: "the dossier is assembled. Under a minute.",
    revises: "rewrites the meta-review. A few minutes.",
  },
  report: {
    draft: "dossier",
    next: "",
    starts: "",
    revises: "rewrites the dossier. A few minutes.",
  },
};

function gateWork(stage) {
  return (
    GATE_WORK[stage] || {
      draft: `${stageLabel(stage)} draft`,
      next: "",
      starts: "",
      revises: "re-runs this stage.",
    }
  );
}

function plural(count, noun) {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

function gateBlockReasons(workflow, requirements, openFindings) {
  // A disabled primary with no explanation reads as a broken button. Every
  // reason it is disabled is nameable, and each one is answered somewhere on
  // this same card.
  const reasons = [];
  if (openFindings.length)
    reasons.push(`${plural(openFindings.length, "safety finding")} unanswered`);
  if (requirements.length)
    reasons.push(`${plural(requirements.length, "input reference")} still needed`);
  if (workflow.pending_artifacts.length)
    reasons.push(
      `${plural(workflow.pending_artifacts.length, "specialist output")} still to approve`,
    );
  return reasons;
}

function approvalCardDraft(card) {
  // Everything the researcher typed into this card and has not yet posted.
  // Answering one governance finding rebuilds the card, and until this existed
  // the rebuild wiped the half-written reasons in all the others.
  if (!card) return null;
  const active = document.activeElement;
  const findings = {};
  card.querySelectorAll(".governance-finding").forEach((node) => {
    const name = node.querySelector(".governance-adjudicator");
    const reason = node.querySelector(".governance-justification");
    findings[node.dataset.reviewId] = {
      adjudicator: name ? name.value : "",
      justification: reason ? reason.value : "",
      focused:
        active === name
          ? "adjudicator"
          : active === reason
            ? "justification"
            : "",
      caret: active === name || active === reason ? active.selectionStart : 0,
    };
  });
  const editor = card.querySelector(".direct-editor");
  const revision = card.querySelector(".revision-field");
  return {
    findings,
    revision: revision ? revision.value : "",
    revisionFocused: active === revision,
    editorOpen: editor ? !editor.hidden : false,
    editorText: card.querySelector(".direct-edit-field")?.value || "",
  };
}

function restoreApprovalCardDraft(card, draft) {
  if (!draft) return;
  const revision = card.querySelector(".revision-field");
  if (revision && draft.revision) revision.value = draft.revision;
  const editor = card.querySelector(".direct-editor");
  if (editor && draft.editorOpen) {
    editor.hidden = false;
    const field = card.querySelector(".direct-edit-field");
    if (field && draft.editorText) field.value = draft.editorText;
  }
  card.querySelectorAll(".governance-finding").forEach((node) => {
    const carried = draft.findings[node.dataset.reviewId];
    if (!carried) return;
    const name = node.querySelector(".governance-adjudicator");
    const reason = node.querySelector(".governance-justification");
    if (name && carried.adjudicator) name.value = carried.adjudicator;
    if (reason && carried.justification) reason.value = carried.justification;
    const target =
      carried.focused === "adjudicator"
        ? name
        : carried.focused === "justification"
          ? reason
          : null;
    if (!target) return;
    target.focus();
    const caret = Math.min(carried.caret, target.value.length);
    target.setSelectionRange(caret, caret);
  });
  if (draft.revisionFocused && revision) revision.focus();
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
  const findings = workflow.governance_blockers || [];
  const openFindings = findings.filter((item) => !item.resolution);
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

  // The same card, re-rendered in place, for as long as the governance block
  // lasts. Each answer used to tear it down and append a new card holding the
  // findings that were left, which read as the gate restarting and lost every
  // reason that was part-typed in the others.
  const stageKey = `${workflow.id}:${workflow.stage}`;
  const reused =
    existing &&
    existing.dataset.stageKey === stageKey &&
    existing.dataset.governanceGate === "true" &&
    findings.length > 0;
  const carried = reused ? approvalCardDraft(existing) : null;
  if (existing && !reused) existing.remove();

  const card = reused ? existing : document.createElement("section");
  card.className = "approval-card";
  card.dataset.workflowId = workflow.id;
  card.dataset.gateKey = gateKey;
  card.dataset.stageKey = stageKey;
  card.dataset.governanceGate = findings.length ? "true" : "false";
  const work = gateWork(workflow.stage);
  const blockReasons = gateBlockReasons(workflow, requirements, openFindings);
  card.innerHTML = `
    <div class="approval-card-head">
      <div>
        <p class="eyebrow">Supervisor gate · ${escapeHtml(workflow.approval_profile)}${runConfigCopy(workflow)}</p>
        <h3>${escapeHtml(stageLabel(workflow.stage))} · ${
          operationActive
            ? "Specialists working"
            : operationFailed
              ? "Stage preparation interrupted"
              : escapeHtml(workflowStatusCopy(workflow))
        }</h3>
        <p class="approval-stage-progress">Stage ${workflow.stage_number} of ${workflow.stage_count} · ${workflow.task_summary.completed}/${workflow.task_summary.total} specialist tasks complete</p>
      </div>
      <span class="approval-badge">${
        operationActive
          ? "In progress"
          : decisionRequired
            ? "Your decision"
            : "Integrity gate"
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
    ${
      findings.length
        ? `<div class="input-requirement governance-warning">
            <strong>Safety and governance review recorded ${plural(findings.length, "fatal flaw")}</strong>
            <p>${findings.length - openFindings.length} of ${findings.length} answered. Nothing advances until every finding below is answered, one at a time and each in place. Withdrawing drops that hypothesis and rebuilds the population without it; overriding keeps it and accepts the flaw. Your name and your reason are recorded against that finding alone and reprinted in the dossier beside the flaw.</p>
          </div>`
        : ""
    }
    <div class="governance-list"></div>
    <div class="artifact-list"></div>
    ${
      operationActive || operationFailed
        ? ""
        : `
          <div class="direct-editor" hidden>
            <label>Directly edit the ${escapeHtml(work.draft)}</label>
            <textarea class="direct-edit-field" rows="14"></textarea>
            <p>Your saved edit becomes a new auditable artifact version.</p>
            <button class="save-edit" type="button" data-decision="edit">Save the edited ${escapeHtml(work.draft)}</button>
          </div>
          <textarea class="revision-field" rows="2" placeholder="Or describe what the agent should change in the ${escapeHtml(work.draft)}…"></textarea>
          ${
            blockReasons.length
              ? `<p class="gate-blocked">Accepting is blocked: ${escapeHtml(blockReasons.join(" · "))}. Each is answered on this card.</p>`
              : work.starts
                ? `<p class="gate-consequence">Accepting the ${escapeHtml(work.draft)} starts <strong>${escapeHtml(work.next)}</strong> — ${escapeHtml(work.starts)}</p>`
                : ""
          }
          <div class="approval-actions">
            <button class="primary" type="button" data-decision="accept" ${
              blockReasons.length ? "disabled" : ""
            }>Accept the ${escapeHtml(work.draft)}${work.next ? ` &amp; run ${escapeHtml(work.next)}` : ""}</button>
            <button type="button" data-decision="toggle_edit">Edit the ${escapeHtml(work.draft)} myself</button>
            <button type="button" data-decision="revise" title="${escapeHtml(work.revises)}">Send the ${escapeHtml(work.draft)} back for revision</button>
            <button class="danger" type="button" data-decision="stop">Stop this session</button>
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

  const governanceList = card.querySelector(".governance-list");
  const adjudicator = rememberedAdjudicator();
  findings.forEach((item) => {
    const block = document.createElement("div");
    block.dataset.reviewId = item.review_id;
    const flaws = `
      <p class="governance-reviewer">${escapeHtml(item.reviewer.replaceAll("_", " "))} · ${escapeHtml(item.candidate_id)}</p>
      <ul class="governance-flaws">
        ${item.fatal_flaws.map((flaw) => `<li>${escapeHtml(flaw)}</li>`).join("")}
      </ul>
      ${
        item.objections.length
          ? `<ul class="governance-objections">${item.objections
              .map((objection) => `<li>${escapeHtml(objection)}</li>`)
              .join("")}</ul>`
          : ""
      }`;
    if (item.resolution) {
      // Answered, and still on the card: collapsed to its verdict so the
      // remaining work is what stands out, but never removed under the reader.
      const withdrawn = item.resolution.action === "withdraw";
      block.className = "governance-finding settled";
      block.innerHTML = `
        <details>
          <summary>
            <span class="governance-verdict ${withdrawn ? "withdrawn" : "overridden"}">${withdrawn ? "Withdrawn" : "Override recorded"}</span>
            <span class="governance-settled-title">${escapeHtml(item.candidate_title)}</span>
          </summary>
          ${flaws}
          <p class="governance-resolution"><strong>${escapeHtml(item.resolution.actor)}</strong>: ${escapeHtml(item.resolution.reason)}</p>
        </details>
      `;
      governanceList.append(block);
      return;
    }
    const others = openFindings.length - 1;
    block.className = "governance-finding";
    block.innerHTML = `
      <strong>${escapeHtml(item.candidate_title)}</strong>
      ${flaws}
      <label class="governance-label">Adjudicator
        <input class="governance-adjudicator" type="text" placeholder="Your name" value="${escapeHtml(adjudicator)}" />
      </label>
      <label class="governance-label">Reason, recorded verbatim
        <textarea class="governance-justification" rows="2" placeholder="Why this hypothesis is withdrawn, or why the flaw is acceptable…"></textarea>
      </label>
      <div class="input-actions">
        <button type="button" data-decision="withdraw_hypothesis">Withdraw this hypothesis</button>
        <button class="danger" type="button" data-decision="override_governance">Override this finding</button>
        ${
          others > 0
            ? `<button class="ghost" type="button" data-decision="copy_reason">Copy this reason into the other ${others} — each still posts its own</button>`
            : ""
        }
      </div>
    `;
    governanceList.append(block);
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
  restoreApprovalCardDraft(card, carried);
  if (!reused) {
    card.addEventListener("click", handleDecisionClick);
    elements.messages.append(card);
  }
}

function stopWorkflowPolling() {
  if (state.pollTimer) window.clearTimeout(state.pollTimer);
  state.pollTimer = null;
}

function pollWorkflow(wait = 1400) {
  stopWorkflowPolling();
  state.pollTimer = window.setTimeout(async () => {
    if (!state.workflowId) return;
    try {
      const workflow = await researchApi(
        `/sessions/${encodeURIComponent(state.workflowId)}`,
      );
      renderWorkflow(workflow);
    } catch {
      // Only renderWorkflow re-arms this timer, so a single dropped poll used
      // to end progress reporting for the whole run: the card sat on
      // "Specialists working" until the page was reloaded. A stage against the
      // deployed service runs for minutes at roughly one poll a second, which
      // makes an occasional failure ordinary rather than exceptional. Back off
      // so a service that is genuinely down is not hammered, and say
      // "reconnecting" rather than reporting a dead run.
      setConnection("error", "Reconnecting to workflow progress");
      pollWorkflow(Math.min(wait * 2, 15000));
    }
  }, wait);
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
    preview.producer ||
      (preview.stage === "report"
        ? "meta_reviewer"
        : "co_scientist_supervisor"),
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
      if (!state.workflow?.report)
        throw new Error("The dossier is not available yet.");
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
    toast(
      error instanceof Error
        ? error.message
        : "Stage output could not be loaded",
    );
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
  // Ahead of the early return for stage previews: a researcher reading an
  // earlier stage while the next one runs still wants to be called back.
  trackStageAlerts(workflow, operationActive);

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
      workflow.status.includes("blocked") ||
        workflow.status === "input_required"
        ? "error"
        : "ready",
      operationActive
        ? "Specialists are preparing the next gate"
        : workflowStatusCopy(workflow),
    );
    if (operationActive) pollWorkflow();
    else stopWorkflowPolling();
  }
}

async function createGuidedWorkflow(prompt, pending) {
  // The option lists arrive over the network. On a local server that lands
  // before anyone can type; against the deployed service it does not, and a
  // prompt submitted first posted an empty model and an empty language.
  await state.optionsReady;
  if (!elements.modelChoice.value || !elements.languageChoice.value) {
    throw new Error("Model and language options are still loading");
  }
  const workflow = await researchApi("/sessions", {
    method: "POST",
    body: JSON.stringify({
      question: prompt,
      approval_profile: elements.approvalProfile.value,
      model: elements.modelChoice.value,
      language: elements.languageChoice.value,
      // Read off the box rather than sent unconditionally: an auto run is
      // disabled above and must not ask for a gate nothing will stop at.
      evidence_review:
        elements.evidenceReview.checked && !elements.evidenceReview.disabled,
    }),
  });
  renderWorkflow(workflow, pending, true);
}

function selectMode(mode) {
  state.mode = mode;
  localStorage.setItem("coscientist.mode", state.mode);
  document
    .querySelectorAll("[data-mode]")
    .forEach((item) =>
      item.classList.toggle("active", item.dataset.mode === mode),
    );
  elements.approvalProfile.hidden = mode === "conversation";
  [
    elements.approvalProfile,
    elements.modelChoice,
    elements.languageChoice,
    elements.evidenceReview,
  ].forEach((control) => {
    control.hidden = mode === "conversation";
    control.closest(".profile-control").hidden = mode === "conversation";
  });
  syncEvidenceReviewControl();
}

function syncEvidenceReviewControl() {
  // An auto run accepts every draft it produces, so a checked box there would
  // promise a stop that never happens. The server settles it the same way; the
  // box says so before the run starts rather than after it has finished.
  const auto = elements.approvalProfile.value === "auto";
  elements.evidenceReview.disabled = auto;
  const control = elements.evidenceReview.closest(".profile-control");
  control.classList.toggle("disabled", auto);
  control.title = auto
    ? ""
    : "Stops after discovery and shows the corpus, its coverage and its gaps, before four generators reason over it.";
  elements.evidenceReviewNote.hidden =
    !auto || state.mode === "conversation" || control.hidden;
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
    const workflow = await researchApi(
      `/sessions/${encodeURIComponent(sessionId)}`,
    );
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
    if (!restore)
      toast(error instanceof Error ? error.message : "Session unavailable");
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
  if (action === "copy_reason") {
    // Four findings on one run were four separate objections to the same
    // hazard, and each had to be retyped in full. Copied rather than applied:
    // every finding still posts its own reason, and every copy can be edited
    // before it is posted.
    const source = button.closest(".governance-finding");
    const name = source.querySelector(".governance-adjudicator").value.trim();
    const reason = source.querySelector(".governance-justification").value.trim();
    if (!reason) {
      toast("Write the reason here first, then copy it");
      source.querySelector(".governance-justification").focus();
      return;
    }
    let copied = 0;
    card
      .querySelectorAll(".governance-finding:not(.settled)")
      .forEach((finding) => {
        if (finding === source) return;
        finding.querySelector(".governance-justification").value = reason;
        if (name) finding.querySelector(".governance-adjudicator").value = name;
        copied += 1;
      });
    toast(`Copied into ${copied} finding${copied === 1 ? "" : "s"} — edit before posting`);
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
  if (action === "withdraw_hypothesis" || action === "override_governance") {
    const finding = button.closest(".governance-finding");
    const name = finding.querySelector(".governance-adjudicator").value.trim();
    const reason = finding
      .querySelector(".governance-justification")
      .value.trim();
    if (!name) {
      toast("Name the person answering this finding");
      finding.querySelector(".governance-adjudicator").focus();
      return;
    }
    if (!reason) {
      toast("A written reason is recorded with every governance decision");
      finding.querySelector(".governance-justification").focus();
      return;
    }
    if (
      action === "override_governance" &&
      !window.confirm(
        "Overriding keeps a hypothesis the reviewer called fatally flawed. " +
          "It stays in the dossier, and so does this decision, under your name.",
      )
    )
      return;
    payload.review_id = finding.dataset.reviewId;
    payload.actor = name;
    payload.feedback = reason;
    localStorage.setItem(ADJUDICATOR_KEY, name);
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

  const inlineResolution =
    action === "withdraw_hypothesis" || action === "override_governance";
  state.busy = true;
  card.querySelectorAll("button").forEach((item) => (item.disabled = true));
  setConnection(
    "",
    action === "accept" ? "Advancing to the next gate" : "Recording decision",
  );
  try {
    const workflow = await researchApi(
      `/sessions/${encodeURIComponent(state.workflowId)}/decisions`,
      { method: "POST", body: JSON.stringify(payload) },
    );
    if (!inlineResolution) {
      // A governance answer settles one finding inside a card that is still
      // being worked; retiring the card here is what spawned a second one.
      card.classList.add("resolved");
      card.querySelector(".approval-badge").textContent = "Recorded";
    }
    renderWorkflow(workflow);
  } catch (error) {
    toast(
      error instanceof Error ? error.message : "Decision could not be recorded",
    );
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
  requestStageAlerts();
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
    showError(
      pending,
      error instanceof Error ? error.message : "Unexpected error",
    );
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
  clearWorkflowDisplay();
  updateSessionIdentity(null);
  updateStageNavigation(null);
  renderSessionHistory();
  setAttention(false);
  state.notifiedGateKey = null;
  state.operationStartedAt = null;
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

elements.approvalProfile.addEventListener("change", syncEvidenceReviewControl);
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
  if (event.target.closest(".session-rename")) {
    // Renaming from the list retitles that session wherever it appears, but
    // the editor lives in the topbar, so the drawer gets out of the way.
    document.body.classList.remove("history-open");
    beginRename(item.dataset.sessionId);
    return;
  }
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
elements.renameSession.addEventListener("click", () => beginRename());
elements.sessionTitleInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") endRename(true);
  if (event.key === "Escape") endRename(false);
});
elements.sessionTitleInput.addEventListener("blur", () => endRename(true));
elements.historyButton.addEventListener("click", () => {
  document.body.classList.add("history-open");
});
elements.closeHistory.addEventListener("click", () => {
  document.body.classList.remove("history-open");
});
elements.notifyEnabled.addEventListener("change", () => {
  state.notifyStageAlerts = elements.notifyEnabled.checked;
  localStorage.setItem(NOTIFY_KEY, state.notifyStageAlerts ? "on" : "off");
  if (state.notifyStageAlerts) requestStageAlerts();
  renderNotifyControl();
});
// Coming back to the tab is the acknowledgement. Clearing the marker on
// return also re-arms the alert, so the next gate is announced again.
window.addEventListener("focus", () => setAttention(false));
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") setAttention(false);
});
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

state.optionsReady = loadResearchOptions();
renderSessionTitle(null);
renderNotifyControl();
selectMode(state.mode);
renderSessionHistory();
updateStageNavigation(null);
if (state.mode === "conversation") {
  createSession().catch(() =>
    setConnection("error", "Could not create session"),
  );
} else {
  const currentWorkflowId = localStorage.getItem(CURRENT_WORKFLOW_KEY);
  if (currentWorkflowId) {
    openResearchSession(currentWorkflowId, { restore: true });
  } else {
    setConnection("ready", "Ready for a governed inquiry");
  }
}
