# Evidence Discovery V2

Evidence Discovery is a mandatory, visible stage between Scope and Generate for
new V2 sessions. It builds a knowledge landscape; it does not verify scientific
claims.

## Runtime contract

1. Start one stored Standard Deep Research interaction with
   `deep-research-preview-04-2026`.
2. Persist the interaction ID immediately. Polling updates the same
   `DiscoveryManifest`, allowing a replacement worker to resume that interaction.
3. Store the complete terminal payload in GCS and normalize only statements and
   citations present in that payload.
4. Audit coverage across research directions and supporting, contradictory,
   negative/null, replication, methods, safety/governance, and
   correction/retraction facets.
5. Run a focused second pass when material gaps remain. A third pass is allowed
   only if the second improves weighted coverage by at least five percentage
   points and adds an authoritative source or closes a material gap.
6. When another Deep Research pass is not justified, queue at most six targeted
   Google Search enrichment requests.
7. Hand all discovered leads to Source Verification. Generation stays blocked
   until original-source locations are verified or a researcher explicitly
   accepts the limited exploratory fallback.

Deep Research uses stored Gemini Interactions. Raw results stay immutable,
cross-pass conflicts remain separate, and every discovery record remains
`discovered_unverified`. No session query or unverified report is eligible for
the shared verified Agent Search corpus.

## Deterministic acceptance matrix

| Control | Required result |
| --- | --- |
| First pass | Exactly one interaction is started for every V2 session. |
| Sequencing | A later pass cannot start before normalization and coverage audit. |
| Pass limit | No more than three interactions per session. |
| Sufficiency | A sufficient first pass stops immediately. |
| Incremental value | Improvement below five points stops Deep Research. |
| Idempotency | A restarted worker resumes the persisted interaction ID. |
| Provenance | Every accepted normalized URL occurs in its originating report. |
| Citation fabrication | Zero invented citations accepted from normalization. |
| Deduplication | Canonical duplicates merge while retaining every pass link. |
| Contradiction | Opposing statements remain distinct records. |
| Verification boundary | Discovery never sets `verified` or `corrected`. |
| Generation gate | Unverified evidence blocks Generate unless the researcher explicitly selects exploratory mode. |
| Hidden reasoning | Thinking summaries are disabled and not stored. |
| Search boundary | Broad discovery uses Deep Research; Google Search is restricted to at most six residual-gap queries. |
| Network safety | Local, private, reserved, link-local, and metadata URLs are rejected. |
| Model Armor | No Model Armor dependency, call, middleware, or deployment component exists. |

## Quality evaluation

The benchmark must contain at least twelve cases spanning all six research
modes. Promotion targets are:

- direction coverage ≥ 0.90;
- Precision@10 ≥ 0.85 and Recall@20 ≥ 0.80;
- primary/authoritative rate in the top ten ≥ 0.75;
- counterevidence and negative-evidence coverage ≥ 0.90;
- DOI/dataset resolution ≥ 0.95 when identifiers exist;
- designated correction/retraction detection = 1.00;
- groundedness and evidence calibration ≥ 0.95;
- fabricated sources/findings = 0;
- ≥ 80% of additional passes add an authoritative source or close a material
  gap;
- targeted enrichment precision ≥ 0.85.

Use the agents-cli quality flywheel for at least five recorded iterations:
generate traces, grade them, analyze failure clusters, make one bounded change,
and compare against the preceding results. Do not enable Pass 3 in production
until the incremental-pass metric demonstrates measurable value.

## Rollout

`EVIDENCE_PIPELINE_VERSION=2` applies only to newly created sessions. Legacy
sessions retain their original stage sequence. Start with repeat passes disabled
in production shadow evaluation, compare the saved V1 and V2 evidence manifests,
then enable Pass 2 and finally Pass 3 independently. The Gemini API key belongs
in Secret Manager and raw Deep Research payloads belong in the dedicated
evidence GCS bucket.

### Cloud Run worker

The same container exposes a separate private entry point:

```text
uv run uvicorn app.evidence_worker:app --host 0.0.0.0 --port 8080
```

Deploy that command as `coscientist-evidence-worker` in `us-east1`, with no
`allUsers` invoker binding. Create a regional Cloud Tasks queue named
`coscientist-evidence`, grant its OIDC service account `roles/run.invoker` only
on the worker, and grant the public application service account permission to
enqueue tasks only on that queue. Configure:

```text
EVIDENCE_WORKER_URL=https://<private-worker-url>
EVIDENCE_CLOUD_TASKS_LOCATION=us-east1
EVIDENCE_CLOUD_TASKS_QUEUE=coscientist-evidence
EVIDENCE_TASKS_SERVICE_ACCOUNT=<task-caller-service-account>
```

Each worker request starts or polls exactly one stored interaction and returns
within the Cloud Tasks deadline. The ledger persists the interaction ID and poll
count before another task is scheduled. Polls begin at 15 seconds and back off
to 60 seconds. Deterministic task names include the session version, so a
delivery retry cannot create another interaction or manifest.
