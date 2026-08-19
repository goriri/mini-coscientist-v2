#!/usr/bin/env bash
#
# Move the evidence stage off the request-handling instance and onto Cloud Tasks.
#
# Today app/evidence_tasks.py:configured() returns false, because neither
# EVIDENCE_WORKER_URL nor EVIDENCE_CLOUD_TASKS_QUEUE is set. The stage therefore
# runs inline in a FastAPI BackgroundTask: the HTTP request returns in well under
# the 300s timeout, and thirty to forty-five minutes of Deep Research polling
# carries on afterwards on an instance Cloud Run may reclaim at any moment. That
# is what puts "Recovering interrupted work" on the page mid-stage.
#
# app/evidence_worker.py is a separate FastAPI app rather than a route on the
# main one, so this deploys it as its own service -- internal ingress, no
# unauthenticated access -- and has Cloud Tasks call it with an OIDC token.
#
# Every step checks for what it is about to create, so the script can be re-run
# after fixing a permission and will only do what is still outstanding.
#
# Usage:
#   scripts/provision_evidence_worker.sh                  # provision the worker
#   scripts/provision_evidence_worker.sh --dry-run        # print, run nothing
#   scripts/provision_evidence_worker.sh --check          # report state, change nothing
#
# The three settings that cost money continuously, or that rotate a live
# credential, are opt-in and never run unless named:
#   --sql-tier db-custom-2-7680     # patches Cloud SQL; RESTARTS the instance
#   --min-instances 1               # keeps one warm instance billed around the clock
#   --admin-token <value>           # replaces COSCIENTIST_ADMIN_TOKEN, currently "admin"

set -euo pipefail

PROJECT="${PROJECT:-cellular-cider-495602-r9}"
REGION="${REGION:-us-east1}"
SERVICE="${SERVICE:-coscientist}"
WORKER_SERVICE="${WORKER_SERVICE:-coscientist-evidence-worker}"
QUEUE="${QUEUE:-coscientist-evidence}"
SQL_INSTANCE="${SQL_INSTANCE:-coscientist-db}"

DRY_RUN=false
CHECK_ONLY=false
SQL_TIER=""
MIN_INSTANCES=""
ADMIN_TOKEN=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true; shift ;;
    --check) CHECK_ONLY=true; shift ;;
    --sql-tier) SQL_TIER="$2"; shift 2 ;;
    --min-instances) MIN_INSTANCES="$2"; shift 2 ;;
    --admin-token) ADMIN_TOKEN="$2"; shift 2 ;;
    -h|--help) sed -n '2,32p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
note() { printf '   %s\n' "$*"; }
fail() { printf '\n\033[31mFAILED: %s\033[0m\n' "$*" >&2; exit 1; }

# Printed before it is run, so a --dry-run transcript is the thing that would
# have happened rather than a description of it.
run() {
  printf '   $ %s\n' "$*"
  if [[ "$DRY_RUN" == true || "$CHECK_ONLY" == true ]]; then
    return 0
  fi
  "$@"
}

command -v gcloud >/dev/null || fail "gcloud is not on PATH."

ACCOUNT="$(gcloud config get-value account 2>/dev/null)"
say "Account and project"
note "account: ${ACCOUNT}"
note "project: ${PROJECT}   region: ${REGION}"
# The compute service account the deploys run under cannot enable a service,
# create a queue, or set a project IAM policy. Named here rather than left to
# surface as three separate permission errors halfway through.
if [[ "$ACCOUNT" == *"-compute@developer.gserviceaccount.com" ]]; then
  note ""
  note "This is the compute service account. It lacks serviceusage.services.enable,"
  note "cloudtasks.queues.create and resourcemanager.projects.setIamPolicy, so steps"
  note "1, 3 and 4 below will be refused. Re-run as an owner, or grant those first."
fi

PROJECT_NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)' 2>/dev/null || true)"
[[ -n "$PROJECT_NUMBER" ]] || fail "Could not read the project number for ${PROJECT}."

# -----------------------------------------------------------------------------
# What the main service is already running. Read rather than pinned: the worker
# has to be the same build as the service whose database it writes to, and a
# digest hardcoded here would be a version behind on the next deploy.
# -----------------------------------------------------------------------------
say "Reading the running service"
MAIN_JSON="$(gcloud run services describe "$SERVICE" --project "$PROJECT" \
  --region "$REGION" --format=json 2>/dev/null)" \
  || fail "Could not describe the ${SERVICE} service."

# Tab-separated and read with IFS set to tab alone, so a field the service does
# not carry stays empty instead of collapsing and shifting every field after it
# one place left -- which would silently deploy the worker under the wrong
# identity rather than fail.
IFS=$'\t' read -r IMAGE RUNTIME_SA SQL_CONNECTION SECRET_NAME SECRET_VERSION BUCKET <<EOF
$(printf '%s' "$MAIN_JSON" | python3 -c '
import json, sys
d = json.load(sys.stdin)
template = d["spec"]["template"]
container = template["spec"]["containers"][0]
env = {item["name"]: item for item in container.get("env", [])}
secret = env.get("DATABASE_PASSWORD", {}).get("valueFrom", {}).get("secretKeyRef", {})
print("\t".join([
    container["image"],
    template["spec"].get("serviceAccountName", ""),
    template["metadata"]["annotations"].get("run.googleapis.com/cloudsql-instances", ""),
    secret.get("name", ""),
    secret.get("key", ""),
    env.get("LOGS_BUCKET_NAME", {}).get("value", ""),
]))
')
EOF

for required in RUNTIME_SA SQL_CONNECTION SECRET_NAME SECRET_VERSION BUCKET; do
  [[ -n "${!required}" ]] || fail "${SERVICE} carries no ${required}; the worker would not match it."
done

[[ -n "$IMAGE" ]] || fail "Could not read the deployed image from ${SERVICE}."
note "image:    ${IMAGE}"
note "identity: ${RUNTIME_SA}"
note "cloudsql: ${SQL_CONNECTION}"
note "secret:   ${SECRET_NAME}:${SECRET_VERSION}"
note "bucket:   ${BUCKET}"

TASKS_AGENT="service-${PROJECT_NUMBER}@gcp-sa-cloudtasks.iam.gserviceaccount.com"

# -----------------------------------------------------------------------------
say "1. Cloud Tasks API"
# -----------------------------------------------------------------------------
if gcloud services list --enabled --project "$PROJECT" \
     --format='value(config.name)' 2>/dev/null | grep -qx cloudtasks.googleapis.com; then
  note "Already enabled."
else
  note "Not enabled. Needs roles/serviceusage.serviceUsageAdmin."
  run gcloud services enable cloudtasks.googleapis.com --project "$PROJECT"
  # The queue's OIDC token is minted by a service agent that does not exist
  # until something asks for it, and step 4 grants a role on an account that
  # has to be there to receive it.
  run gcloud beta services identity create \
    --service=cloudtasks.googleapis.com --project "$PROJECT"
fi

# -----------------------------------------------------------------------------
say "2. The worker service"
# -----------------------------------------------------------------------------
# Same image, different entrypoint: the Dockerfile's CMD starts
# app.fast_api_app:app, and this one has to start app.evidence_worker:app.
WORKER_ENV="GOOGLE_CLOUD_PROJECT=${PROJECT}"
WORKER_ENV+=",GOOGLE_CLOUD_LOCATION=global"
WORKER_ENV+=",GOOGLE_GENAI_USE_VERTEXAI=TRUE"
WORKER_ENV+=",CLOUD_SQL_CONNECTION_NAME=${SQL_CONNECTION}"
WORKER_ENV+=",DATABASE_NAME=coscientist"
WORKER_ENV+=",DATABASE_USER=coscientist_app"
WORKER_ENV+=",SESSION_DATABASE_NAME=coscientist_adk"
WORKER_ENV+=",LOGS_BUCKET_NAME=${BUCKET}"

if gcloud run services describe "$WORKER_SERVICE" --project "$PROJECT" \
     --region "$REGION" --format='value(status.url)' >/dev/null 2>&1; then
  note "Exists; redeploying it onto the image the main service is running."
fi
# The request timeout below is half an hour, matching the dispatch deadline in
# EVIDENCE_TASK_DEADLINE_SECONDS. A task is one stage, and folding a finished
# Deep Research wave in -- a model call per pass, then a fetch of every source
# those passes named -- took six and a half minutes on a live run. At the five
# minutes this used to allow, Cloud Run killed the request mid-read and the
# retry landed on a lease the killed instance still held.
run gcloud run deploy "$WORKER_SERVICE" \
  --project "$PROJECT" --region "$REGION" --image "$IMAGE" \
  --command uv \
  --args "run,uvicorn,app.evidence_worker:app,--host,0.0.0.0,--port,8080" \
  --ingress internal --no-allow-unauthenticated \
  --service-account "$RUNTIME_SA" \
  --add-cloudsql-instances "$SQL_CONNECTION" \
  --cpu 1 --memory 4Gi --timeout 1800 --max-instances 4 --no-cpu-throttling \
  --set-env-vars "$WORKER_ENV" \
  --set-secrets "DATABASE_PASSWORD=${SECRET_NAME}:${SECRET_VERSION}" \
  --quiet

WORKER_URL="$(gcloud run services describe "$WORKER_SERVICE" --project "$PROJECT" \
  --region "$REGION" --format='value(status.url)' 2>/dev/null || true)"
if [[ -z "$WORKER_URL" ]]; then
  if [[ "$DRY_RUN" == true || "$CHECK_ONLY" == true ]]; then
    WORKER_URL="https://${WORKER_SERVICE}-<hash>-ue.a.run.app"
    note "url (not deployed yet): ${WORKER_URL}"
  else
    fail "The worker deployed but reported no URL."
  fi
else
  note "url: ${WORKER_URL}"
fi

# -----------------------------------------------------------------------------
say "3. The queue"
# -----------------------------------------------------------------------------
# One in-flight step per session is what enqueue_evidence_step's deterministic
# task id already guarantees; the dispatch cap is across sessions. Backoff is
# the poll interval the inline loop uses, so a retried step arrives no sooner
# than the next poll would have.
#
# Reading a queue is a separate permission from creating tasks on it, and the
# enqueuer role this deployment runs as does not have it. Told apart, because
# "Missing." over a queue that had just delivered two tasks is a report of the
# account rather than of the project: a refused read means unknown, and the
# create below is then the way to find out, being harmless if it exists.
QUEUE_READ="$(gcloud tasks queues describe "$QUEUE" --project "$PROJECT" \
  --location "$REGION" 2>&1 >/dev/null || true)"
if [[ -z "$QUEUE_READ" ]]; then
  note "Already exists."
else
  if [[ "$QUEUE_READ" == *PERMISSION_DENIED* ]]; then
    note "Cannot tell: this account may not read queues (cloudtasks.queues.get)."
    note "Creating is harmless if it is already there."
  else
    note "Missing. Needs roles/cloudtasks.admin."
  fi
  run gcloud tasks queues create "$QUEUE" \
    --project "$PROJECT" --location "$REGION" \
    --max-concurrent-dispatches 8 --max-attempts 5 \
    --min-backoff 15s --max-backoff 300s || \
    note "  refused or already there -- see the message above."
fi

# -----------------------------------------------------------------------------
say "4. Permissions"
# -----------------------------------------------------------------------------
# add-iam-policy-binding is idempotent, and the policy cannot be read back on
# this account, so each binding is applied rather than tested for. A refusal is
# reported and the rest still run: three separate roles on three surfaces, and
# knowing which of them you already hold is the point of getting to the end.
note "Enqueue tasks (project-level):"
run gcloud projects add-iam-policy-binding "$PROJECT" \
  --member "serviceAccount:${RUNTIME_SA}" \
  --role roles/cloudtasks.enqueuer --condition=None --quiet || \
  note "  refused -- needs roles/resourcemanager.projectIamAdmin."

note "Call the private worker:"
run gcloud run services add-iam-policy-binding "$WORKER_SERVICE" \
  --project "$PROJECT" --region "$REGION" \
  --member "serviceAccount:${RUNTIME_SA}" \
  --role roles/run.invoker --quiet || \
  note "  refused -- needs roles/run.admin on the worker service."

note "Let Cloud Tasks mint the OIDC token it calls the worker with:"
run gcloud iam service-accounts add-iam-policy-binding "$RUNTIME_SA" \
  --project "$PROJECT" \
  --member "serviceAccount:${TASKS_AGENT}" \
  --role roles/iam.serviceAccountTokenCreator --quiet || \
  note "  refused -- needs roles/iam.serviceAccountAdmin."

# -----------------------------------------------------------------------------
say "5. Point both services at the queue"
# -----------------------------------------------------------------------------
# This is the switch. Until these four are set, configured() is false and the
# stage keeps running inline, whatever else above succeeded -- so it is last.
QUEUE_ENV="EVIDENCE_WORKER_URL=${WORKER_URL}"
QUEUE_ENV+=",EVIDENCE_CLOUD_TASKS_QUEUE=${QUEUE}"
QUEUE_ENV+=",EVIDENCE_CLOUD_TASKS_LOCATION=${REGION}"
QUEUE_ENV+=",EVIDENCE_TASKS_SERVICE_ACCOUNT=${RUNTIME_SA}"
run gcloud run services update "$SERVICE" \
  --project "$PROJECT" --region "$REGION" \
  --update-env-vars "$QUEUE_ENV" --quiet

# The worker needs them too, and for the same reason the main service does: a
# task is one poll, and the poll after it is enqueued from inside whichever
# process decided another was needed. Deployed without them the worker answered
# its first task by polling and sleeping for the whole three hundred seconds
# Cloud Run allows, died with a 504, and was retried into a lease the killed
# instance still held. It is a second update rather than part of step 2 because
# EVIDENCE_WORKER_URL is the worker's own address, which Cloud Run does not
# issue until the service exists.
run gcloud run services update "$WORKER_SERVICE" \
  --project "$PROJECT" --region "$REGION" \
  --update-env-vars "$QUEUE_ENV" --quiet

# -----------------------------------------------------------------------------
# Opt-in. Nothing below runs unless its flag was named on the command line.
# -----------------------------------------------------------------------------
if [[ -n "$MIN_INSTANCES" ]]; then
  say "Optional: keep ${MIN_INSTANCES} instance(s) warm"
  note "Billed around the clock whether or not anyone is running research."
  run gcloud run services update "$SERVICE" \
    --project "$PROJECT" --region "$REGION" \
    --min-instances "$MIN_INSTANCES" --quiet
fi

if [[ -n "$SQL_TIER" ]]; then
  say "Optional: Cloud SQL tier -> ${SQL_TIER}"
  note "This RESTARTS ${SQL_INSTANCE}. Any run mid-stage will lose its connection"
  note "and come back through lease recovery. Currently: $(gcloud sql instances \
describe "$SQL_INSTANCE" --project "$PROJECT" --format='value(settings.tier)' 2>/dev/null)"
  if [[ "$DRY_RUN" == false && "$CHECK_ONLY" == false ]]; then
    read -r -p "   Restart the database now? [y/N] " reply
    [[ "$reply" == [yY] ]] || fail "Left the tier alone."
  fi
  run gcloud sql instances patch "$SQL_INSTANCE" \
    --project "$PROJECT" --tier "$SQL_TIER" --quiet
fi

if [[ -n "$ADMIN_TOKEN" ]]; then
  say "Optional: rotate the admin token"
  # Plaintext on the command line and then plaintext in the service's inspectable
  # env-var configuration, which is where it is today and is the lesser half of
  # the problem; the value being "admin" on a public service is the greater one.
  note "Set as a plain env var, as the current one is. To hold it in Secret"
  note "Manager instead, create a secret and use --set-secrets, as the database"
  note "password already does."
  run gcloud run services update "$SERVICE" \
    --project "$PROJECT" --region "$REGION" \
    --update-env-vars "COSCIENTIST_ADMIN_TOKEN=${ADMIN_TOKEN}" --quiet
fi

say "Done"
if [[ "$CHECK_ONLY" == true ]]; then
  note "--check: nothing was changed."
elif [[ "$DRY_RUN" == true ]]; then
  note "--dry-run: nothing was changed."
else
  note "Confirm the switch took by starting a run and watching a session's"
  note "operation.detail: with the queue configured the evidence stage advances"
  note "one poll per task instead of holding a single background call open."
  note ""
  note "  curl -s ${APP_URL:-https://coscientist-766918001064.us-east1.run.app}/api/research/sessions/<id> \\"
  note "    | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d[\"stage\"], d[\"operation\"])'"
fi
