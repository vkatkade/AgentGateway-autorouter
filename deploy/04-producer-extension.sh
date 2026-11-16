#!/usr/bin/env bash
# ProducerExtension pointing at the PSC service attachment.
# Uses the Network Services REST API: public gcloud (beta/alpha) has no producer-extensions
# command yet. `supportedEvents` is required; REQUEST_BODY is what delivers the body.
set -euo pipefail
source "$(dirname "$0")/env.sh"
SA_URI="projects/${PRODUCER_PROJECT_ID}/regions/${REGION}/serviceAttachments/${SERVICE_ATTACHMENT}"
PE="projects/${PRODUCER_PROJECT_ID}/locations/${REGION}/producerExtensions/${PRODUCER_EXTENSION_NAME}"
API="https://networkservices.googleapis.com/v1beta1"
TOKEN="${CLOUDSDK_AUTH_ACCESS_TOKEN:-$(gcloud auth print-access-token)}"
H=(-H "Authorization: Bearer ${TOKEN}" -H "x-goog-user-project: ${PRODUCER_PROJECT_ID}" -H "Content-Type: application/json")

if curl -sf "${H[@]}" "${API}/${PE}" >/dev/null; then
  echo "ProducerExtension already exists: ${PE}"; exit 0
fi
EVENTS=$(printf '"%s",' ${SUPPORTED_EVENTS//,/ }); EVENTS="[${EVENTS%,}]"
BODY=$(cat <<JSON
{"phase":"${PRODUCER_PHASE}","description":"Agent Gateway auto-router",
 "extensionSettings":{"service":"${SA_URI}","authority":"${AUTHORITY}","supportedEvents":${EVENTS}}}
JSON
)
OP=$(curl -sS "${H[@]}" -X POST "${API}/projects/${PRODUCER_PROJECT_ID}/locations/${REGION}/producerExtensions?producerExtensionId=${PRODUCER_EXTENSION_NAME}" -d "$BODY")
echo "$OP" | grep -q '"error"' && { echo "$OP" >&2; exit 1; }
OP_NAME=$(echo "$OP" | python3 -c 'import json,sys;print(json.load(sys.stdin)["name"])')
for _ in $(seq 1 30); do
  R=$(curl -sS "${H[@]}" "${API}/${OP_NAME}")
  if echo "$R" | grep -q '"done": true'; then
    echo "$R" | grep -q '"error"' && { echo "$R" >&2; exit 1; }
    echo "ProducerExtension: ${PE}"; exit 0
  fi
  sleep 4
done
echo "Timed out waiting for ${OP_NAME}" >&2; exit 1
