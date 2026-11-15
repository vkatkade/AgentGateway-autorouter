#!/usr/bin/env bash
# Publish the callout as a ProducerExtension (command shape from the extension-bindings docs).
set -euo pipefail
source "$(dirname "$0")/env.sh"

SA_URI="projects/${PRODUCER_PROJECT_ID}/regions/${REGION}/serviceAttachments/${SERVICE_ATTACHMENT}"

# The docs only show --phase authz. Confirm with the Agent Gateway team which phase delivers
# and accepts REQUEST_BODY mutations on Agent -> LLM egress before going past shadow mode.
# The docs' example has no --location flag; add one here if the preview CLI asks for it.
gcloud beta service-extensions producer-extensions create "$PRODUCER_EXTENSION_NAME" \
  --project="$PRODUCER_PROJECT_ID" \
  --phase "$PRODUCER_PHASE" \
  --extension-settings="service=${SA_URI},authority=${AUTHORITY}"

echo "ProducerExtension: projects/${PRODUCER_PROJECT_ID}/locations/${REGION}/producerExtensions/${PRODUCER_EXTENSION_NAME}"
