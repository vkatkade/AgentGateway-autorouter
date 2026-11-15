#!/usr/bin/env bash
# Build the image and deploy the callout to Cloud Run as plaintext gRPC over HTTP/2.
# Per the extension-bindings docs the producer must: listen on plaintext gRPC, accept HTTP/2,
# and allow unauthenticated access (reachability is limited to the internal LB via ingress).
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$(dirname "$0")/.."

gcloud builds submit --project="$PRODUCER_PROJECT_ID" --tag "$IMAGE" .

gcloud run deploy "$SERVICE" \
  --project="$PRODUCER_PROJECT_ID" --region="$REGION" \
  --image="$IMAGE" \
  --use-http2 --port=8080 \
  --allow-unauthenticated \
  --ingress=internal-and-cloud-load-balancing \
  --service-account="$SA_EMAIL" \
  --set-secrets="GEMINI_API_KEY=${SECRET_NAME}:latest" \
  --set-env-vars="ARGON_MODEL=${ARGON_MODEL},HYDROGEN_MODEL=${HYDROGEN_MODEL},DECISIONS_TIMEOUT_MS=${DECISIONS_TIMEOUT_MS},SHADOW_MODE=${SHADOW_MODE},SERVER_THREADS=64" \
  --min-instances=1 \
  --concurrency=64 \
  --cpu=1 --memory=512Mi \
  --timeout=300

echo "Deployed: $(gcloud run services describe "$SERVICE" --region="$REGION" --format='value(status.url)')"
