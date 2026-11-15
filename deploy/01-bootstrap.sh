#!/usr/bin/env bash
# APIs, Artifact Registry, service account, and the Decisions API key in Secret Manager.
# Usage: source deploy/env.sh && ./deploy/01-bootstrap.sh   (prompts for the key)
set -euo pipefail
source "$(dirname "$0")/env.sh"
gcloud config set project "$PRODUCER_PROJECT_ID" >/dev/null

gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com \
  secretmanager.googleapis.com compute.googleapis.com networkservices.googleapis.com

gcloud artifacts repositories describe "$AR_REPO" --location="$REGION" >/dev/null 2>&1 ||
  gcloud artifacts repositories create "$AR_REPO" --repository-format=docker --location="$REGION"

gcloud iam service-accounts describe "$SA_EMAIL" >/dev/null 2>&1 ||
  gcloud iam service-accounts create "$SA_NAME" --display-name="Agent Gateway auto-router"

if ! gcloud secrets describe "$SECRET_NAME" >/dev/null 2>&1; then
  gcloud secrets create "$SECRET_NAME" --replication-policy=automatic
fi
read -r -s -p "Gemini API key for the Decisions API (blank = keep current): " KEY; echo
if [[ -n "$KEY" ]]; then
  printf '%s' "$KEY" | gcloud secrets versions add "$SECRET_NAME" --data-file=-
fi

gcloud secrets add-iam-policy-binding "$SECRET_NAME" \
  --member="serviceAccount:${SA_EMAIL}" --role=roles/secretmanager.secretAccessor >/dev/null
echo "Bootstrap done."
