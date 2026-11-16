#!/usr/bin/env bash
# Reference test client: a second Cloud Run service with a prompt page. It sends each prompt
# through the router over ext_proc gRPC (as Agent Gateway would), applies the rewrite, and
# calls Vertex AI. Reaches the router's internal LB over Direct VPC egress. Requires auth.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$(dirname "$0")/../client"
P=(--project="$PRODUCER_PROJECT_ID"); R=(--region="$REGION")

ROUTER_IP=$(gcloud compute forwarding-rules describe "$FWD_RULE" "${P[@]}" "${R[@]}" --format='value(IPAddress)')
[[ -n "$ROUTER_IP" ]] || { echo "Forwarding rule $FWD_RULE not found; run 03-lb-and-psc.sh first" >&2; exit 1; }

gcloud iam service-accounts describe "$CLIENT_SA_EMAIL" "${P[@]}" >/dev/null 2>&1 || {
  gcloud iam service-accounts create "$CLIENT_SA_NAME" "${P[@]}" --display-name="Auto-router test client"
  sleep 10  # let the new service account propagate before binding roles
}
gcloud projects add-iam-policy-binding "$PRODUCER_PROJECT_ID" --member="serviceAccount:${CLIENT_SA_EMAIL}" \
  --role=roles/aiplatform.user --condition=None >/dev/null

gcloud builds submit "${P[@]}" "${R[@]}" --tag "$CLIENT_IMAGE" .
gcloud run deploy "$CLIENT_SERVICE" "${P[@]}" "${R[@]}" \
  --image="$CLIENT_IMAGE" \
  --service-account="$CLIENT_SA_EMAIL" \
  --no-allow-unauthenticated \
  --network="$NETWORK" --subnet="$CLIENT_SUBNET" --vpc-egress=private-ranges-only \
  --set-env-vars="ROUTER_TARGET=${ROUTER_IP}:80,PROJECT_ID=${PRODUCER_PROJECT_ID},VERTEX_LOCATION=${VERTEX_LOCATION}" \
  --cpu=1 --memory=512Mi --timeout=180

echo
echo "Open the test client:"
echo "  gcloud run services proxy $CLIENT_SERVICE --project=$PRODUCER_PROJECT_ID --region=$REGION"
echo "  then browse http://localhost:8080"
