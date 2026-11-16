#!/usr/bin/env bash
# Remove everything the deploy scripts created (secret and SA are kept).
set -uo pipefail
source "$(dirname "$0")/env.sh"
P=(--project="$PRODUCER_PROJECT_ID"); R=(--region="$REGION"); Q=(--quiet)

gcloud beta service-extensions extension-bindings delete "$BINDING_NAME" \
  --project="$CONSUMER_PROJECT_ID" --location="$GATEWAY_LOCATION" "${Q[@]}"
# ProducerExtension: REST (no public gcloud command yet).
curl -sS -X DELETE -H "Authorization: Bearer ${CLOUDSDK_AUTH_ACCESS_TOKEN:-$(gcloud auth print-access-token)}" \
  -H "x-goog-user-project: ${PRODUCER_PROJECT_ID}" \
  "https://networkservices.googleapis.com/v1beta1/projects/${PRODUCER_PROJECT_ID}/locations/${REGION}/producerExtensions/${PRODUCER_EXTENSION_NAME}" >/dev/null
gcloud run services delete "$CLIENT_SERVICE" "${P[@]}" "${R[@]}" "${Q[@]}"   # test client, if deployed
gcloud compute service-attachments delete "$SERVICE_ATTACHMENT" "${P[@]}" "${R[@]}" "${Q[@]}"
gcloud compute forwarding-rules delete "$FWD_RULE" "${P[@]}" "${R[@]}" "${Q[@]}"
gcloud compute target-http-proxies delete "$HTTP_PROXY_NAME" "${P[@]}" "${R[@]}" "${Q[@]}"
gcloud compute url-maps delete "$URL_MAP" "${P[@]}" "${R[@]}" "${Q[@]}"
gcloud compute backend-services delete "$BACKEND" "${P[@]}" "${R[@]}" "${Q[@]}"
gcloud compute network-endpoint-groups delete "$NEG" "${P[@]}" "${R[@]}" "${Q[@]}"
gcloud compute networks subnets delete "$PSC_NAT_SUBNET" "$PROXY_SUBNET" "$LB_SUBNET" "${P[@]}" "${R[@]}" "${Q[@]}"
gcloud run services delete "$SERVICE" "${P[@]}" "${R[@]}" "${Q[@]}"
