#!/usr/bin/env bash
# Regional internal Application Load Balancer -> serverless NEG -> Cloud Run,
# then a Private Service Connect service attachment on the LB's forwarding rule.
set -euo pipefail
source "$(dirname "$0")/env.sh"
P=(--project="$PRODUCER_PROJECT_ID")
R=(--region="$REGION")

exists() { "$@" >/dev/null 2>&1; }

# Subnets: one for the LB frontend IP, a proxy-only subnet, and a PSC NAT subnet.
exists gcloud compute networks subnets describe "$LB_SUBNET" "${P[@]}" "${R[@]}" ||
  gcloud compute networks subnets create "$LB_SUBNET" "${P[@]}" "${R[@]}" \
    --network="$NETWORK" --range="$LB_SUBNET_RANGE"
exists gcloud compute networks subnets describe "$PROXY_SUBNET" "${P[@]}" "${R[@]}" ||
  gcloud compute networks subnets create "$PROXY_SUBNET" "${P[@]}" "${R[@]}" \
    --network="$NETWORK" --range="$PROXY_SUBNET_RANGE" \
    --purpose=REGIONAL_MANAGED_PROXY --role=ACTIVE
exists gcloud compute networks subnets describe "$PSC_NAT_SUBNET" "${P[@]}" "${R[@]}" ||
  gcloud compute networks subnets create "$PSC_NAT_SUBNET" "${P[@]}" "${R[@]}" \
    --network="$NETWORK" --range="$PSC_NAT_SUBNET_RANGE" --purpose=PRIVATE_SERVICE_CONNECT

# Serverless NEG + backend service (HTTP2 so gRPC streams survive end to end).
exists gcloud compute network-endpoint-groups describe "$NEG" "${P[@]}" "${R[@]}" ||
  gcloud compute network-endpoint-groups create "$NEG" "${P[@]}" "${R[@]}" \
    --network-endpoint-type=serverless --cloud-run-service="$SERVICE"
exists gcloud compute backend-services describe "$BACKEND" "${P[@]}" "${R[@]}" || {
  gcloud compute backend-services create "$BACKEND" "${P[@]}" "${R[@]}" \
    --load-balancing-scheme=INTERNAL_MANAGED --protocol=HTTP2
  gcloud compute backend-services add-backend "$BACKEND" "${P[@]}" "${R[@]}" \
    --network-endpoint-group="$NEG" --network-endpoint-group-region="$REGION"
}

exists gcloud compute url-maps describe "$URL_MAP" "${P[@]}" "${R[@]}" ||
  gcloud compute url-maps create "$URL_MAP" "${P[@]}" "${R[@]}" --default-service="$BACKEND"
exists gcloud compute target-http-proxies describe "$HTTP_PROXY_NAME" "${P[@]}" "${R[@]}" ||
  gcloud compute target-http-proxies create "$HTTP_PROXY_NAME" "${P[@]}" "${R[@]}" --url-map="$URL_MAP"

exists gcloud compute forwarding-rules describe "$FWD_RULE" "${P[@]}" "${R[@]}" ||
  gcloud compute forwarding-rules create "$FWD_RULE" "${P[@]}" "${R[@]}" \
    --load-balancing-scheme=INTERNAL_MANAGED --network="$NETWORK" --subnet="$LB_SUBNET" \
    --target-http-proxy="$HTTP_PROXY_NAME" --target-http-proxy-region="$REGION" --ports=80

# PSC service attachment. ACCEPT_MANUAL + an explicit consumer allowlist; switch to
# ACCEPT_AUTOMATIC only if you are comfortable with any project connecting.
exists gcloud compute service-attachments describe "$SERVICE_ATTACHMENT" "${P[@]}" "${R[@]}" ||
  gcloud compute service-attachments create "$SERVICE_ATTACHMENT" "${P[@]}" "${R[@]}" \
    --producer-forwarding-rule="$FWD_RULE" \
    --connection-preference=ACCEPT_MANUAL \
    --consumer-accept-list="${CONSUMER_PROJECT_ID}=10" \
    --nat-subnets="$PSC_NAT_SUBNET"

echo "Service attachment: projects/${PRODUCER_PROJECT_ID}/regions/${REGION}/serviceAttachments/${SERVICE_ATTACHMENT}"
