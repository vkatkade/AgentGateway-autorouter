#!/usr/bin/env bash
# Shared settings. Edit, then: source deploy/env.sh
# Producer = where the callout runs. Consumer = the project that owns the Agent Gateway.
# They can be the same project.

export PRODUCER_PROJECT_ID="${PRODUCER_PROJECT_ID:-my-producer-project}"
export CONSUMER_PROJECT_ID="${CONSUMER_PROJECT_ID:-$PRODUCER_PROJECT_ID}"
export REGION="${REGION:-us-central1}"

# Callout service
export SERVICE="${SERVICE:-agw-autorouter}"
export AR_REPO="${AR_REPO:-agw}"
export IMAGE="${REGION}-docker.pkg.dev/${PRODUCER_PROJECT_ID}/${AR_REPO}/${SERVICE}:${IMAGE_TAG:-latest}"
export SA_NAME="${SA_NAME:-agw-autorouter}"
export SA_EMAIL="${SA_NAME}@${PRODUCER_PROJECT_ID}.iam.gserviceaccount.com"
export SECRET_NAME="${SECRET_NAME:-gemini-decisions-api-key}"

# Router behavior (see autorouter/config.py for every knob)
export ARGON_MODEL="${ARGON_MODEL:-gemini-4-argon}"
export HYDROGEN_MODEL="${HYDROGEN_MODEL:-gemini-3.8-flash}"   # swap to the Hydrogen ID at launch
export DECISIONS_TIMEOUT_MS="${DECISIONS_TIMEOUT_MS:-250}"
export SHADOW_MODE="${SHADOW_MODE:-true}"                     # start in shadow; flip to false to route

# Producer networking: regional internal ALB -> serverless NEG -> Cloud Run, published via PSC
export NETWORK="${NETWORK:-default}"
export LB_SUBNET="${LB_SUBNET:-autorouter-lb}"
export LB_SUBNET_RANGE="${LB_SUBNET_RANGE:-10.120.0.0/24}"
export PROXY_SUBNET="${PROXY_SUBNET:-autorouter-proxy-only}"
export PROXY_SUBNET_RANGE="${PROXY_SUBNET_RANGE:-10.120.1.0/24}"
export PSC_NAT_SUBNET="${PSC_NAT_SUBNET:-autorouter-psc-nat}"
export PSC_NAT_SUBNET_RANGE="${PSC_NAT_SUBNET_RANGE:-10.120.2.0/24}"
export NEG="${NEG:-${SERVICE}-neg}"
export BACKEND="${BACKEND:-${SERVICE}-be}"
export URL_MAP="${URL_MAP:-${SERVICE}-um}"
export HTTP_PROXY_NAME="${HTTP_PROXY_NAME:-${SERVICE}-proxy}"
export FWD_RULE="${FWD_RULE:-${SERVICE}-fr}"
export SERVICE_ATTACHMENT="${SERVICE_ATTACHMENT:-${SERVICE}-sa}"

# Service Extensions resources
export PRODUCER_EXTENSION_NAME="${PRODUCER_EXTENSION_NAME:-autorouter}"
export PRODUCER_PHASE="${PRODUCER_PHASE:-AUTHZ}"   # the API accepts only AUTHZ (TRAFFIC is "not supported")
# Events delivered to the callout; REQUEST_BODY is what gives it the generateContent body.
export SUPPORTED_EVENTS="${SUPPORTED_EVENTS:-REQUEST_HEADERS,REQUEST_BODY,RESPONSE_HEADERS}"
export AUTHORITY="${AUTHORITY:-autorouter.internal}"
export BINDING_NAME="${BINDING_NAME:-autorouter-llm-egress}"
export AGENT_GATEWAY_NAME="${AGENT_GATEWAY_NAME:-my-agent-gateway}"
export GATEWAY_LOCATION="${GATEWAY_LOCATION:-$REGION}"
# Vertex hosts the agent calls. Add every regional host your agents use.
export VERTEX_HOSTS="${VERTEX_HOSTS:-aiplatform.googleapis.com ${REGION}-aiplatform.googleapis.com}"

# Reference test client (deploy/06-test-client.sh): a second Cloud Run service that plays
# the gateway's role over ext_proc so you can test the router before a binding exists.
export CLIENT_SERVICE="${CLIENT_SERVICE:-agw-client}"
export CLIENT_SA_NAME="${CLIENT_SA_NAME:-agw-client}"
export CLIENT_SA_EMAIL="${CLIENT_SA_NAME}@${PRODUCER_PROJECT_ID}.iam.gserviceaccount.com"
export CLIENT_IMAGE="${REGION}-docker.pkg.dev/${PRODUCER_PROJECT_ID}/${AR_REPO}/${CLIENT_SERVICE}:${IMAGE_TAG:-latest}"
export CLIENT_SUBNET="${CLIENT_SUBNET:-default}"          # subnet in $NETWORK for Direct VPC egress
export VERTEX_LOCATION="${VERTEX_LOCATION:-global}"
