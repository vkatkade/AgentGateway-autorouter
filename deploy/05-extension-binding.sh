#!/usr/bin/env bash
# Render and import the consumer-side ExtensionBinding on the Agent Gateway.
# Run against the consumer project; the producer must have accepted its PSC connection.
set -euo pipefail
source "$(dirname "$0")/env.sh"
DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$DIR/.rendered/extension-binding.yaml"
mkdir -p "$DIR/.rendered"

HOSTS_YAML=""
for h in $VERTEX_HOSTS; do
  HOSTS_YAML+="      - exact: \"${h}\""$'\n'
done
export HOSTS_YAML="${HOSTS_YAML%$'\n'}"

# perl instead of envsubst: present on macOS and Cloud Shell alike.
perl -pe 's/\$\{(\w+)\}/defined $ENV{$1} ? $ENV{$1} : die "unset: $1\n"/ge' \
  "$DIR/extension-binding.yaml.tmpl" > "$OUT"
echo "--- $OUT"; cat "$OUT"; echo "---"

gcloud beta service-extensions extension-bindings import "$BINDING_NAME" \
  --project="$CONSUMER_PROJECT_ID" \
  --location="$GATEWAY_LOCATION" \
  --source="$OUT"
