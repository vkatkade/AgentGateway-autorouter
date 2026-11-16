# Reference test client

A Cloud Run service with a prompt page that exercises the deployed router the way Agent
Gateway would, over Envoy ext_proc v3 gRPC, then calls Vertex AI with the rewritten request.
Use it to test the router before an ExtensionBinding exists.

Deploy with `deploy/06-test-client.sh` (after deploy steps 1–3), then open it with
`gcloud run services proxy agw-client --region=$REGION` and browse http://localhost:8080.

| Env var | Default | Meaning |
|---|---|---|
| `ROUTER_TARGET` | `10.120.0.2:80` | Router address: the internal LB forwarding rule (set by the deploy script) |
| `ROUTER_TLS` | `false` | Use TLS to the router (e.g. its Cloud Run URL on :443) instead of h2c |
| `PROJECT_ID` | ADC project | Project used in the Vertex path and billed for calls |
| `VERTEX_LOCATION` | `global` | Vertex location; `global` uses aiplatform.googleapis.com |
| `DEFAULT_MODEL` | `gemini-3.8-flash` | Model the simulated agent asks for |

`POST /api/prompt` with `{"prompt", "model"?, "system"?, "opt_out"?, "route_only"?}` returns the
router's headers, whether the request was rewritten, the model and `generationConfig` sent to Vertex,
and the answer. This is a test tool: it handles single-turn, non-streaming `generateContent` only.
