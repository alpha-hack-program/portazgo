#!/usr/bin/env bash
# Get APPS_DOMAIN from OpenShift (oc), set LLAMA_STACK_HOST, then run the simple_invoke example.
# Arguments = the question to ask the agent (not the script path).
#
# Usage (from cohorte):
#   ./scripts/run_invoke_oc.sh "What is 2+2?"
#   ./scripts/run_invoke_oc.sh "Tell me about taxes in Lysmark."
#   ./scripts/run_invoke_oc.sh                    # uses default question
#
# Requires: oc logged in, .env optional (PROJECT, AGENT_MODEL_ID, etc.)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

# Load .env if present (PROJECT, AGENT_MODEL_ID, AGENT_VECTOR_STORE_NAME, etc.) from the scripts directory
CURRENT_DIR="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "${SCRIPT_DIR}/.env" ]; then
  set -a
  # shellcheck source=/dev/null
  source "$SCRIPT_DIR/.env"
  set +a
fi

PROJECT="${PROJECT:-llama-stack-demo}"

echo "Getting APPS_DOMAIN from OpenShift (oc)..."
APPS_DOMAIN=$(oc get ingresses.config.openshift.io cluster -o jsonpath='{.spec.domain}')
if [ -z "$APPS_DOMAIN" ]; then
  echo "Error: could not get APPS_DOMAIN. Ensure you are logged in (oc login) and the cluster has ingress config." >&2
  exit 1
fi
echo "APPS_DOMAIN=$APPS_DOMAIN"

export LLAMA_STACK_HOST="${PROJECT}-route-${PROJECT}.${APPS_DOMAIN}"
export LLAMA_STACK_PORT="${LLAMA_STACK_PORT:-443}"
export LLAMA_STACK_SECURE="${LLAMA_STACK_SECURE:-True}"

echo "LLAMA_STACK_HOST=${LLAMA_STACK_HOST}"
# Strip any leading arg that looks like the example script path (e.g. user pasted "examples/simple_invoke.py" by mistake)
INVOKE_ARGS=()
for arg in "$@"; do
  if [[ "$arg" == *"simple_invoke"* ]]; then
    [ ${#INVOKE_ARGS[@]} -eq 0 ] && echo "Hint: skipping script path in arguments; pass only the question and --tools."
    continue
  fi
  INVOKE_ARGS+=("$arg")
done
echo "Running: uv run python examples/simple_invoke.py ${INVOKE_ARGS[*]}"
echo "---"
exec uv run python examples/simple_invoke.py "${INVOKE_ARGS[@]}"
