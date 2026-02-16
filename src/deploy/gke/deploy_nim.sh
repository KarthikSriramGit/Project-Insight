#!/bin/bash
# Deploy NVIDIA NIM to GKE.
# Prerequisites: cluster created and NGC_CLI_API_KEY set in environment.
# In Cloud Shell: export NGC_CLI_API_KEY="your-key"

set -e

NGC_CLI_API_KEY="${NGC_CLI_API_KEY:?Set NGC_CLI_API_KEY}"
NIM_CHART_VERSION="${NIM_CHART_VERSION:-1.3.0}"
NAMESPACE="${NAMESPACE:-nim}"
RELEASE_NAME="${RELEASE_NAME:-my-nim}"

echo "Fetching NIM Helm chart ${NIM_CHART_VERSION}"
helm fetch "https://helm.ngc.nvidia.com/nim/charts/nim-llm-${NIM_CHART_VERSION}.tgz" \
    --username='$oauthtoken' \
    --password="${NGC_CLI_API_KEY}"

echo "Creating namespace ${NAMESPACE}"
kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

echo "Creating registry secret"
kubectl create secret docker-registry registry-secret \
    --docker-server=nvcr.io \
    --docker-username='$oauthtoken' \
    --docker-password="${NGC_CLI_API_KEY}" \
    -n "${NAMESPACE}" \
    --dry-run=client -o yaml | kubectl apply -f -

echo "Creating NGC API secret"
kubectl create secret generic ngc-api \
    --from-literal=NGC_API_KEY="${NGC_CLI_API_KEY}" \
    -n "${NAMESPACE}" \
    --dry-run=client -o yaml | kubectl apply -f -

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALUES_FILE="${SCRIPT_DIR}/nim_values.yaml"

echo "Installing NIM"
helm upgrade --install "${RELEASE_NAME}" "nim-llm-${NIM_CHART_VERSION}.tgz" \
    -f "${VALUES_FILE}" \
    --namespace "${NAMESPACE}"

echo "Deployment started. Check status with:"
echo "  kubectl get pods -n ${NAMESPACE}"
echo ""
echo "When pod is Running, port-forward to test:"
echo "  kubectl port-forward service/${RELEASE_NAME}-nim-llm 8000:8000 -n ${NAMESPACE}"
