#!/bin/bash
# Create GKE cluster with GPU node pool for NVIDIA NIM.

set -e

export PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID}"
export REGION="${REGION:?Set REGION}"
export ZONE="${ZONE:?Set ZONE}"
export CLUSTER_NAME="${CLUSTER_NAME:-nim-demo}"
export NODE_POOL_MACHINE_TYPE="${NODE_POOL_MACHINE_TYPE:-g2-standard-16}"
export CLUSTER_MACHINE_TYPE="${CLUSTER_MACHINE_TYPE:-e2-standard-4}"
export GPU_TYPE="${GPU_TYPE:-nvidia-l4}"
export GPU_COUNT="${GPU_COUNT:-1}"

echo "Creating GKE cluster: ${CLUSTER_NAME}"

gcloud container clusters create "${CLUSTER_NAME}" \
    --project="${PROJECT_ID}" \
    --location="${ZONE}" \
    --release-channel=rapid \
    --machine-type="${CLUSTER_MACHINE_TYPE}" \
    --num-nodes=1

echo "Creating GPU node pool: gpupool"

gcloud container node-pools create gpupool \
    --accelerator type="${GPU_TYPE}",count="${GPU_COUNT}",gpu-driver-version=latest \
    --project="${PROJECT_ID}" \
    --location="${ZONE}" \
    --cluster="${CLUSTER_NAME}" \
    --machine-type="${NODE_POOL_MACHINE_TYPE}" \
    --num-nodes=1

echo "Cluster ${CLUSTER_NAME} ready. Get credentials with:"
echo "  gcloud container clusters get-credentials ${CLUSTER_NAME} --zone=${ZONE}"
