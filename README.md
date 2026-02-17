# H.E.I.M.D.A.L.L

<p align="center">
  <img src="docs/assets/images/heimdall_banner.png" alt="H.E.I.M.D.A.L.L" width="100%" style="max-width: 1200px" />
</p>

**Heroically Excessive Inference Methodology (for) Data Analytics (on) Large Loads**

A telemetry-to-insight pipeline for robotics and autonomous systems. Turns fleet telemetry into natural-language insights via GPU-accelerated data loading (cuDF + UVM), NVIDIA NIM on GKE for LLM inference, and model format selection for production deployment.

## Quick Start (Notebooks 01 and 02)

1. Open [01 Data Ingest](https://colab.research.google.com/github/KarthikSriramGit/H.E.I.M.D.A.L.L/blob/main/notebooks/01_data_ingest_benchmark.ipynb) or [02 Inference Pipeline](https://colab.research.google.com/github/KarthikSriramGit/H.E.I.M.D.A.L.L/blob/main/notebooks/02_inference_pipeline.ipynb) in Colab.
2. **Runtime > Change runtime type > Hardware accelerator: GPU (T4) > Save.**
3. Run all cells.

## Setup for Notebook 03 (Full Pipeline with NIM)

Notebook 03 needs NIM running on GKE. Follow these steps.

### Prerequisites

- Google account with [Google Cloud](https://console.cloud.google.com) (billing enabled)
- [NGC account](https://ngc.nvidia.com) for the API key

### Step 1: Create a GCP project (if needed)

1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Create a project or select an existing one
3. Enable billing for the project
4. Note your **Project ID** (e.g. `heimdall-487621`)

### Step 2: Enable APIs

1. Go to [APIs & Services > Library](https://console.cloud.google.com/apis/library)
2. Enable **Kubernetes Engine API** and **Compute Engine API**

### Step 3: Get your NGC API key

1. Go to [ngc.nvidia.com](https://ngc.nvidia.com) and sign in
2. Profile (top right) > **Setup** > **Generate API Key**
3. Copy the key (starts with `nvapi-`)

### Step 4: Deploy NIM on GKE

Open [Google Cloud Shell](https://shell.cloud.google.com) (browser terminal) and run:

```bash
# Set your values
export PROJECT_ID="your-gcp-project-id"
export ZONE="us-central1-a"
export NGC_CLI_API_KEY="your-ngc-api-key"

# Configure gcloud
gcloud config set project $PROJECT_ID

# Create cluster and GPU node pool (skip if already exists)
gcloud container clusters create nim-demo \
  --project=$PROJECT_ID \
  --location=$ZONE \
  --release-channel=rapid \
  --machine-type=e2-standard-4 \
  --num-nodes=1

gcloud container node-pools create gpupool \
  --accelerator type=nvidia-l4,count=1,gpu-driver-version=latest \
  --project=$PROJECT_ID \
  --location=$ZONE \
  --cluster=nim-demo \
  --machine-type=g2-standard-16 \
  --num-nodes=1

# Connect to cluster
gcloud container clusters get-credentials nim-demo --zone=$ZONE --project=$PROJECT_ID

# Clone repo and deploy NIM
git clone -q https://github.com/KarthikSriramGit/H.E.I.M.D.A.L.L.git
cd H.E.I.M.D.A.L.L
bash src/deploy/gke/deploy_nim.sh

# Expose NIM with LoadBalancer
kubectl patch svc my-nim-nim-llm -n nim -p '{"spec": {"type": "LoadBalancer"}}'
kubectl get svc -n nim
```

**If cluster already exists**, run only:

```bash
export PROJECT_ID="your-gcp-project-id"
export ZONE="us-central1-a"
export NGC_CLI_API_KEY="your-ngc-api-key"
gcloud config set project $PROJECT_ID
gcloud container clusters get-credentials nim-demo --zone=$ZONE --project=$PROJECT_ID
git clone -q https://github.com/KarthikSriramGit/H.E.I.M.D.A.L.L.git
cd H.E.I.M.D.A.L.L
bash src/deploy/gke/deploy_nim.sh
kubectl patch svc my-nim-nim-llm -n nim -p '{"spec": {"type": "LoadBalancer"}}'
kubectl get svc -n nim
```

### Step 5: Wait and get your NIM URL

1. Wait for **EXTERNAL-IP** to appear (not `<pending>`):
   ```bash
   kubectl get svc -n nim
   ```

2. Wait for the pod to be **Running**:
   ```bash
   kubectl get pods -n nim
   ```

3. Note your NIM URL: `http://EXTERNAL_IP:8000` (replace `EXTERNAL_IP` with the IP from `kubectl get svc -n nim`)

**Do not commit this URL.** Keep it private.

### Step 6: Run notebook 03 on Colab

1. Open [03 Query Telemetry](https://colab.research.google.com/github/KarthikSriramGit/H.E.I.M.D.A.L.L/blob/main/notebooks/03_query_telemetry.ipynb) in Colab
2. Store your NIM URL securely using **Colab Secrets**:
   - Click the **key icon** (Secrets) in the left sidebar
   - Add: Name = `NIM_BASE_URL`, Value = `http://YOUR_EXTERNAL_IP:8000`
   - Toggle **Notebook access** to ON
3. Run all cells

### Troubleshooting: Connection refused from Colab

If you get `ConnectionRefusedError` when calling NIM from Colab, run these in Cloud Shell:

**1. Check the NIM pod is Running**
```bash
kubectl get pods -n nim
```
Wait until `STATUS` is `Running` and `READY` is `1/1`. First deployment can take 10–20 minutes for the model to download.

**2. Test from Cloud Shell (port-forward)**
```bash
kubectl port-forward svc/my-nim-nim-llm 8000:8000 -n nim &
sleep 5
curl -s http://localhost:8000/v1/models
kill %1
```
If this works, NIM is fine; the problem is external access.

**3. Check service endpoints**
```bash
kubectl get endpoints -n nim
```
`my-nim-nim-llm` should have at least one address. If it shows `<none>`, the pod is not ready.

**4. Verify LoadBalancer backends**
In [GCP Console > Network Services > Load balancing](https://console.cloud.google.com/net-services/loadbalancing/list/loadBalancers), open the load balancer for the NIM service and confirm the backends are healthy.

**5. Allow firewall (if needed)**
Create a firewall rule to allow ingress on port 8000:
```bash
gcloud compute firewall-rules create allow-nim-8000 \
  --allow tcp:8000 \
  --source-ranges 0.0.0.0/0 \
  --target-tags $(kubectl get nodes -o jsonpath='{.items[0].metadata.labels.cloud\.google\.com\/gke-nodepool}' 2>/dev/null || echo "gke-nim-demo")
```

### Step 7: Cleanup (when finished)

To avoid ongoing GCP charges:

```bash
kubectl delete namespace nim
gcloud container clusters delete nim-demo --zone=$ZONE --project=$PROJECT_ID --quiet
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| [01 Data Ingest](https://colab.research.google.com/github/KarthikSriramGit/H.E.I.M.D.A.L.L/blob/main/notebooks/01_data_ingest_benchmark.ipynb) | cuDF + UVM loading, pandas vs cuDF benchmark (GPU recommended) |
| [02 Inference Pipeline](https://colab.research.google.com/github/KarthikSriramGit/H.E.I.M.D.A.L.L/blob/main/notebooks/02_inference_pipeline.ipynb) | Format selection, Gemma 2 2B local inference, latency plots (GPU + HF token) |
| [03 Query Telemetry](https://colab.research.google.com/github/KarthikSriramGit/H.E.I.M.D.A.L.L/blob/main/notebooks/03_query_telemetry.ipynb) | Full pipeline: retrieve data, NIM summarization, latency plots (requires NIM on GKE) |

## Architecture

```
Data Layer               Inference Layer               Deployment Layer
Synthetic Generator  -->  cuDF + UVM Loader  -->       Format Selector
                         Benchmark (pandas vs cuDF)    Inference Pipeline
                                                       Metrics (p50, p90, TTFT)
                         Query Engine  <--  NIM Client  <--  NIM on GKE
```

## Results & Takeaways

Benchmark results from running the notebooks on Colab with a T4 GPU (notebook 01, 02) and NIM on GKE (notebook 03).

### 1. Data ingest: pandas vs cuDF (2M rows)

| Backend | Operation | Time (s) | Memory (MB) |
|---------|-----------|----------|-------------|
| pandas  | load      | 0.88     | 105.5       |
| pandas  | groupby   | 0.14     | 62.8        |
| pandas  | filter    | 0.03     | 12.6        |
| pandas  | sort      | 0.36     | 534.1       |
| cuDF    | load      | 0.17     | 0.2         |
| cuDF    | groupby   | 0.01     | 0.03        |
| cuDF    | filter    | 0.02     | 0.2         |
| cuDF    | sort      | 0.03     | 0.2         |

![pandas vs cuDF benchmark](docs/assets/01_benchmark_pandas_vs_cudf.png)

**Takeaway:** cuDF gives ~5× faster load and ~10–13× faster groupby/sort, with negligible host memory (data stays in GPU VRAM via UVM spill).

### 2. Local inference: Gemma 2 2B (notebook 02)

Natural-language answers to telemetry queries on a T4 GPU. Typical run: ~21s total for 5 queries (~4.2s avg latency).

![Local Gemma 2 2B inference](docs/assets/02_inference_local_gemma.png)

**Takeaway:** Local GGUF inference works for low-latency, offline telemetry Q&A. Good for prototyping; heavier workloads may need cloud scaling.

### 3. Full pipeline: NIM on GKE (notebook 03)

Same queries via Llama 3 8B on NVIDIA NIM (L4 on GKE). Longer answers, higher latency due to model size and network.

![NIM Llama 3 8B on GKE](docs/assets/03_query_nim.png)

**Takeaway:** NIM on GKE scales inference for production. Use notebook 02 for fast local iteration and notebook 03 for production-style deployment.

## Inspiration

1. [Deploy Faster Generative AI models with NVIDIA NIM on GKE](https://developers.google.com/learn/pathways/deploy-faster-gen-ai-models-nvidia-gke)
2. [Intro to Inference: How to Run AI Models on a GPU](https://developers.google.com/learn/pathways/ai-models-on-gpu-intro)
3. [Speed Up Data Analytics on GPUs](https://developers.google.com/learn/pathways/speed-up-data-analytics-GPUs)

## Real Data

See [data/README_data_sources.md](data/README_data_sources.md) for a plan to gather real telemetry from nuScenes, KITTI, CARLA, ROS2 bags, and OBD-II CAN data.

## Contributing

Contributions are welcome. Areas: adapters for nuScenes/KITTI, ROS2 bag-to-Parquet scripts, NIM prompt templates, benchmark results on different GPU configurations.

## License

Apache 2.0
