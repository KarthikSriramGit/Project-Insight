<div align="center">

# H.E.I.M.D.A.L.L
</div>

<p align="center">
  <img src="./docs/assets/images/heimdall_banner.png" alt="H.E.I.M.D.A.L.L" width="100%" style="max-width: 1200px" />
</p>

<p align="center">
  <img src="./docs/assets/images/heimdall_waymo_banner.png" alt="H.E.I.M.D.A.L.L with Waymo fleet" width="100%" style="max-width: 1200px" />
</p>

**Heroically Excessive Inference Methodology (for) Data Analytics (on) Large Loads**

A telemetry-to-insight pipeline for robotics and autonomous systems. Turns fleet telemetry into natural-language insights via GPU-accelerated data loading (cuDF + UVM), NVIDIA NIM on GKE for LLM inference, and model format selection for production deployment.

*Named after Heimdall, the Norse guardian from above, who sees and hears everything. Like him, this pipeline watches over your fleet telemetry from the "cloud" and turns it into insights.*

---

## Introduction

Imagine a fleet of a thousand Waymo autonomous vehicles, Tesla Optimus units, or similar—each generating streams of telemetry. You need to identify which units had anomalous brake events or abnormal sensor measurements last week, or which vehicles exceeded a speed threshold in a given region, or which robots showed elevated motor temperatures during a deployment. Manually querying and cross-referencing that data across hundreds of assets does not scale.

H.E.I.M.D.A.L.L addresses this. You load your fleet telemetry into the pipeline, then ask natural-language questions such as *"Which vehicles had brake pressure above 90% in the last 24 hours?"* or *"List robots with gyro z-axis variance exceeding 0.5."* The system returns responses with vehicle or robot IDs, timestamps, and relevant metrics. This enables rapid insights and operational visibility across large fleets of cars, autonomous vehicles, or robots; without writing complex queries.

---

## Table of Contents

- [Introduction](#introduction)
- [What You Need](#what-you-need)
- [Choose Your Path](#choose-your-path)
- [Quick Start (Notebooks 01 & 02)](#quick-start-notebooks-01--02)
- [Notebooks](#notebooks)
- [Setup for Notebook 03 (NIM on GKE)](#setup-for-notebook-03-nim-on-gke)
- [Architecture](#architecture)
- [Results & Takeaways](#results--takeaways)
- [Troubleshooting](#troubleshooting)
- [Real Data](#real-data)
- [Contributing](#contributing)
- [Code of Conduct](#code-of-conduct)

---

## What You Need

| To run… | You need |
|--------|----------|
| **Notebook 01** (Data Ingest) | Colab account, GPU runtime (L4/T4 recommended) |
| **Notebook 02** (Local Inference) | Colab + GPU + [Hugging Face token](https://huggingface.co/settings/tokens) (for Gemma 2) |
| **Notebook 03** (Full Pipeline) | Colab + GCP account (billing) + NGC API key + NIM on GKE |

---

## Choose Your Path

```
┌─────────────────────────────────────────────────────────────────┐
│  New to the project?                                            │
│  → Start with Notebook 01 (Data Ingest pandas/cuDF/cudf.pandas) │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Want natural-language telemetry Q&A on your machine?           │
│  → Run Notebook 02 (Local Inference) with Gemma 2 2B            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Need production-scale inference in the cloud?                  │
│  → Deploy NIM on GKE, then run Notebook 03                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start (Notebooks 01 & 02) (~5-10 min)

**Prerequisites:** Google account (for Colab), GPU runtime (T4 is default; L4 works too).

1. Open [01 Data Ingest](https://colab.research.google.com/github/KarthikSriramGit/H.E.I.M.D.A.L.L/blob/main/notebooks/01_data_ingest_benchmark.ipynb) (~5 min) and [02 Inference Pipeline](https://colab.research.google.com/github/KarthikSriramGit/H.E.I.M.D.A.L.L/blob/main/notebooks/02_inference_pipeline.ipynb) (~10 min first run).
2. **Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save.** (L4 if available.)
3. **Notebook 02 only:** Add your Hugging Face token so the notebook can load Gemma 2:
   - Go to [huggingface.co/google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it) and accept the license.
   - Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (click **New token**, set role to **Read**).
   - In Colab: click the **key icon** in the left sidebar → **Add new secret** → name: `HF_TOKEN`, value: paste your token → **Save**.
4. Run all cells.

---

## Vertex AI Setup (Colab Enterprise + L4)

Run the data ingest notebook on **Vertex AI** with an **NVIDIA L4** GPU, following the same flow as the [Accelerated Data Analytics with GPUs](https://codelabs.developers.google.com/accelerated-analytics-with-gpus) Codelab.

### Step 1: Open Colab Enterprise

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. **Navigation menu** → **Vertex AI** → **Colab Enterprise**

### Step 2: Create a runtime template

1. Click **Runtime templates** → **New template**
2. Under **Runtime basics**:
   - **Display name:** `gpu-l4-template`
   - **Region:** Your preferred region (e.g. `us-central1`)
3. Under **Configure compute**:
   - **Machine type:** `g2-standard-4` (1× NVIDIA L4 GPU)
   - **Idle shutdown:** 60 minutes (or as desired)
4. Click **Create**

> **Note:** If you see `NVIDIA_L4_GPUS exceeded`, your L4 quota is used or too low. See [Troubleshooting: L4 quota](#l4-quota-exceeded) below. `g2-standard-8` (1× L4) and `g2-standard-16` (2× L4) are alternatives if you have quota.

### Step 3: Start a runtime

1. Click **Runtimes** → **Create**
2. Under **Runtime template**, select `gpu-l4-template`
3. Click **Create** and wait for the runtime to boot (a few minutes)

### Step 4: Import the notebook

1. Click **My notebooks** → **Import**
2. Select **URL** and paste:
   ```
   https://github.com/KarthikSriramGit/H.E.I.M.D.A.L.L/blob/main/notebooks/01_data_ingest_benchmark.ipynb
   ```
3. Click **Import**

### Step 5: Connect and run

1. Open the imported notebook
2. Click the **Connect** dropdown → **Connect to a Runtime**
3. Select your `gpu-l4-template` runtime → **Connect**
4. Run the setup cell (clone + pip install). cuDF will use the L4 GPU.
5. Run all cells. The benchmark will execute on the L4.

### cudf.pandas (optional)

To enable zero-code-change GPU acceleration for pandas:

```python
%load_ext cudf.pandas
import pandas as pd
```

**Reference:** [Accelerated Data Analytics with Google Cloud and NVIDIA](https://codelabs.developers.google.com/accelerated-analytics-with-gpus) Codelab

---

### Alternative: Regular Colab

For quick runs without GCP setup: open the [notebook in Colab](https://colab.research.google.com/github/KarthikSriramGit/H.E.I.M.D.A.L.L/blob/main/notebooks/01_data_ingest_benchmark.ipynb), then **Runtime → Change runtime type → GPU (L4)** when available (e.g. Colab Pro).

---

## Notebooks

| Notebook | What it does | Requirements |
|----------|--------------|--------------|
| [01 Data Ingest](https://colab.research.google.com/github/KarthikSriramGit/H.E.I.M.D.A.L.L/blob/main/notebooks/01_data_ingest_benchmark.ipynb) | cuDF + UVM loading, pandas vs cuDF vs cudf.pandas benchmark | GPU (L4) |
| [02 Inference Pipeline](https://colab.research.google.com/github/KarthikSriramGit/H.E.I.M.D.A.L.L/blob/main/notebooks/02_inference_pipeline.ipynb) | Format selection, Gemma 2 2B local inference | GPU + HF token |
| [03 Query Telemetry](https://colab.research.google.com/github/KarthikSriramGit/H.E.I.M.D.A.L.L/blob/main/notebooks/03_query_telemetry.ipynb) | Full pipeline with NIM (Llama 3 8B on GKE) | NIM deployed (see below) |

---

## Setup for Notebook 03 (NIM on GKE)

Notebook 03 requires NIM running on GKE. Follow these steps.

### Prerequisites

- Google account with [Google Cloud](https://console.cloud.google.com) (billing enabled)
- [NGC account](https://ngc.nvidia.com) for the API key

### Step 1: Create a GCP project (if needed)

1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Create a project or select an existing one
3. Enable billing for the project
4. Note your **Project ID** (e.g. `heimdall-487621`)

### Step 2: Enable APIs

1. Go to [APIs & Services → Library](https://console.cloud.google.com/apis/library)
2. Enable **Kubernetes Engine API** and **Compute Engine API**

### Step 3: Get your NGC API key

1. Go to [ngc.nvidia.com](https://ngc.nvidia.com) and sign in
2. Profile (top right) → **Setup** → **Generate API Key**
3. Copy the key (starts with `nvapi-`)

### Step 4: Deploy NIM on GKE

Open [Google Cloud Shell](https://shell.cloud.google.com) and run:

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

### Step 7: Cleanup (when finished)

To avoid ongoing GCP charges:

```bash
kubectl delete namespace nim
gcloud container clusters delete nim-demo --zone=$ZONE --project=$PROJECT_ID --quiet
```

---

## Architecture

```
  Data Layer              Inference Layer                  Deployment Layer
  ----------------        ----------------                 ----------------
  Synthetic Generator --> cuDF + UVM Loader ----------->   Format Selector
                          Benchmark                        Inference Pipeline
                          (pandas, cuDF, cudf.pandas)      Metrics (p50, p90, TTFT)
                        Query Engine  <-- NIM Client  <--  NIM on GKE
```

---

## Results & Takeaways

Benchmark results from running the notebooks on Colab: L4 GPU for notebook 01, T4 GPU for notebook 02, and NIM (L4 on GKE) for notebook 03.

### 1. Data ingest: pandas vs cuDF vs cudf.pandas (2M rows)

The benchmark produces a single figure with **3 plots** (time comparison, memory comparison, cuDF speedup) and a **summary table**.

![pandas vs cuDF vs cudf.pandas benchmark](docs/assets/01_benchmark_pandas_vs_cudf.png)

*Run notebook 01 to regenerate this figure; it is saved to `docs/assets/01_benchmark_pandas_vs_cudf.png`.*

**Takeaway:** cuDF gives ~5× faster load and ~10–13× faster groupby/sort, with negligible host memory (data stays in GPU VRAM via UVM spill). **cudf.pandas** achieves similar GPU performance using the same pandas API—zero code change required. Use cuDF for explicit control, cudf.pandas to accelerate existing pandas code.

### 2. Local inference: Gemma 2 2B (notebook 02)

Natural-language answers to telemetry queries on a T4 GPU. Typical run: ~21s total for 5 queries (~4.2s avg latency).

![Local Gemma 2 2B inference](docs/assets/02_inference_local_gemma.png)

**Takeaway:** Local GGUF inference works for low-latency, offline telemetry Q&A. Good for prototyping; heavier workloads may need cloud scaling.

### 3. Full pipeline: NIM on GKE (notebook 03)

Same queries via Llama 3 8B on NVIDIA NIM (L4 on GKE). Longer answers, higher latency due to model size and network.

![NIM Llama 3 8B on GKE](docs/assets/03_query_nim.png)

**Takeaway:** NIM on GKE scales inference for production. Use notebook 02 for fast local iteration and notebook 03 for production-style deployment.

---

## Troubleshooting

### Notebook 01: Data Ingest

| Issue | Solution |
|-------|----------|
| **L4 quota exceeded** | See [L4 quota exceeded](#l4-quota-exceeded) below. |
| **cuDF fails to import** | Ensure you selected a **GPU** runtime (L4 or T4). cuDF requires a GPU. |
| **"No GPU" or cuDF falls back to CPU** | Runtime → Change runtime type → GPU (L4 or T4) → Save, then restart runtime. |
| **Out of memory** | Reduce row count in the generate cell (e.g. `ROWS = 500_000` instead of `2_000_000`). |

#### L4 quota exceeded

If you see `Quota 'NVIDIA_L4_GPUS' exceeded. Limit: 1.0 in region us-central1`:

1. **Free up quota:** Stop or delete any existing Colab Enterprise runtimes or other L4 instances in that region.
2. **Request more L4 quota (e.g. Iowa / us-central1):**
   - Go to [IAM & Admin → Quotas](https://console.cloud.google.com/iam-admin/quotas)
   - In the filter box, enter `NVIDIA_L4_GPUS`
   - Select your project and **Location: us-central1** (Iowa)
   - Check the box next to **NVIDIA L4 GPUs**
   - Click **Edit quotas** (top of page)
   - Enter the new limit (e.g. `2` if you want one more)
   - Add a short justification (e.g. "Need additional L4 for Colab Enterprise data analytics workloads")
   - Click **Submit**. Google typically reviews instantly.
3. **Try another region:** Create the runtime template in a different region (e.g. `us-east1`, `europe-west1`) where you may have quota. See [L4 availability](https://cloud.google.com/compute/docs/gpus/regions-zones).
4. **Use T4 instead:** Create a template with machine type `n1-standard-4` and **Attach GPU: NVIDIA Tesla T4** (1). T4 works with cuDF; the notebook runs unchanged. Then import and connect as usual.

### Notebook 02: Inference Pipeline

| Issue | Solution |
|-------|----------|
| **"HF_TOKEN not set" or 401** | Add `HF_TOKEN` in Colab Secrets. Accept the [Gemma 2 license](https://huggingface.co/google/gemma-2-2b-it) first. |
| **Out of memory (OOM)** | Use a GPU runtime (T4 or better). Gemma 2 2B needs ~4–6 GB VRAM. |
| **Model download is slow** | First run downloads ~5 GB. Use a stable connection; subsequent runs use cache. |

### Notebook 03: NIM Connection

| Issue | Solution |
|-------|----------|
| **ConnectionRefusedError** | NIM may not be ready or not reachable. See steps below. |
| **EXTERNAL-IP stuck on &lt;pending&gt;** | Wait 2–5 min. If it stays pending, check [GCP quotas](https://console.cloud.google.com/iam-admin/quotas) for forwarding rules. |
| **Pod not Running** | First deployment can take 10–20 min for model download. Run `kubectl get pods -n nim -w` to watch. |

**If you get `ConnectionRefusedError` when calling NIM from Colab:**

1. **Check the NIM pod is Running**
   ```bash
   kubectl get pods -n nim
   ```
   Wait until `STATUS` is `Running` and `READY` is `1/1`.

2. **Test from Cloud Shell (port-forward)**
   ```bash
   kubectl port-forward svc/my-nim-nim-llm 8000:8000 -n nim &
   sleep 5
   curl -s http://localhost:8000/v1/models
   kill %1
   ```
   If this works, NIM is fine; the problem is external access.

3. **Check service endpoints**
   ```bash
   kubectl get endpoints -n nim
   ```
   `my-nim-nim-llm` should have at least one address. If it shows `<none>`, the pod is not ready.

4. **Verify LoadBalancer backends**  
   In [GCP Console → Network Services → Load balancing](https://console.cloud.google.com/net-services/loadbalancing/list/loadBalancers), open the load balancer for the NIM service and confirm the backends are healthy.

5. **Allow firewall (if needed)**
   ```bash
   gcloud compute firewall-rules create allow-nim-8000 \
     --allow tcp:8000 \
     --source-ranges 0.0.0.0/0 \
     --target-tags $(kubectl get nodes -o jsonpath='{.items[0].metadata.labels.cloud\.google\.com\/gke-nodepool}' 2>/dev/null || echo "gke-nim-demo")
   ```

### General

| Issue | Solution |
|-------|----------|
| **Colab disconnects or session dies** | Notebooks auto-clone the repo; re-run the setup cell. For long runs, consider Colab Pro for longer sessions. |
| **Image or asset not loading** | Ensure you cloned the full repo and that `docs/` is present. |

---

## Real Data

See [data/README_data_sources.md](data/README_data_sources.md) for a plan to gather real telemetry from nuScenes, KITTI, CARLA, ROS2 bags, and OBD-II CAN data.

---

## Inspiration

1. [Deploy Faster Generative AI models with NVIDIA NIM on GKE](https://developers.google.com/learn/pathways/deploy-faster-gen-ai-models-nvidia-gke)
2. [Intro to Inference: How to Run AI Models on a GPU](https://developers.google.com/learn/pathways/ai-models-on-gpu-intro)
3. [Speed Up Data Analytics on GPUs](https://developers.google.com/learn/pathways/speed-up-data-analytics-GPUs)

---

## Contributing

Contributions are welcome. To keep the codebase safe, **direct pushes to `main` are not allowed**. All changes must go through pull requests (PRs).

### Safe contribution workflow

1. **Fork** the repository to your GitHub account.
2. **Clone** your fork locally.
3. **Create a branch** for your change: `git checkout -b feature/your-feature-name`
4. **Make your changes**, commit, and push to your fork.
5. **Open a pull request** from your fork’s branch to this repo’s `main`.
6. **Wait for review**; maintainers will review and merge after approval.

This workflow ensures every change is reviewed before it reaches `main`. Repository branch protection rules enforce this.

**Idea areas:** adapters for nuScenes/KITTI, ROS2 bag-to-Parquet scripts, NIM prompt templates, benchmark results on different GPU configurations.

---

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow.

- Be respectful and inclusive. Welcome newcomers and diverse perspectives.
- Focus on constructive feedback. Critique ideas, not people.
- No harassment, trolling, or discriminatory behavior.
- Help keep the community safe and productive for everyone.

Violations can be reported to the maintainers. We reserve the right to remove contributions or block users who do not follow these guidelines.

---

## License

Apache 2.0
