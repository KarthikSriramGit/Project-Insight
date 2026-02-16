# Project Insight: Telemetry-to-Insight Pipeline

A pipeline for robotics and autonomous systems that turns fleet telemetry into natural-language insights. Combines GPU-accelerated data loading (cuDF + UVM), NVIDIA NIM on GKE for LLM inference, and model format selection for production deployment.

## Run on Colab

| Notebook | Description |
|----------|-------------|
| [01 Data Ingest](https://colab.research.google.com/github/KarthikSriramGit/Project-Insight/blob/main/notebooks/01_data_ingest_benchmark.ipynb) | cuDF + UVM loading, pandas vs cuDF benchmark |
| [02 Inference Pipeline](https://colab.research.google.com/github/KarthikSriramGit/Project-Insight/blob/main/notebooks/02_inference_pipeline.ipynb) | Format selection, TinyLlama inference |
| [03 Query Telemetry](https://colab.research.google.com/github/KarthikSriramGit/Project-Insight/blob/main/notebooks/03_query_telemetry.ipynb) | Full pipeline: retrieve data, NIM summarization |

Open any notebook and run all cells. Use **Runtime > Change runtime type > GPU** for faster execution.

## Inspiration

This project was inspired by three courses from Google for Developers and NVIDIA:

1. [Deploy Faster Generative AI models with NVIDIA NIM on GKE](https://developers.google.com/learn/pathways/deploy-faster-gen-ai-models-nvidia-gke): Deploy pre-built LLM inference on GKE with Helm and a GPU node pool.

2. [Intro to Inference: How to Run AI Models on a GPU](https://developers.google.com/learn/pathways/ai-models-on-gpu-intro): Model format selection (Safetensors, GGUF, TensorRT, ONNX) for different targets.

3. [Speed Up Data Analytics on GPUs](https://developers.google.com/learn/pathways/speed-up-data-analytics-GPUs): cuDF with Unified Virtual Memory for GPU-accelerated analytics.

## Architecture

```
Data Layer               Inference Layer               Deployment Layer
Synthetic Generator  -->  cuDF + UVM Loader  -->       Format Selector
                         Benchmark (pandas vs cuDF)    Inference Pipeline
                                                       Metrics (p50, p90, TTFT)
                         Query Engine  <--  NIM Client  <--  NIM on GKE
```

## Deploy NIM on GKE (for notebook 03)

Use [Google Cloud Shell](https://shell.cloud.google.com) (browser-based Linux). From the project directory:

1. Get your NGC API key from [ngc.nvidia.com](https://ngc.nvidia.com) → Profile → Setup → Generate API Key.

2. Set environment variables and run the deploy scripts:

```bash
export PROJECT_ID="your-gcp-project-id"
export ZONE="us-central1-a"
export REGION="us-central1"
export NGC_CLI_API_KEY="your-ngc-api-key"

./src/deploy/gke/cluster_setup.sh
gcloud container clusters get-credentials nim-demo --zone=$ZONE
./src/deploy/gke/deploy_nim.sh
```

3. Expose NIM with a LoadBalancer so Colab can reach it:

```bash
kubectl expose deployment my-nim-nim-llm -n nim --type=LoadBalancer --port=8000
kubectl get svc -n nim
```

4. Set `NIM_BASE_URL` in notebook 03 to your service external IP (e.g. `http://34.x.x.x:8000`).

## Real Data

See [data/README_data_sources.md](data/README_data_sources.md) for a plan to gather real telemetry from nuScenes, KITTI, CARLA, ROS2 bags, and OBD-II CAN data.

## Contributing

Contributions are welcome. Areas to contribute: adapters for nuScenes/KITTI, ROS2 bag-to-Parquet scripts, NIM prompt templates, benchmark results on different GPU configurations.

## License

Apache 2.0
