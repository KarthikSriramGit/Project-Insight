# Project Insight: Telemetry-to-Insight Pipeline

A pipeline for robotics and autonomous systems that turns fleet telemetry into natural-language insights. Combines GPU-accelerated data loading (cuDF + UVM), NVIDIA NIM on GKE for LLM inference, and model format selection for production deployment.

## Inspiration

This project was inspired by three courses from Google for Developers and NVIDIA:

1. [Deploy Faster Generative AI models with NVIDIA NIM on GKE](https://developers.google.com/learn/pathways/deploy-faster-gen-ai-models-nvidia-gke): Learned to deploy pre-built LLM inference on GKE with Helm and a GPU node pool. No custom inference code, just config and an API. NIM is the default when a chat/completion API is needed in the cloud.

2. [Intro to Inference: How to Run AI Models on a GPU](https://developers.google.com/learn/pathways/ai-models-on-gpu-intro): Learned that model format determines where a model can run. Safetensors for sharing, GGUF for local/quantized, TensorRT for production on NVIDIA GPUs, ONNX for portability. Format is chosen by target (local vs cloud, CPU vs GPU) from the start.

3. [Speed Up Data Analytics on GPUs](https://developers.google.com/learn/pathways/speed-up-data-analytics-GPUs): Learned that cuDF with Unified Virtual Memory lets you use GPU speed and spill to CPU RAM, avoiding pandas OOM on large CSVs. With cudf.pandas, existing pandas code can get GPU acceleration.

## Architecture

```
Data Layer               Inference Layer               Deployment Layer 
Synthetic Generator  -->  cuDF + UVM Loader  -->       Format Selector
                         Benchmark (pandas vs cuDF)    Inference Pipeline
                                                       Metrics (p50, p90, TTFT)
                         Query Engine  <--  NIM Client  <--  NIM on GKE
```

The pipeline ingests large ROS2/DRIVE-style telemetry with cuDF and UVM, serves natural-language queries via NIM on GKE, and applies the right inference format for production.

## Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU (optional, for cuDF; pipeline falls back to pandas without it)
- Google Cloud project with billing (for GKE/NIM deployment)

### Install

```bash
pip install -r requirements.txt
```

For GPU acceleration:

```bash
pip install cudf-cu12
```

### Generate Synthetic Data

```bash
python data/synthetic/generate_telemetry.py --rows 5000000 --output-dir data/synthetic --format parquet
```

### Deploy NIM on GKE

1. Set environment variables: `PROJECT_ID`, `ZONE`, `NGC_CLI_API_KEY`
2. Run `src/deploy/gke/cluster_setup.sh`
3. Run `src/deploy/gke/deploy_nim.sh`
4. Port-forward: `kubectl port-forward service/my-nim-nim-llm 8000:8000 -n nim`

## Usage

### Data Ingest and Benchmark

```python
from src.ingest.cudf_loader import load_telemetry
df = load_telemetry("data/synthetic/fleet_telemetry.parquet", spill=True)
```

### Query Telemetry with NIM

```python
from src.query.engine import TelemetryQueryEngine
engine = TelemetryQueryEngine("data/synthetic/fleet_telemetry.parquet", nim_base_url="http://localhost:8000")
answer = engine.query("What is the max brake pressure across all vehicles?", sensor_type="can")
```

### Format Selection

```python
from src.inference.format_selector import select_format
fmt, rationale = select_format("production", hardware="gpu")
# Returns: ("tensorrt", "Compiled engine for NVIDIA GPUs...")
```

## Real Data

See [data/README_data_sources.md](data/README_data_sources.md) for a plan to gather real telemetry from nuScenes, KITTI, CARLA, ROS2 bags, and OBD-II CAN data.

## Contributing

Contributions are welcome. If you have completed the Google for Developers and NVIDIA courses linked above, your experience with NIM on GKE, inference formats, or cuDF is especially valuable. Areas to contribute:

- Adapters for nuScenes, KITTI, or other datasets
- ROS2 bag-to-Parquet conversion scripts
- NIM prompt templates for telemetry-specific queries
- Benchmark results on different GPU configurations

## License

Apache 2.0
