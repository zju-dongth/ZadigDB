# ZadigDB: Optimizing Semantic Queries over Multi-Camera Video

Reference implementation of **ZadigDB** for the paper *"ZadigDB: Optimizing Semantic Queries over Multi-Camera Video"*.

## Repository Structure

```text
.
├── main.py                    # Main query compilation and execution pipeline
├── load_data.py               # Camera topology, detection slice, and query loader
├── eval.py                    # Query accuracy and hardware telemetry evaluation
├── zadigdb/
│   ├── schema.py              # Camera topology, competitive provenance view, and plan DAG
│   ├── synthesizer.py         # SQL-extended parser and Intent-to-Plan Optimizer (IPO)
│   ├── admission_gate.py      # Clause-Level Admission Gate (Clauses C1, C2, C3)
│   ├── buffer_pool.py         # Multi-granularity visual token buffer pool and cache
│   ├── algebra.py             # Multi-camera operators (SEM_FILTER, SEM_LINK, SEM_COMOVE_JOIN)
│   ├── vlm_backend.py         # 4-bit NF4 Qwen2.5-VL-7B-Instruct visual-KV slot backend
│   └── pipeline.py            # End-to-end ZadigDBEngine query compiler and executor
└── configs/
    └── default_4090.yaml      # Hardware budget, kinematic bounds, and engine configuration
```

## Quick Start

1. Install dependencies (Python 3.10+):
   ```bash
   pip install -r requirements.txt
   ```

2. Set dataset and model paths in [`configs/default_4090.yaml`](configs/default_4090.yaml).

3. Run query execution:
   ```bash
   python3 main.py --cfg_file configs/default_4090.yaml --mode warm
   python3 main.py --cfg_file configs/default_4090.yaml --mode warm --allow-vlm
   ```
