# Xaptns (χάρτης)

Xaptns (pronounced Captain) is a high-performance engine designed to navigate the n-dimensional seas of scientific literature. By synthesizing citation-informed embeddings with structural graph analysis, Xaptns allows researchers to map the intellectual lineage of any arXiv paper, discover hidden cross-disciplinary links, and identify foundational "dependency nodes".

## Core Mission: Structural Over Generative
In an era dominated by Large Language Models, Xaptns takes a deterministic and structural approach. We map the actual vector-space proximity and citation dependencies of the scientific record, ensuring links are based on real-world citations and mathematical foundations.

## Key Features
- **n-Dimensional Semantic Clustering**: Maps papers into a 768-dimensional space using SPECTER 2.0.
- **Hardware Acceleration**: Targets Intel CPUs, iGPUs, NPUs, and the Neural Compute Stick 2 (NCS2) via OpenVINO and ONNX Runtime.
- **Cargo Crane Ingestion**: Bulk processing of arXiv datasets with OpenAlex and Semantic Scholar enrichment.
- **Efficient Indexing**: USearch-powered vector search with binary quantization for 32x memory savings.
- **Hybrid Storage**: SQLite + sqlite-vec for reliable metadata and vector persistence.
- **N-Way Centroids**: Calculate the synthetic interest vector $V_C$ to find the thematic center of multiple topics.
- **Bridge Discovery**: Identify papers that link disparate research clusters using Betweenness Centrality.
- **Void Analysis**: Use Persistent Homology (TDA) to detect topological gaps in the arXiv corpus.
- **SAE Concept Mapping**: Disentangle high-dimensional vectors into human-readable concepts using Sparse Autoencoders.

## Installation

### Prerequisites
- Python 3.10+
- OpenVINO supported hardware (optional but recommended for speed)

### Setup
```bash
git clone <repo-url>
cd xaptns
pip install -e .
```

## Usage

The primary interface is the `xaptns` CLI.

### Quick Start (20 Minutes)
1. **Install**: `pip install -e .`
2. **Ingest & Search**:
   ```bash
   xaptns search --id 2301.10140 --limit 5
   ```
3. **Find Centroid**:
   ```bash
   xaptns centroid --ids 2301.10140,2212.11223
   ```
4. **Detect Voids**:
   ```bash
   xaptns voids --ids 2301.10140,2212.11223,2305.11111,2306.22222,2307.33333
   ```

### Search for Similar Papers
Find the top 5 most similar papers to a given arXiv ID and list the top 3 common citations among them:

```bash
xaptns search --id 2301.10140 --limit 5 --rank-citations 3
```

**Options:**
- `--id`: (Required) The arXiv ID of the seed paper (e.g., `2301.10140`).
- `--limit`: Number of semantically similar papers to retrieve (default: 10).
- `--rank-citations`: Number of top foundational papers (common citations) to display (default: 3).

## Hardware Support & Performance

Xaptns is optimized for limited hardware:
- **Intel Graphics/NCS2**: Automatically detected and used via OpenVINO.
- **AMD GPUs**: Supported via OpenVINO's OpenCL backend.
- **NPU/Laptop Acceleration**: Targets modern NPUs via ONNX Runtime OpenVINO execution provider.
- **CPU Fallback**: Uses optimized PyTorch CPU kernels if no specialized hardware is found.

### Hardware Optimization Guide
To force a specific device, set the `OPENVINO_DEVICE` environment variable:
```bash
export OPENVINO_DEVICE=MYRIAD  # For Intel NCS2
export OPENVINO_DEVICE=GPU     # For Integrated/Discrete GPU
export OPENVINO_DEVICE=NPU     # For modern AI chips
```

## Mathematical Appendices

### Synthetic Interest Vector ($V_C$)
$V_C = \frac{1}{N} \sum_{i=1}^N v_i$

### Bridge Papers
Calculated using Betweenness Centrality on a semantic similarity graph $G=(V,E)$ where $E = \{ (u,v) \mid \text{cos\_sim}(u,v) > 0.7 \}$.

### Void Detection
Uses Ripser for persistent homology to find $H_1$ and $H_2$ features. Gap coordinates are identified using Maximin sampling within the bounding box of the research cluster.

### First Run Note
The first time you run a search, Xaptns will:
1. Download the SPECTER 2.0 model (~400MB).
2. Convert the model to OpenVINO IR format.
3. Cache the converted model in `.model_cache/`.

This one-time process may take 1-2 minutes on slower hardware. Subsequent runs will use the cache and start significantly faster.

## Troubleshooting

Xaptns follows a "fail visibly" principle. If a network error occurs or a hardware backend fails, the full traceback and error message will be displayed to help diagnose the issue.

### Common Issues
- **Connection Timeouts**: Check your internet connection; Xaptns relies on the arXiv and Semantic Scholar APIs.
- **NaN Outputs**: If you see `nan` in distances, ensure your OpenVINO drivers are up to date. The system will attempt to fallback to CPU if instability is detected.

## License
Xaptns is released under the MIT License.
