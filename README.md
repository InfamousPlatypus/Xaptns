# Xaptns (χάρτης)

Xaptns (pronounced Captain) is a high-performance engine designed to navigate the n-dimensional seas of scientific literature. By synthesizing citation-informed embeddings with structural graph analysis, Xaptns allows researchers to map the intellectual lineage of any arXiv paper, discover hidden cross-disciplinary links, and identify foundational "dependency nodes".

## Core Mission: Structural Over Generative
In an era dominated by Large Language Models, Xaptns takes a deterministic and structural approach. We map the actual vector-space proximity and citation dependencies of the scientific record, ensuring links are based on real-world citations and mathematical foundations.

## Key Features
- **n-Dimensional Semantic Clustering**: Maps papers into a 768-dimensional space using SPECTER 2.0.
- **Hardware Acceleration**: Targets Intel Haswell-ULT iGPUs, Intel Neural Compute Stick 2 (NCS2), and AMD GPUs via OpenVINO.
- **Citation Ranking**: Identifies foundational papers by finding common citations among semantically similar results.
- **Model Caching**: Local caching of OpenVINO IR models to ensure fast startup on limited hardware.

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
- **CPU Fallback**: Uses optimized PyTorch CPU kernels if no specialized hardware is found.

### First Run Note
The first time you run a search, Xaptns will:
1. Download the SPECTER 2.0 model (~400MB).
2. Convert the model to OpenVINO IR format.
3. Cache the converted model in `.model_cache/`.

This one-time process may take 1-2 minutes on slower hardware. Subsequent runs will use the cache and start significantly faster.

## Data Freshness & Infrastructure

Xaptns is designed to be lightweight and "infrastructure-agnostic."

- **Model Updates**: We use the **SPECTER 2.0** model, which is a pre-trained foundation model provided by the Allen Institute for AI. Xaptns does not require the heavy infrastructure needed to train or fine-tune this model; we simply consume the optimized weights.
- **Live Discovery**: Instead of hosting a multi-terabyte static index of all 2M+ arXiv papers (which would require significant infrastructure), Xaptns uses a **Live Discovery** approach. It queries the arXiv and Semantic Scholar APIs in real-time to find candidate papers.
- **Auto-Refresh**: Data is "refreshed" automatically with every search. Since we pull metadata and recommendations directly from source APIs, you always see the latest preprints and citation counts without needing a manual update process.
- **Scalability**: For users with larger hardware, the underlying `USearch` engine supports memory-mapped indices on disk, allowing for the scaling of searches to millions of vectors if a local corpus is desired.

## Troubleshooting

Xaptns follows a "fail visibly" principle. If a network error occurs or a hardware backend fails, the full traceback and error message will be displayed to help diagnose the issue.

### Common Issues
- **Connection Timeouts**: Check your internet connection; Xaptns relies on the arXiv and Semantic Scholar APIs.
- **NaN Outputs**: If you see `nan` in distances, ensure your OpenVINO drivers are up to date. The system will attempt to fallback to CPU if instability is detected.

## License
Xaptns is released under the MIT License.
