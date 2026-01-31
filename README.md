Xaptns (χάρτης)
![Logo Placeholder: 8-bit cape with an X]

Xaptns (pronounced Captain) is a high-performance, non-LLM open-source engine designed to navigate the n-dimensional seas of scientific literature. By synthesizing citation-informed embeddings with structural graph analysis, Xaptns allows researchers to map the intellectual lineage of any arXiv paper, discover hidden cross-disciplinary links, and identify the foundational "dependency nodes" of any research cluster.

The name is a visual play on the word "Captain" while drawing its roots from the Greek χάρτης (khártēs), meaning map, chart, or sheet of paper.

Core Mission: Structural Over Generative
In an era dominated by Large Language Models, Xaptns takes a deterministic and structural approach. We don't "hallucinate" summaries; we map the actual vector-space proximity and citation dependencies of the scientific record. This ensures that when Xaptns identifies a link between quantum computing and organic chemistry, it is based on shared mathematical foundations and real-world citations, not linguistic probability.

Key Features
n-Dimensional Semantic Clustering: Maps papers into a 768-dimensional space where proximity is determined by thematic relevance, not just metadata tags.

Citation-Informed Discovery: Utilizes models like SPECTER 2.0  that use citation graphs as a supervision signal, ensuring that "similar" papers are those that truly belong in the same research conversation.

Heterogeneous Ingestion: Drop in an arXiv ID, a raw text abstract, or a PDF. Xaptns uses GROBID to extract structured bibliographies from unstructured files.

Citation Aggregator: Find the "most referenced" papers within a specific cluster of results to identify the foundational work that everyone in that niche is citing.

Custom Dependency Graphs: Build a "Papers of Interest" list to generate a chronological and logical map of research evolution.

The Engine Room (Architecture)
Xaptns is built as a modular pipeline for performance and transparency:

Ingestion Layer: Resolves arXiv IDs via the arXiv API  and parses PDFs using a GROBID Docker instance.

Embedding Layer: Vectorizes text using SPECTER 2.0, providing document-level representations that solve the "cold start" problem for uncited new preprints.

Search Layer: Employs USearch, a lightweight HNSW engine that is 10x faster than traditional vector search libraries for near-real-time clustering.

Graph Layer: Syncs with OpenAlex and Semantic Scholar to build out full citation and reference networks.

Getting Started
Prerequisites
Docker (for GROBID parsing)

Python 3.10+

Installation
Bash

pip install xaptns
docker pull lfoppiano/grobid:0.7.1
Basic Usage
Find the 10 most similar papers to a given arXiv ID and list the top 3 common citations among them:

Bash

xaptns search --id 2301.10140 --limit 10 --rank-citations 3
Roadmap
[ ] D3.js-powered interactive "Dependency Graph" visualizer.

[ ] Custom recommendation engine based on "Anchor-based traversal" of the semantic similarity graph.

[ ] Local library syncing with Zotero and BibTeX.

License
Xaptns is released under the MIT License.

Captain your own discovery. Map the uncharted.
