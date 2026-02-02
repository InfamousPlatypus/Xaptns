from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel
from xaptns.ingestion import fetch_arxiv_data
from xaptns.model import Embedder
from xaptns.search import VectorIndex
from xaptns.navigator import Navigator
from xaptns.cartographer import Cartographer
from xaptns.concepts import ConceptMapper
import numpy as np

app = FastAPI(title="Xaptns API", description="High-performance engine for navigating scientific literature.")

# Global state
embedder = None
vindex = None
nav = None
carto = None
mapper = None

@app.on_event("startup")
async def startup_event():
    global embedder, vindex, nav, carto, mapper
    embedder = Embedder()
    vindex = VectorIndex()
    nav = Navigator(vindex)
    carto = Cartographer(vindex)
    mapper = ConceptMapper()

class PaperMetadata(BaseModel):
    id: str
    title: str
    abstract: str
    distance: Optional[float] = None

class SearchResponse(BaseModel):
    seed_id: str
    results: List[PaperMetadata]

@app.get("/search", response_model=SearchResponse)
async def search(id: str, limit: int = 10):
    """Find similar papers to a given arXiv ID."""
    paper = fetch_arxiv_data(id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    text = f"{paper['title']} {paper['abstract']}"
    vec = embedder.embed(text)

    # For MVP, we might want to add this paper to the index if it's not there
    vindex.add(id, vec, {"title": paper['title'], "abstract": paper['abstract']})

    matches = vindex.search(vec, limit=limit)

    results = []
    for m in matches:
        results.append(PaperMetadata(
            id=m['arxiv_id'],
            title=m['metadata'].get('title', 'Unknown'),
            abstract=m['metadata'].get('abstract', ''),
            distance=m['distance']
        ))

    return SearchResponse(seed_id=id, results=results)

@app.get("/hardware")
async def get_hardware():
    """Returns information about the detected acceleration hardware."""
    return {
        "device": embedder.device,
        "acceleration": "OpenVINO/ONNX" if embedder.ort_session or embedder.ov_compiled_model else "CPU"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
