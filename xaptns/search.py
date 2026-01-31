from usearch.index import Index
import numpy as np
import sys

class VectorIndex:
    def __init__(self, dim=768):
        self.dim = dim
        # Using cosine metric for semantic similarity
        self.index = Index(ndim=dim, metric='cos')
        self.id_to_metadata = {}

    def add(self, arxiv_id, vector, metadata=None):
        """
        Adds a vector to the index.
        arxiv_id: The string ID of the paper.
        vector: 1D numpy array of size dim.
        metadata: Optional dictionary of paper metadata.
        """
        # usearch needs integer keys
        numeric_id = len(self.id_to_metadata)
        self.index.add(numeric_id, vector.flatten())
        self.id_to_metadata[numeric_id] = {
            "arxiv_id": arxiv_id,
            "metadata": metadata
        }

    def search(self, vector, limit=10):
        """
        Searches for the nearest neighbors of the given vector.
        returns: List of dicts with arxiv_id, metadata, and distance.
        """
        if len(self.id_to_metadata) == 0:
            return []

        matches = self.index.search(vector.flatten(), limit)
        results = []
        for match in matches:
            meta = self.id_to_metadata.get(match.key, {})
            results.append({
                "arxiv_id": meta.get("arxiv_id"),
                "metadata": meta.get("metadata"),
                "distance": match.distance
            })
        return results

if __name__ == "__main__":
    vindex = VectorIndex(dim=4)
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    vindex.add("paper1", v1, {"title": "Paper 1"})
    vindex.add("paper2", v2, {"title": "Paper 2"})
    vindex.add("paper3", v3, {"title": "Paper 3"})

    print("Searching for vector similar to paper1...")
    results = vindex.search(v1, limit=2)
    for res in results:
        print(f"Found {res['arxiv_id']} ({res['metadata']['title']}) with distance {res['distance']:.4f}")
