from usearch.index import Index
import numpy as np
import sys
import sqlite3
import json
import os

class VectorIndex:
    def __init__(self, dim=768, db_path="xaptns.db"):
        self.dim = dim
        self.db_path = db_path
        # USearch with binary quantization (i8 or b1) for memory savings
        self.index = Index(ndim=dim, metric='cos', dtype='i8')
        self._init_db()

    def _init_db(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                arxiv_id TEXT UNIQUE,
                metadata TEXT,
                vector BLOB
            )
        ''')
        self.conn.commit()

    def add(self, arxiv_id, vector, metadata=None):
        """
        Adds a vector to the index and SQLite.
        """
        vector_flat = vector.flatten().astype(np.float32)

        # Add to SQLite
        meta_json = json.dumps(metadata) if metadata else "{}"
        try:
            self.cursor.execute(
                "INSERT OR REPLACE INTO papers (arxiv_id, metadata, vector) VALUES (?, ?, ?)",
                (arxiv_id, meta_json, vector_flat.tobytes())
            )
            # If it was a REPLACE, lastrowid might be the new ID or the old one depending on SQLite version
            # To be safe, let's get the ID of the inserted arxiv_id
            self.cursor.execute("SELECT id FROM papers WHERE arxiv_id = ?", (arxiv_id,))
            row_id = self.cursor.fetchone()[0]
            self.conn.commit()

            # Add to USearch (requires integer key)
            self.index.add(row_id, vector_flat)
        except Exception as e:
            print(f"Error adding to index: {e}", file=sys.stderr)

    def search(self, vector, limit=10):
        """
        Searches for the nearest neighbors.
        """
        vector_flat = vector.flatten().astype(np.float32)
        if len(self.index) == 0:
            return []

        matches = self.index.search(vector_flat, limit)
        results = []
        for match in matches:
            key = int(match.key)
            self.cursor.execute("SELECT arxiv_id, metadata FROM papers WHERE id = ?", (key,))
            row = self.cursor.fetchone()
            if row:
                results.append({
                    "arxiv_id": row[0],
                    "metadata": json.loads(row[1]),
                    "distance": float(match.distance)
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

    print(f"Index size: {len(vindex.index)}")

    print("Searching for vector similar to paper1...")
    results = vindex.search(v1, limit=2)
    print(f"Found {len(results)} results")
    for res in results:
        print(f"Found {res['arxiv_id']} ({res['metadata']['title']}) with distance {res['distance']:.4f}")
