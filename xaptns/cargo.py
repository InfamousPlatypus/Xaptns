import json
import os
import sys
import requests
from typing import List, Dict, Any

class CargoCrane:
    """
    The Cargo Crane: Ingestion scripts for bulk arXiv data and Semantic Scholar/OpenAlex enrichment.
    """
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def process_kaggle_batch(self, file_path: str, limit: int = 100000):
        """
        Processes the arXiv Kaggle dataset (JSONL format).
        """
        if not os.path.exists(file_path):
            print(f"Error: {file_path} not found.", file=sys.stderr)
            return []

        papers = []
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                papers.append(json.loads(line))
        return papers

    def enrich_with_openalex(self, arxiv_id: str) -> Dict[str, Any]:
        """
        Enriches a paper with OpenAlex citation counts and reference lists.
        """
        # OpenAlex API endpoint for arXiv IDs
        url = f"https://api.openalex.org/works/https://arxiv.org/abs/{arxiv_id}"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    "citation_count": data.get("cited_by_count", 0),
                    "referenced_works": data.get("referenced_works", []),
                    "concepts": data.get("concepts", [])
                }
        except Exception as e:
            print(f"Error enriching with OpenAlex for {arxiv_id}: {e}", file=sys.stderr)
        return {}

if __name__ == "__main__":
    crane = CargoCrane()
    print("Cargo Crane initialized. Use process_kaggle_batch to ingest data.")
