import arxiv
import requests
import sys

def fetch_arxiv_data(arxiv_id):
    """
    Fetches paper metadata from arXiv using its ID.
    """
    try:
        # Normalize ID (remove version if present)
        clean_id = arxiv_id.split('v')[0]
        client = arxiv.Client()
        search = arxiv.Search(id_list=[clean_id])
        results = list(client.results(search))
        if not results:
            print(f"Error: No paper found with ID {arxiv_id}", file=sys.stderr)
            return None
        paper = results[0]
        return {
            "id": paper.entry_id.split("/")[-1],
            "title": paper.title,
            "abstract": paper.summary,
            "authors": [author.name for author in paper.authors],
            "url": paper.entry_id
        }
    except Exception as e:
        print(f"Error fetching from arXiv: {e}", file=sys.stderr)
        return None

def fetch_citations(arxiv_id):
    """
    Fetches references for the given arXiv ID using Semantic Scholar API.
    """
    try:
        # Semantic Scholar uses 'arXiv:<id>' for arXiv papers
        url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}?fields=references.title,references.paperId,references.externalIds"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("references", [])
        elif response.status_code == 404:
            print(f"Warning: Paper with ID {arxiv_id} not found in Semantic Scholar.", file=sys.stderr)
            return []
        else:
            print(f"Warning: Semantic Scholar API returned status {response.status_code} for ID {arxiv_id}", file=sys.stderr)
            return []
    except Exception as e:
        print(f"Error fetching citations: {e}", file=sys.stderr)
        return []

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        aid = sys.argv[1]
        print(f"Fetching data for {aid}...")
        data = fetch_arxiv_data(aid)
        if data:
            print(f"Title: {data['title']}")
            print(f"Abstract preview: {data['abstract'][:100]}...")
            refs = fetch_citations(aid)
            print(f"Found {len(refs)} references.")
            if refs:
                print(f"First reference: {refs[0].get('title', 'N/A')}")
