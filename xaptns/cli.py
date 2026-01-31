import click
import sys
import requests
from collections import Counter
from xaptns.ingestion import fetch_arxiv_data, fetch_citations
from xaptns.model import Embedder
from xaptns.search import VectorIndex

@click.group()
def cli():
    """Xaptns: High-performance engine for navigating scientific literature."""
    pass

@cli.command()
@click.option('--id', required=True, help='arXiv ID of the seed paper.')
@click.option('--limit', default=10, help='Number of similar papers to find.')
@click.option('--rank-citations', 'rank_citations', default=3, help='Number of top common citations to list.')
def search(id, limit, rank_citations):
    """Find similar papers and rank common citations."""
    try:
        # 1. Fetch seed paper
        click.echo(f"[*] Fetching seed paper {id}...")
        seed_paper = fetch_arxiv_data(id)
        if not seed_paper:
            click.echo(f"Error: Could not find paper {id}", err=True)
            sys.exit(1)

        click.echo(f"[*] Seed Title: {seed_paper['title']}")

        # 2. Find candidate papers
        click.echo(f"[*] Discovering candidate papers related to {id}...")
        # Use Semantic Scholar Recommendations API
        rec_url = f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/arXiv:{id}?limit=50&fields=title,externalIds,abstract"
        resp = requests.get(rec_url, timeout=15)

        candidates = []
        if resp.status_code == 200:
            candidates = resp.json().get('recommendedPapers', [])
            click.echo(f"[*] Found {len(candidates)} recommended candidates.")
        else:
            click.echo(f"Warning: Recommendations API returned {resp.status_code}. Falling back to references.", err=True)
            refs = fetch_citations(id)
            # Reformat references to match candidate structure
            for r in refs:
                if 'citedPaper' in r:
                    candidates.append(r['citedPaper'])
                else:
                    candidates.append(r)

        if not candidates:
            click.echo("Error: No candidate papers found to compare.", err=True)
            sys.exit(1)

        # 3. Embed papers
        # Initialize embedder (will detect Intel/AMD hardware)
        embedder = Embedder()

        click.echo(f"[*] Embedding seed paper...")
        seed_text = f"{seed_paper['title']} {seed_paper['abstract']}"
        seed_vec = embedder.embed(seed_text)

        vindex = VectorIndex(dim=768)

        click.echo(f"[*] Embedding {len(candidates)} candidates...")
        for cand in candidates:
            # Try to get ArXiv ID, fallback to paperId
            ext_ids = cand.get('externalIds', {})
            cand_id = ext_ids.get('ArXiv') if ext_ids else None
            if not cand_id:
                cand_id = cand.get('paperId', 'unknown')

            title = cand.get('title')
            abstract = cand.get('abstract', '')
            if not title:
                continue

            # Use title and abstract for embedding if available
            cand_text = f"{title} {abstract}"
            cand_vec = embedder.embed(cand_text)

            vindex.add(cand_id, cand_vec, {"title": title, "paperId": cand.get('paperId')})

        # 4. Search
        click.echo(f"[*] Finding top {limit} similar papers in semantic space...")
        results = vindex.search(seed_vec, limit=limit)

        click.echo("\n" + "="*60)
        click.echo(f"{'Top Similar Papers':^60}")
        click.echo("="*60)
        for i, res in enumerate(results, 1):
            click.echo(f"{i:2d}. [{res['arxiv_id']:>12}] {res['metadata']['title'][:70]}")
            click.echo(f"    (Distance: {res['distance']:.4f})")

        # 5. Rank Citations
        if rank_citations > 0 and results:
            click.echo("\n" + "="*60)
            click.echo(f"{'Ranking Top ' + str(rank_citations) + ' Common Citations':^60}")
            click.echo("="*60)

            all_refs = []
            # To avoid excessive API calls in MVP, we only check the top 'limit' papers
            for res in results:
                p_id = res['metadata'].get('paperId')
                if not p_id:
                    continue

                # Fetch references for this paper
                ref_url = f"https://api.semanticscholar.org/graph/v1/paper/{p_id}/references?fields=title&limit=50"
                try:
                    r_resp = requests.get(ref_url, timeout=10)
                    if r_resp.status_code == 200:
                        json_data = r_resp.json()
                        ref_data = json_data.get('data')
                        if ref_data:
                            for r in ref_data:
                                cited = r.get('citedPaper')
                                if cited and cited.get('title'):
                                    all_refs.append(cited.get('title'))
                except Exception as e:
                    click.echo(f"Warning: Failed to fetch references for {p_id}: {e}", err=True)

            if all_refs:
                common = Counter(all_refs).most_common(rank_citations)
                for i, (title, count) in enumerate(common, 1):
                    click.echo(f"{i:2d}. {title}")
                    click.echo(f"    (Cited by {count} of the similar papers)")
            else:
                click.echo("No common citations found.")

    except Exception as e:
        click.echo(f"\nFATAL ERROR: {e}", err=True)
        # Visible error as requested
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    cli()
