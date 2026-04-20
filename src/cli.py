from __future__ import annotations

import argparse

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.classification.paper_classifier import PaperClassifier
from src.gaps.gap_detector import GapDetector
from src.retrieval.arxiv_client import ArxivClient
from src.retrieval.semantic_scholar_client import SemanticScholarClient
from src.synthesis.review_synthesizer import ReviewSynthesizer

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(prog="scoutlit", description="Academic Literature Assistant")
    parser.add_argument("topic", help="Research topic or hypothesis")
    parser.add_argument("--max-results", type=int, default=20, help="Papers to retrieve per source")
    parser.add_argument("--no-gaps", action="store_true", help="Skip research gap detection")
    args = parser.parse_args()

    console.rule("[bold blue]ScoutLit — Academic Literature Assistant")

    with console.status("Retrieving papers from arXiv..."):
        arxiv_papers = ArxivClient(max_results=args.max_results).search(args.topic)

    with console.status("Retrieving papers from Semantic Scholar..."):
        ss_papers = SemanticScholarClient().search(args.topic, limit=args.max_results)

    all_papers = arxiv_papers + ss_papers
    console.print(f"[green]Retrieved {len(all_papers)} papers.[/green]")

    with console.status("Classifying papers..."):
        classified = PaperClassifier().classify_batch(all_papers, args.topic)

    table = Table(title="Classified Papers", show_lines=True)
    table.add_column("Title", max_width=60)
    table.add_column("Year", width=6)
    table.add_column("Category", width=14)
    for cp in classified:
        table.add_row(cp.paper.title[:60], cp.paper.published[:4], cp.category)
    console.print(table)

    with console.status("Synthesising literature review..."):
        review = ReviewSynthesizer().synthesize(args.topic, classified)

    console.print(Panel(review.summary, title="Literature Review", border_style="blue"))

    if not args.no_gaps:
        with console.status("Detecting research gaps..."):
            gaps = GapDetector().detect(args.topic, classified)
        console.rule("[bold yellow]Research Gaps")
        for i, gap in enumerate(gaps, 1):
            console.print(f"[bold]{i}. {gap.description}[/bold]")
            console.print(f"   Suggested question: {gap.suggested_question}\n")


if __name__ == "__main__":
    main()
