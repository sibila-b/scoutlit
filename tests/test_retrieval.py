from unittest.mock import MagicMock, patch

from src.retrieval.arxiv_client import ArxivClient, Paper


def _make_arxiv_result(title: str = "Test Paper") -> MagicMock:
    r = MagicMock()
    r.entry_id = "https://arxiv.org/abs/2401.00001"
    r.title = title
    author = MagicMock()
    author.name = "Alice Smith"
    r.authors = [author]
    r.summary = "An abstract."
    r.published.isoformat.return_value = "2024-01-01T00:00:00"
    return r


def test_arxiv_client_returns_papers():
    with patch("arxiv.Client") as MockClient:
        instance = MockClient.return_value
        instance.results.return_value = [_make_arxiv_result("Attention Is All You Need")]
        client = ArxivClient(max_results=5)
        papers = client.search("transformers")

    assert len(papers) == 1
    assert isinstance(papers[0], Paper)
    assert papers[0].title == "Attention Is All You Need"
    assert papers[0].source == "arxiv"
