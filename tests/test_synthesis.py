from unittest.mock import MagicMock, patch

from src.classification.paper_classifier import ClassifiedPaper, PaperCategory
from src.retrieval.arxiv_client import Paper
from src.synthesis.review_synthesizer import LiteratureReview, ReviewSynthesizer


def _classified(title: str, year: str, category: PaperCategory) -> ClassifiedPaper:
    return ClassifiedPaper(
        paper=Paper(
            id=f"id-{year}",
            title=title,
            authors=["Smith et al."],
            abstract="Abstract text.",
            published=f"{year}-01-01",
            url=f"https://arxiv.org/abs/{year}",
        ),
        category=category,
        rationale="Important work.",
    )


def test_synthesize_returns_review():
    review_text = (
        "Background: ...\nKey Contributions: ...\nRecent Advances: ...\nResearch Gaps:\n- Gap 1"
    )

    with patch("anthropic.Anthropic") as MockAnthropic:
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.get_final_text.return_value = review_text
        MockAnthropic.return_value.messages.stream.return_value = mock_stream

        papers = [
            _classified("Seminal Work", "2010", PaperCategory.SEMINAL),
            _classified("Recent Work", "2024", PaperCategory.RECENT),
        ]
        result = ReviewSynthesizer().synthesize("transformers", papers)

    assert isinstance(result, LiteratureReview)
    assert result.topic == "transformers"
    assert "Background" in result.sections
