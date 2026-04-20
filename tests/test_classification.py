from unittest.mock import MagicMock, patch

from src.classification.paper_classifier import ClassifiedPaper, PaperCategory, PaperClassifier
from src.retrieval.arxiv_client import Paper


def _sample_paper() -> Paper:
    return Paper(
        id="arxiv:2401.00001",
        title="Attention Is All You Need",
        authors=["Vaswani et al."],
        abstract="We propose a new architecture...",
        published="2017-06-12T00:00:00",
        url="https://arxiv.org/abs/1706.03762",
    )


def test_classify_seminal():
    response_text = '{"category": "seminal", "rationale": "Introduced the Transformer."}'
    mock_content = MagicMock()
    mock_content.text = response_text

    with patch("anthropic.Anthropic") as MockAnthropic:
        mock_client = MockAnthropic.return_value
        mock_client.messages.create.return_value.content = [mock_content]

        result = PaperClassifier().classify(_sample_paper(), "transformer architectures")

    assert isinstance(result, ClassifiedPaper)
    assert result.category == PaperCategory.SEMINAL
    assert "Transformer" in result.rationale
