from unittest.mock import MagicMock, patch

from src.classification.paper_classifier import ClassifiedPaper, PaperCategory, PaperClassifier
from src.models.paper import PaperResult


def _sample_paper() -> PaperResult:
    return PaperResult(
        id="arxiv:2401.00001",
        title="Attention Is All You Need",
        authors=["Vaswani et al."],
        abstract="We propose a new architecture...",
        year="2017",
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
