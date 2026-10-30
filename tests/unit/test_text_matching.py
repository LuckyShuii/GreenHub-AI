"""Tests for French waste-name normalization and similarity scoring."""

from src.text_matching import normalize, score, tokenize


def test_normalize_removes_accents_and_normalizes_punctuation() -> None:
    """Accents and supported punctuation become comparable words."""
    assert normalize("  Épluchures légumes + fruits / thé—café  ") == "epluchures legumes fruits the—cafe"


def test_tokenize_removes_stopwords_and_normalizes_plurals() -> None:
    """French stopwords and light plural endings are normalized."""
    assert tokenize("Les bouteilles de métaux") == ("bouteille", "metal")


def test_similarity_is_bounded_and_rewards_typographical_variants() -> None:
    """Similarity is normalized and reasonably robust to misspellings."""
    assert 0.0 <= score("boutielle plastique", "Bouteille plastique") <= 1.0
    assert score("boutielle plastique", "Bouteille plastique") > 0.6
