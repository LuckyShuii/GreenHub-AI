"""Unit tests for Pydantic schema validation rules."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas import QdrantPayload, QdrantPoint, WasteItem


class TestWasteItem:
    """Validation behaviour of the WasteItem model."""

    def test_accepts_valid_fields(self) -> None:
        """A well-formed item is accepted and preserves its values."""
        item = WasteItem(nom="carton", region="bretagne")
        assert item.nom == "carton"
        assert item.region == "bretagne"

    @pytest.mark.parametrize("field", ["nom", "region"])
    def test_rejects_empty_strings(self, field: str) -> None:
        """Empty strings violate the min_length constraint.

        Args:
            field: Name of the field set to an empty value.
        """
        values = {"nom": "carton", "region": "bretagne", field: ""}
        with pytest.raises(ValidationError):
            WasteItem(**values)

    @pytest.mark.parametrize("field", ["nom", "region"])
    def test_rejects_missing_fields(self, field: str) -> None:
        """Omitting a required field raises a validation error.

        Args:
            field: Name of the omitted field.
        """
        values = {"nom": "carton", "region": "bretagne"}
        values.pop(field)
        with pytest.raises(ValidationError):
            WasteItem(**values)


class TestQdrantPayload:
    """Validation behaviour of the QdrantPayload model."""

    def test_accepts_valid_payload(self) -> None:
        """All three metadata fields are stored verbatim."""
        payload = QdrantPayload(
            nom="verre", region="occitanie", poubelle="verte"
        )
        assert payload.model_dump() == {
            "nom": "verre",
            "region": "occitanie",
            "poubelle": "verte",
        }

    @pytest.mark.parametrize("field", ["nom", "region", "poubelle"])
    def test_rejects_empty_strings(self, field: str) -> None:
        """Every field enforces a non-empty value.

        Args:
            field: Name of the field set to an empty value.
        """
        values = {
            "nom": "verre",
            "region": "occitanie",
            "poubelle": "verte",
            field: "",
        }
        with pytest.raises(ValidationError):
            QdrantPayload(**values)


class TestQdrantPoint:
    """Validation behaviour of the QdrantPoint model."""

    def test_accepts_valid_point(
        self, sample_vector: list[float], sample_payload: QdrantPayload
    ) -> None:
        """A point with a non-negative id and vector is accepted.

        Args:
            sample_vector: Embedding vector fixture.
            sample_payload: Metadata fixture.
        """
        point = QdrantPoint(
            point_id=7, vector=sample_vector, payload=sample_payload
        )
        assert point.point_id == 7
        assert len(point.vector) == len(sample_vector)

    def test_rejects_negative_identifier(
        self, sample_vector: list[float], sample_payload: QdrantPayload
    ) -> None:
        """Negative identifiers violate the ge=0 constraint.

        Args:
            sample_vector: Embedding vector fixture.
            sample_payload: Metadata fixture.
        """
        with pytest.raises(ValidationError):
            QdrantPoint(
                point_id=-1, vector=sample_vector, payload=sample_payload
            )

    def test_rejects_non_numeric_vector(
        self, sample_payload: QdrantPayload
    ) -> None:
        """Non-coercible vector entries raise a validation error.

        Args:
            sample_payload: Metadata fixture.
        """
        with pytest.raises(ValidationError):
            QdrantPoint(
                point_id=0,
                vector=["not-a-float"],  # type: ignore
                payload=sample_payload,
            )
