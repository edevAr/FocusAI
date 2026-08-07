from __future__ import annotations

from frontend.app import status_messages


def test_status_messages_show_readiness_identity_warnings_and_checklist() -> None:
    messages = status_messages(
        {
            "readiness": {"status": "degraded", "causes": {"model": "alias missing"}},
            "production": {"name": "productivity_ensemble", "alias": "production", "version": "7"},
            "quality": {"warnings": ["Accuracy is below threshold."]},
            "checklist": {"production_alias": True, "quality_gates": False},
        }
    )

    assert "degraded" in messages["readiness"]
    assert "productivity_ensemble@production (version 7)" == messages["production"]
    assert messages["warnings"] == ["Accuracy is below threshold."]
    assert messages["checklist"] == [
        "✅ Production alias observed",
        "⚠️ Quality gates passed",
    ]
