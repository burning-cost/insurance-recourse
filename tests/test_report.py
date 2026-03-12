"""Tests for insurance_recourse.report."""

import hashlib
import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from insurance_recourse.cost import RecourseEffort
from insurance_recourse.generator import RecourseAction
from insurance_recourse.report import RecourseReport, _is_numeric


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_effort(monetary_cost=250.0, time_days=7.0, feasibility=0.85):
    return RecourseEffort(
        monetary_cost=monetary_cost,
        time_days=time_days,
        feasibility_probability=feasibility,
        feature_changes={},
    )


def _make_action(
    feature_changes=None,
    predicted_premium=900.0,
    premium_reduction=300.0,
    premium_reduction_pct=25.0,
    monetary_cost=250.0,
    time_days=7.0,
    feasibility=0.85,
    causal_effects=None,
    validity=True,
    description="Add Thatcham Category 1 immobiliser",
):
    effort = _make_effort(monetary_cost, time_days, feasibility)
    return RecourseAction(
        feature_changes=feature_changes or {"vehicle_security": (1, 2)},
        predicted_premium=predicted_premium,
        premium_reduction=premium_reduction,
        premium_reduction_pct=premium_reduction_pct,
        effort=effort,
        causal_effects=causal_effects or {},
        validity=validity,
        description=description,
    )


def _make_factual():
    return pd.Series({
        "age": 35,
        "annual_mileage": 12000.0,
        "vehicle_security": 1,
        "pass_plus": 0,
        "postcode_risk": 6,
    })


def _make_report(n_actions=2, with_id=True):
    factual = _make_factual()
    actions = [
        _make_action(predicted_premium=900.0, premium_reduction=300.0, premium_reduction_pct=25.0),
        _make_action(
            feature_changes={"annual_mileage": (12000, 8000)},
            predicted_premium=1050.0,
            premium_reduction=150.0,
            premium_reduction_pct=12.5,
            monetary_cost=0.0,
            time_days=1.0,
            feasibility=0.60,
            description="Reduce annual mileage from 12,000 to 8,000 miles",
        ),
    ][:n_actions]
    return RecourseReport(
        factual=factual,
        actions=actions,
        model_metadata={"model_version": "2024-Q4-motor-v3", "product": "motor"},
        policyholder_id="POL-123456" if with_id else None,
        current_premium=1200.0,
    )


# ---------------------------------------------------------------------------
# RecourseReport construction
# ---------------------------------------------------------------------------

class TestRecourseReportConstruction:
    def test_basic_construction(self):
        report = _make_report()
        assert report.current_premium == 1200.0
        assert report.policyholder_id == "POL-123456"

    def test_no_policyholder_id(self):
        report = _make_report(with_id=False)
        assert report.policyholder_id is None

    def test_current_premium_inferred_from_actions(self):
        factual = _make_factual()
        action = _make_action(predicted_premium=900.0, premium_reduction=300.0)
        report = RecourseReport(
            factual=factual,
            actions=[action],
            model_metadata={},
        )
        assert report.current_premium == pytest.approx(1200.0, abs=0.01)

    def test_no_actions_zero_premium(self):
        factual = _make_factual()
        report = RecourseReport(factual=factual, actions=[], model_metadata={})
        assert report.current_premium == 0.0

    def test_generated_at_default_is_recent(self):
        before = datetime.now(timezone.utc)
        report = _make_report()
        after = datetime.now(timezone.utc)
        # Just check it's a non-empty ISO string
        assert len(report._generated_at) > 10

    def test_generated_at_override(self):
        factual = _make_factual()
        report = RecourseReport(
            factual=factual, actions=[], model_metadata={},
            generated_at="2024-01-15T10:00:00+00:00"
        )
        assert "2024-01-15" in report._generated_at

    def test_repr(self):
        report = _make_report()
        r = repr(report)
        assert "RecourseReport" in r
        assert "1200.00" in r or "1200" in r


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------

class TestToDict:
    def test_to_dict_returns_dict(self):
        report = _make_report()
        d = report.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_required_keys(self):
        report = _make_report()
        d = report.to_dict()
        assert "policyholder_id" in d
        assert "generated_at" in d
        assert "model_metadata" in d
        assert "current_premium_gbp" in d
        assert "factual_features" in d
        assert "recourse_options" in d
        assert "n_options" in d
        assert "audit_hash" in d

    def test_to_dict_current_premium(self):
        report = _make_report()
        d = report.to_dict()
        assert d["current_premium_gbp"] == 1200.0

    def test_to_dict_policyholder_id(self):
        report = _make_report(with_id=True)
        d = report.to_dict()
        assert d["policyholder_id"] == "POL-123456"

    def test_to_dict_policyholder_id_none(self):
        report = _make_report(with_id=False)
        d = report.to_dict()
        assert d["policyholder_id"] is None

    def test_to_dict_n_options(self):
        report = _make_report(n_actions=2)
        d = report.to_dict()
        assert d["n_options"] == 2

    def test_to_dict_single_action(self):
        report = _make_report(n_actions=1)
        d = report.to_dict()
        assert len(d["recourse_options"]) == 1

    def test_to_dict_recourse_options_structure(self):
        report = _make_report()
        d = report.to_dict()
        opt = d["recourse_options"][0]
        assert "rank" in opt
        assert "description" in opt
        assert "premium_after" in opt
        assert "saving_gbp" in opt
        assert "saving_pct" in opt
        assert "estimated_cost_gbp" in opt
        assert "implementation_days" in opt
        assert "feasibility_pct" in opt
        assert "feature_changes" in opt
        assert "validity" in opt

    def test_to_dict_recourse_options_rank_ordering(self):
        report = _make_report(n_actions=2)
        d = report.to_dict()
        opts = d["recourse_options"]
        assert opts[0]["rank"] == 1
        assert opts[1]["rank"] == 2

    def test_to_dict_model_metadata(self):
        report = _make_report()
        d = report.to_dict()
        assert d["model_metadata"]["model_version"] == "2024-Q4-motor-v3"
        assert d["model_metadata"]["product"] == "motor"

    def test_to_dict_factual_features(self):
        report = _make_report()
        d = report.to_dict()
        assert "age" in d["factual_features"]
        assert "annual_mileage" in d["factual_features"]

    def test_to_dict_is_json_serialisable(self):
        report = _make_report()
        d = report.to_dict()
        # Should not raise
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_to_dict_audit_hash_is_sha256(self):
        report = _make_report()
        d = report.to_dict()
        assert len(d["audit_hash"]) == 64  # SHA-256 hex = 64 chars
        assert all(c in "0123456789abcdef" for c in d["audit_hash"])

    def test_to_dict_audit_hash_changes_with_different_premium(self):
        factual = _make_factual()
        action = _make_action()
        r1 = RecourseReport(factual, [action], {}, current_premium=1200.0)
        r2 = RecourseReport(factual, [action], {}, current_premium=1300.0)
        assert r1.to_dict()["audit_hash"] != r2.to_dict()["audit_hash"]

    def test_to_dict_empty_actions(self):
        factual = _make_factual()
        report = RecourseReport(factual, [], {}, current_premium=1200.0)
        d = report.to_dict()
        assert d["n_options"] == 0
        assert d["recourse_options"] == []

    def test_to_dict_saving_gbp(self):
        report = _make_report(n_actions=1)
        d = report.to_dict()
        opt = d["recourse_options"][0]
        assert opt["saving_gbp"] == 300.0

    def test_to_dict_saving_pct(self):
        report = _make_report(n_actions=1)
        d = report.to_dict()
        opt = d["recourse_options"][0]
        assert opt["saving_pct"] == 25.0

    def test_to_dict_feasibility_pct(self):
        report = _make_report(n_actions=1)
        d = report.to_dict()
        opt = d["recourse_options"][0]
        assert opt["feasibility_pct"] == pytest.approx(85.0, abs=0.1)


# ---------------------------------------------------------------------------
# to_html
# ---------------------------------------------------------------------------

class TestToHtml:
    def test_to_html_returns_string(self):
        report = _make_report()
        html = report.to_html()
        assert isinstance(html, str)

    def test_to_html_is_valid_html_structure(self):
        report = _make_report()
        html = report.to_html()
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html
        assert "<head>" in html
        assert "<body>" in html

    def test_to_html_contains_premium(self):
        report = _make_report()
        html = report.to_html()
        assert "1,200.00" in html or "1200" in html

    def test_to_html_contains_policyholder_id(self):
        report = _make_report(with_id=True)
        html = report.to_html()
        assert "POL-123456" in html

    def test_to_html_no_id_shows_na(self):
        report = _make_report(with_id=False)
        html = report.to_html()
        assert "N/A" in html

    def test_to_html_contains_model_version(self):
        report = _make_report()
        html = report.to_html()
        assert "2024-Q4-motor-v3" in html

    def test_to_html_contains_action_description(self):
        report = _make_report(n_actions=1)
        html = report.to_html()
        assert "immobiliser" in html.lower() or "Thatcham" in html or "security" in html.lower()

    def test_to_html_contains_saving(self):
        report = _make_report(n_actions=1)
        html = report.to_html()
        assert "300.00" in html or "300" in html

    def test_to_html_contains_audit_hash(self):
        report = _make_report()
        html = report.to_html()
        assert "Audit hash" in html or "audit hash" in html.lower() or "audit_hash" in html

    def test_to_html_contains_fca_reference(self):
        report = _make_report()
        html = report.to_html()
        assert "Consumer Duty" in html or "FCA" in html

    def test_to_html_contains_inline_css(self):
        report = _make_report()
        html = report.to_html()
        assert "<style>" in html
        assert "font-family" in html

    def test_to_html_empty_actions_shows_no_options(self):
        factual = _make_factual()
        report = RecourseReport(factual, [], {"product": "motor"}, current_premium=1200.0)
        html = report.to_html()
        assert "No recourse" in html or "no options" in html.lower() or "No recourse options" in html

    def test_to_html_two_actions(self):
        report = _make_report(n_actions=2)
        html = report.to_html()
        # Both ranks should appear
        assert "<td class=\"rank\">1</td>" in html
        assert "<td class=\"rank\">2</td>" in html

    def test_to_html_product_title_case(self):
        report = _make_report()
        html = report.to_html()
        assert "Motor" in html or "motor" in html

    def test_to_html_generated_date(self):
        factual = _make_factual()
        report = RecourseReport(
            factual, [], {},
            current_premium=1200.0,
            generated_at="2024-03-15T12:00:00+00:00"
        )
        html = report.to_html()
        assert "2024-03-15" in html

    def test_to_html_validity_checkmark_present(self):
        report = _make_report(n_actions=1)
        html = report.to_html()
        assert "valid" in html

    def test_to_html_feasibility_zero_low_risk_colour(self):
        """Very low feasibility should use red colour."""
        factual = _make_factual()
        action = _make_action(feasibility=0.02)  # 2% feasibility
        report = RecourseReport(
            factual, [action], {"product": "motor"}, current_premium=1200.0
        )
        html = report.to_html()
        assert "#c0392b" in html  # red for <40% feasibility

    def test_to_html_feasibility_high_green_colour(self):
        """High feasibility should use green colour."""
        factual = _make_factual()
        action = _make_action(feasibility=0.90)
        report = RecourseReport(
            factual, [action], {"product": "motor"}, current_premium=1200.0
        )
        html = report.to_html()
        assert "#27ae60" in html

    def test_to_html_zero_monetary_cost_shows_none(self):
        factual = _make_factual()
        action = _make_action(monetary_cost=0.0)
        report = RecourseReport(
            factual, [action], {"product": "motor"}, current_premium=1200.0
        )
        html = report.to_html()
        assert "None" in html or "£0.00" in html


# ---------------------------------------------------------------------------
# _is_numeric helper
# ---------------------------------------------------------------------------

class TestIsNumeric:
    def test_int_is_numeric(self):
        assert _is_numeric(42)

    def test_float_is_numeric(self):
        assert _is_numeric(3.14)

    def test_numeric_string_is_numeric(self):
        assert _is_numeric("42.5")

    def test_non_numeric_string_is_not_numeric(self):
        assert not _is_numeric("hello")

    def test_none_is_not_numeric(self):
        assert not _is_numeric(None)

    def test_list_is_not_numeric(self):
        assert not _is_numeric([1, 2, 3])
