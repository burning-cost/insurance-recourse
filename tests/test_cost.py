"""Tests for insurance_recourse.cost."""

import pytest
import pandas as pd

from insurance_recourse.cost import InsuranceCostFunction, RecourseEffort


# ---------------------------------------------------------------------------
# RecourseEffort
# ---------------------------------------------------------------------------

class TestRecourseEffort:
    def test_basic_construction(self):
        effort = RecourseEffort(
            monetary_cost=250.0,
            time_days=7.0,
            feasibility_probability=0.85,
            feature_changes={"vehicle_security": (1, 2)},
        )
        assert effort.monetary_cost == 250.0
        assert effort.time_days == 7.0
        assert effort.feasibility_probability == 0.85

    def test_negative_monetary_cost_raises(self):
        with pytest.raises(ValueError, match="monetary_cost"):
            RecourseEffort(monetary_cost=-10.0, time_days=7.0, feasibility_probability=0.5)

    def test_negative_time_raises(self):
        with pytest.raises(ValueError, match="time_days"):
            RecourseEffort(monetary_cost=0.0, time_days=-1.0, feasibility_probability=0.5)

    def test_feasibility_out_of_range_raises(self):
        with pytest.raises(ValueError, match="feasibility_probability"):
            RecourseEffort(monetary_cost=0.0, time_days=0.0, feasibility_probability=1.5)

    def test_feasibility_negative_raises(self):
        with pytest.raises(ValueError):
            RecourseEffort(monetary_cost=0.0, time_days=0.0, feasibility_probability=-0.1)

    def test_is_feasible_above_threshold(self):
        effort = RecourseEffort(monetary_cost=0, time_days=0, feasibility_probability=0.1)
        assert effort.is_feasible

    def test_is_feasible_below_threshold(self):
        effort = RecourseEffort(monetary_cost=0, time_days=0, feasibility_probability=0.04)
        assert not effort.is_feasible

    def test_is_feasible_at_threshold(self):
        effort = RecourseEffort(monetary_cost=0, time_days=0, feasibility_probability=0.05)
        assert effort.is_feasible

    def test_as_dict_structure(self):
        effort = RecourseEffort(
            monetary_cost=100.0,
            time_days=14.0,
            feasibility_probability=0.75,
            feature_changes={"mileage": (10000, 8000)},
        )
        d = effort.as_dict()
        assert "monetary_cost_gbp" in d
        assert "time_days" in d
        assert "feasibility_probability" in d
        assert "feature_changes" in d
        assert d["monetary_cost_gbp"] == 100.0
        assert d["feature_changes"]["mileage"]["from"] == 10000
        assert d["feature_changes"]["mileage"]["to"] == 8000

    def test_as_dict_empty_changes(self):
        effort = RecourseEffort(monetary_cost=0, time_days=0, feasibility_probability=1.0)
        d = effort.as_dict()
        assert d["feature_changes"] == {}

    def test_zero_cost(self):
        effort = RecourseEffort(monetary_cost=0.0, time_days=0.0, feasibility_probability=1.0)
        assert effort.monetary_cost == 0.0
        assert effort.time_days == 0.0


# ---------------------------------------------------------------------------
# InsuranceCostFunction
# ---------------------------------------------------------------------------

class TestInsuranceCostFunction:
    def _make_series(self, d: dict) -> pd.Series:
        return pd.Series(d)

    def test_compute_no_changes(self):
        cf = InsuranceCostFunction()
        factual = self._make_series({"mileage": 10000, "security": 1})
        counterfactual = self._make_series({"mileage": 10000, "security": 1})
        effort = cf.compute(factual, counterfactual)
        assert effort.monetary_cost == 0.0
        assert effort.time_days == 0.0
        assert effort.feasibility_probability == 1.0
        assert effort.feature_changes == {}

    def test_compute_single_change_categorical(self):
        cf = InsuranceCostFunction(
            monetary_costs={"security": 250.0},
            time_costs_days={"security": 7.0},
            feasibility_rates={"security": 0.85},
        )
        factual = self._make_series({"security": 1})
        counterfactual = self._make_series({"security": 2})
        effort = cf.compute(factual, counterfactual)
        assert effort.monetary_cost == 250.0  # flat per-event cost * magnitude=1
        assert effort.time_days == 7.0
        assert abs(effort.feasibility_probability - 0.85) < 1e-6
        assert "security" in effort.feature_changes
        assert effort.feature_changes["security"] == (1, 2)

    def test_compute_continuous_feature_scaled(self):
        cf = InsuranceCostFunction(
            monetary_costs={"mileage": 0.01},  # £0.01 per mile reduction
            time_costs_days={"mileage": 1.0},
            feasibility_rates={"mileage": 0.6},
        )
        factual = self._make_series({"mileage": 10000.0})
        counterfactual = self._make_series({"mileage": 8000.0})
        effort = cf.compute(factual, counterfactual)
        # magnitude = 2000, cost = 2000 * 0.01 = 20
        assert abs(effort.monetary_cost - 20.0) < 1e-4
        assert effort.time_days == 2000.0  # max(1, 2000) * 1.0
        assert abs(effort.feasibility_probability - 0.6) < 1e-6

    def test_compute_multiple_changes_feasibility_product(self):
        cf = InsuranceCostFunction(
            monetary_costs={"security": 250.0, "pass_plus": 150.0},
            time_costs_days={"security": 7.0, "pass_plus": 90.0},
            feasibility_rates={"security": 0.85, "pass_plus": 0.40},
        )
        factual = self._make_series({"security": 1, "pass_plus": 0})
        counterfactual = self._make_series({"security": 2, "pass_plus": 1})
        effort = cf.compute(factual, counterfactual)
        assert effort.monetary_cost == 250.0 + 150.0  # both flat costs
        assert effort.time_days == 90.0  # max of (7, 90)
        expected_feasibility = 0.85 * 0.40
        assert abs(effort.feasibility_probability - expected_feasibility) < 1e-6

    def test_compute_uses_default_for_unknown_feature(self):
        cf = InsuranceCostFunction(
            default_monetary_cost=5.0,
            default_time_days=3.0,
            default_feasibility=0.7,
        )
        factual = self._make_series({"unknown_feature": 1})
        counterfactual = self._make_series({"unknown_feature": 2})
        effort = cf.compute(factual, counterfactual)
        assert effort.monetary_cost == 5.0  # magnitude 1 * default 5.0
        assert effort.time_days == 3.0
        assert abs(effort.feasibility_probability - 0.7) < 1e-6

    def test_compute_ignores_features_not_in_factual(self):
        cf = InsuranceCostFunction()
        factual = self._make_series({"mileage": 10000})
        counterfactual = self._make_series({"mileage": 8000, "extra_feature": 99})
        effort = cf.compute(factual, counterfactual)
        assert "extra_feature" not in effort.feature_changes

    def test_compute_feature_changes_recorded(self):
        cf = InsuranceCostFunction()
        factual = self._make_series({"mileage": 10000, "security": 1})
        counterfactual = self._make_series({"mileage": 8000, "security": 1})
        effort = cf.compute(factual, counterfactual)
        assert "mileage" in effort.feature_changes
        assert effort.feature_changes["mileage"] == (10000, 8000)
        assert "security" not in effort.feature_changes  # unchanged

    def test_compute_time_is_max_not_sum(self):
        """Time to completion is the slowest change, not the sum."""
        cf = InsuranceCostFunction(
            time_costs_days={"fast_change": 1.0, "slow_change": 90.0},
        )
        factual = self._make_series({"fast_change": 0, "slow_change": 0})
        counterfactual = self._make_series({"fast_change": 1, "slow_change": 1})
        effort = cf.compute(factual, counterfactual)
        assert effort.time_days == 90.0

    def test_motor_defaults_loads(self):
        cf = InsuranceCostFunction.motor_defaults()
        assert isinstance(cf, InsuranceCostFunction)
        assert "vehicle_security" in cf.monetary_costs
        assert "pass_plus" in cf.monetary_costs

    def test_motor_defaults_security_cost(self):
        cf = InsuranceCostFunction.motor_defaults()
        factual = self._make_series({"vehicle_security": 1})
        counterfactual = self._make_series({"vehicle_security": 2})
        effort = cf.compute(factual, counterfactual)
        assert effort.monetary_cost == 250.0  # 250 * magnitude=1

    def test_motor_defaults_pass_plus_cost(self):
        cf = InsuranceCostFunction.motor_defaults()
        factual = self._make_series({"pass_plus": 0})
        counterfactual = self._make_series({"pass_plus": 1})
        effort = cf.compute(factual, counterfactual)
        assert effort.monetary_cost == 150.0

    def test_motor_defaults_mileage_no_direct_cost(self):
        cf = InsuranceCostFunction.motor_defaults()
        factual = self._make_series({"annual_mileage": 15000.0})
        counterfactual = self._make_series({"annual_mileage": 8000.0})
        effort = cf.compute(factual, counterfactual)
        assert effort.monetary_cost == 0.0  # mileage reduction has no monetary cost

    def test_home_defaults_loads(self):
        cf = InsuranceCostFunction.home_defaults()
        assert isinstance(cf, InsuranceCostFunction)
        assert "alarm_grade" in cf.monetary_costs
        assert "locks_grade" in cf.monetary_costs

    def test_home_defaults_alarm_cost(self):
        cf = InsuranceCostFunction.home_defaults()
        factual = self._make_series({"alarm_grade": 0})
        counterfactual = self._make_series({"alarm_grade": 2})
        effort = cf.compute(factual, counterfactual)
        assert effort.monetary_cost == 800.0 * 2  # 800 per step, 2 steps

    def test_home_defaults_smoke_alarm_cost(self):
        cf = InsuranceCostFunction.home_defaults()
        factual = self._make_series({"smoke_alarms": 0})
        counterfactual = self._make_series({"smoke_alarms": 1})
        effort = cf.compute(factual, counterfactual)
        assert effort.monetary_cost == 25.0

    def test_repr(self):
        cf = InsuranceCostFunction.motor_defaults()
        r = repr(cf)
        assert "InsuranceCostFunction" in r
        assert "features_with_costs" in r

    def test_compute_categorical_string_values(self):
        """Cost function handles string categoricals."""
        cf = InsuranceCostFunction(
            monetary_costs={"occupation": 0.0},
            feasibility_rates={"occupation": 0.3},
        )
        factual = self._make_series({"occupation": "teacher"})
        counterfactual = self._make_series({"occupation": "civil_servant"})
        effort = cf.compute(factual, counterfactual)
        assert "occupation" in effort.feature_changes
        assert effort.feasibility_probability == pytest.approx(0.3)

    def test_compute_feasibility_clipped_to_one(self):
        """Feasibility of 1.0 stays at 1.0."""
        cf = InsuranceCostFunction(
            feasibility_rates={"x": 1.0, "y": 1.0},
        )
        factual = self._make_series({"x": 0, "y": 0})
        counterfactual = self._make_series({"x": 1, "y": 1})
        effort = cf.compute(factual, counterfactual)
        assert effort.feasibility_probability == 1.0
