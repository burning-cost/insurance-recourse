"""Tests for insurance_recourse.constraints."""

import warnings

import pytest

from insurance_recourse.constraints import (
    ActionabilityGraph,
    FeatureConstraint,
    Mutability,
)


# ---------------------------------------------------------------------------
# FeatureConstraint
# ---------------------------------------------------------------------------

class TestFeatureConstraint:
    def test_immutable_default(self):
        fc = FeatureConstraint(name="age", mutability=Mutability.IMMUTABLE, direction=None)
        assert not fc.is_actionable()

    def test_mutable_default(self):
        fc = FeatureConstraint(name="annual_mileage")
        assert fc.is_actionable()

    def test_conditionally_mutable_is_actionable(self):
        fc = FeatureConstraint(name="postcode_risk", mutability=Mutability.CONDITIONALLY_MUTABLE)
        assert fc.is_actionable()

    def test_feasibility_rate_out_of_range_raises(self):
        with pytest.raises(ValueError, match="feasibility_rate"):
            FeatureConstraint(name="x", feasibility_rate=1.5)

    def test_feasibility_rate_negative_raises(self):
        with pytest.raises(ValueError):
            FeatureConstraint(name="x", feasibility_rate=-0.1)

    def test_effort_weight_negative_raises(self):
        with pytest.raises(ValueError, match="effort_weight"):
            FeatureConstraint(name="x", effort_weight=-1.0)

    def test_immutable_with_direction_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            FeatureConstraint(name="age", mutability=Mutability.IMMUTABLE, direction="increase")
            assert any("direction" in str(warning.message) for warning in w)

    def test_clip_counterfactual_decrease_only(self):
        fc = FeatureConstraint(name="mileage", direction="decrease")
        assert fc.clip_counterfactual(10000, 8000) == 8000
        assert fc.clip_counterfactual(10000, 12000) == 10000  # clipped to original

    def test_clip_counterfactual_increase_only(self):
        fc = FeatureConstraint(name="security", direction="increase")
        assert fc.clip_counterfactual(2, 3) == 3
        assert fc.clip_counterfactual(2, 1) == 2  # clipped to original

    def test_clip_counterfactual_either(self):
        fc = FeatureConstraint(name="x", direction="either")
        assert fc.clip_counterfactual(5, 3) == 3
        assert fc.clip_counterfactual(5, 7) == 7

    def test_clip_counterfactual_with_bounds(self):
        fc = FeatureConstraint(name="mileage", direction="decrease", min_value=1000.0, max_value=50000.0)
        assert fc.clip_counterfactual(20000, 500) == 1000.0  # bounded by min
        assert fc.clip_counterfactual(20000, 60000) == 20000.0  # direction prevents increase

    def test_clip_counterfactual_immutable(self):
        fc = FeatureConstraint(name="age", mutability=Mutability.IMMUTABLE, direction=None)
        assert fc.clip_counterfactual(35, 40) == 35  # unchanged

    def test_validate_counterfactual_immutable_unchanged(self):
        fc = FeatureConstraint(name="age", mutability=Mutability.IMMUTABLE, direction=None)
        valid, reason = fc.validate_counterfactual(35, 35)
        assert valid
        assert reason == ""

    def test_validate_counterfactual_immutable_changed_invalid(self):
        fc = FeatureConstraint(name="age", mutability=Mutability.IMMUTABLE, direction=None)
        valid, reason = fc.validate_counterfactual(35, 40)
        assert not valid
        assert "immutable" in reason

    def test_validate_counterfactual_direction_decrease_valid(self):
        fc = FeatureConstraint(name="mileage", direction="decrease")
        valid, reason = fc.validate_counterfactual(10000, 8000)
        assert valid

    def test_validate_counterfactual_direction_decrease_invalid(self):
        fc = FeatureConstraint(name="mileage", direction="decrease")
        valid, reason = fc.validate_counterfactual(10000, 12000)
        assert not valid
        assert "decrease" in reason

    def test_validate_counterfactual_direction_increase_valid(self):
        fc = FeatureConstraint(name="security", direction="increase")
        valid, reason = fc.validate_counterfactual(1, 3)
        assert valid

    def test_validate_counterfactual_direction_increase_invalid(self):
        fc = FeatureConstraint(name="security", direction="increase")
        valid, reason = fc.validate_counterfactual(3, 1)
        assert not valid
        assert "increase" in reason

    def test_validate_counterfactual_allowed_values_valid(self):
        fc = FeatureConstraint(name="garaging", allowed_values=[0, 1, 2, 3])
        valid, _ = fc.validate_counterfactual(0, 2)
        assert valid

    def test_validate_counterfactual_allowed_values_invalid(self):
        fc = FeatureConstraint(name="garaging", allowed_values=[0, 1, 2, 3])
        valid, reason = fc.validate_counterfactual(0, 5)
        assert not valid
        assert "allowed_values" in reason

    def test_validate_counterfactual_min_value(self):
        fc = FeatureConstraint(name="mileage", direction="decrease", min_value=1000.0)
        valid, reason = fc.validate_counterfactual(10000, 500)
        assert not valid
        assert "min_value" in reason

    def test_validate_counterfactual_max_value(self):
        fc = FeatureConstraint(name="mileage", direction="increase", max_value=50000.0)
        valid, reason = fc.validate_counterfactual(10000, 60000)
        assert not valid
        assert "max_value" in reason

    def test_feasibility_zero(self):
        fc = FeatureConstraint(name="age", mutability=Mutability.IMMUTABLE, direction=None,
                               feasibility_rate=0.0)
        assert fc.feasibility_rate == 0.0

    def test_causal_children_default_empty(self):
        fc = FeatureConstraint(name="x")
        assert fc.causal_children == []


# ---------------------------------------------------------------------------
# ActionabilityGraph
# ---------------------------------------------------------------------------

class TestActionabilityGraph:
    def _simple_graph(self) -> ActionabilityGraph:
        constraints = {
            "age": FeatureConstraint(name="age", mutability=Mutability.IMMUTABLE, direction=None),
            "mileage": FeatureConstraint(name="mileage", direction="decrease", min_value=1000.0),
            "security": FeatureConstraint(name="security", direction="increase",
                                          allowed_values=[0, 1, 2, 3]),
            "postcode": FeatureConstraint(
                name="postcode",
                mutability=Mutability.CONDITIONALLY_MUTABLE,
                direction="decrease",
                causal_children=["crime_rate"],
            ),
            "crime_rate": FeatureConstraint(
                name="crime_rate",
                mutability=Mutability.CONDITIONALLY_MUTABLE,
                direction=None,
            ),
        }

        def propagate_postcode(factual, interventions):
            new_pc = interventions.get("postcode", factual.get("postcode", 5))
            return {"crime_rate": max(1, round(new_pc * 0.8))}

        return ActionabilityGraph(constraints, {"postcode": propagate_postcode})

    def test_get_mutable_features(self):
        graph = self._simple_graph()
        mutable = graph.get_mutable_features()
        assert "mileage" in mutable
        assert "security" in mutable
        assert "postcode" in mutable
        assert "crime_rate" in mutable
        assert "age" not in mutable

    def test_get_immutable_features(self):
        graph = self._simple_graph()
        immutable = graph.get_immutable_features()
        assert "age" in immutable
        assert "mileage" not in immutable

    def test_get_constraint_existing(self):
        graph = self._simple_graph()
        c = graph.get_constraint("age")
        assert c.name == "age"

    def test_get_constraint_missing_raises(self):
        graph = self._simple_graph()
        with pytest.raises(KeyError):
            graph.get_constraint("nonexistent")

    def test_propagate_causal_effects_basic(self):
        graph = self._simple_graph()
        factual = {"age": 30, "mileage": 10000, "security": 1, "postcode": 6, "crime_rate": 5}
        interventions = {"postcode": 3, "mileage": 8000}
        result = graph.propagate_causal_effects(factual, interventions)
        assert result["postcode"] == 3
        assert result["mileage"] == 8000
        assert result["crime_rate"] == max(1, round(3 * 0.8))  # propagated
        assert result["age"] == 30  # unchanged

    def test_propagate_causal_effects_no_changes(self):
        graph = self._simple_graph()
        factual = {"age": 30, "mileage": 10000, "security": 1, "postcode": 6, "crime_rate": 5}
        result = graph.propagate_causal_effects(factual, {})
        assert result == factual

    def test_propagate_causal_effects_preserves_unmodified(self):
        graph = self._simple_graph()
        factual = {"age": 35, "mileage": 12000, "security": 2, "postcode": 5, "crime_rate": 4}
        interventions = {"security": 3}
        result = graph.propagate_causal_effects(factual, interventions)
        assert result["security"] == 3
        assert result["age"] == 35
        assert result["mileage"] == 12000

    def test_validate_counterfactual_all_valid(self):
        graph = self._simple_graph()
        factual = {"age": 30, "mileage": 10000, "security": 1, "postcode": 6, "crime_rate": 5}
        cf = {"age": 30, "mileage": 8000, "security": 2, "postcode": 4, "crime_rate": 3}
        valid, errors = graph.validate_counterfactual(factual, cf)
        assert valid
        assert errors == []

    def test_validate_counterfactual_immutable_changed(self):
        graph = self._simple_graph()
        factual = {"age": 30, "mileage": 10000, "security": 1, "postcode": 6, "crime_rate": 5}
        cf = {"age": 40, "mileage": 8000, "security": 2, "postcode": 4, "crime_rate": 3}
        valid, errors = graph.validate_counterfactual(factual, cf)
        assert not valid
        assert any("age" in e for e in errors)

    def test_validate_counterfactual_direction_violated(self):
        graph = self._simple_graph()
        factual = {"age": 30, "mileage": 10000, "security": 1, "postcode": 6, "crime_rate": 5}
        cf = {"age": 30, "mileage": 15000, "security": 2, "postcode": 4, "crime_rate": 3}
        valid, errors = graph.validate_counterfactual(factual, cf)
        assert not valid
        assert any("mileage" in e for e in errors)

    def test_validate_counterfactual_allowed_values_violated(self):
        graph = self._simple_graph()
        factual = {"age": 30, "mileage": 10000, "security": 1, "postcode": 6, "crime_rate": 5}
        cf = {"age": 30, "mileage": 8000, "security": 7, "postcode": 4, "crime_rate": 3}
        valid, errors = graph.validate_counterfactual(factual, cf)
        assert not valid
        assert any("security" in e for e in errors)

    def test_add_constraint(self):
        graph = self._simple_graph()
        new_c = FeatureConstraint(name="telematics", allowed_values=[0, 1])
        graph.add_constraint(new_c)
        assert "telematics" in graph.get_mutable_features()

    def test_add_propagation_function(self):
        graph = self._simple_graph()

        def fn(factual, interventions):
            return {"crime_rate": 1}

        graph.add_propagation_function("mileage", fn)
        assert "mileage" in graph._propagation_functions

    def test_all_constraints_returns_copy(self):
        graph = self._simple_graph()
        constraints = graph.all_constraints()
        assert isinstance(constraints, dict)
        constraints["new"] = None  # mutating the copy should not affect graph
        assert "new" not in graph._constraints

    def test_repr(self):
        graph = self._simple_graph()
        r = repr(graph)
        assert "ActionabilityGraph" in r
        assert "mutable" in r

    def test_topological_order_parents_before_children(self):
        graph = self._simple_graph()
        order = graph._topological_order()
        assert order.index("postcode") < order.index("crime_rate")

    def test_undefined_causal_child_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            constraints = {
                "x": FeatureConstraint(name="x", causal_children=["undefined_child"]),
            }
            ActionabilityGraph(constraints)
            assert any("undefined_child" in str(warning.message) for warning in w)


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

class TestTemplates:
    def test_motor_template_loads(self):
        graph = ActionabilityGraph.from_template("motor")
        assert "age" in graph.get_immutable_features()
        assert "annual_mileage" in graph.get_mutable_features()
        assert "vehicle_security" in graph.get_mutable_features()
        assert "pass_plus" in graph.get_mutable_features()
        assert "garaging" in graph.get_mutable_features()

    def test_motor_template_causal_propagation(self):
        graph = ActionabilityGraph.from_template("motor")
        factual = {f: 5 for f in graph.all_constraints()}
        factual["postcode_risk"] = 8
        interventions = {"postcode_risk": 3}
        result = graph.propagate_causal_effects(factual, interventions)
        # crime_rate and flood_zone should change
        assert result["crime_rate_decile"] != 5
        assert "flood_zone_risk" in result

    def test_motor_template_vehicle_age_propagation(self):
        graph = ActionabilityGraph.from_template("motor")
        factual = {f: 0 for f in graph.all_constraints()}
        factual["vehicle_age"] = 5
        factual["vehicle_value"] = 15000.0
        interventions = {"vehicle_age": 2}
        result = graph.propagate_causal_effects(factual, interventions)
        assert result["vehicle_value"] != 15000.0

    def test_home_template_loads(self):
        graph = ActionabilityGraph.from_template("home")
        assert "property_age" in graph.get_immutable_features()
        assert "alarm_grade" in graph.get_mutable_features()
        assert "locks_grade" in graph.get_mutable_features()
        assert "voluntary_excess" in graph.get_mutable_features()

    def test_home_template_causal_propagation(self):
        graph = ActionabilityGraph.from_template("home")
        factual = {f: 2 for f in graph.all_constraints()}
        interventions = {"postcode_flood_risk": 1}
        result = graph.propagate_causal_effects(factual, interventions)
        assert "subsidence_risk" in result

    def test_invalid_template_raises(self):
        with pytest.raises(ValueError, match="Unknown product template"):
            ActionabilityGraph.from_template("travel")  # type: ignore

    def test_motor_immutable_features(self):
        graph = ActionabilityGraph.from_template("motor")
        immutable = graph.get_immutable_features()
        assert "age" in immutable
        assert "gender" in immutable
        assert "years_no_claims" in immutable
        assert "at_fault_claims_3yr" in immutable
        assert "licence_years" in immutable

    def test_motor_mutable_direction_constraints(self):
        graph = ActionabilityGraph.from_template("motor")
        mileage_c = graph.get_constraint("annual_mileage")
        assert mileage_c.direction == "decrease"
        security_c = graph.get_constraint("vehicle_security")
        assert security_c.direction == "increase"

    def test_home_mutable_direction_constraints(self):
        graph = ActionabilityGraph.from_template("home")
        alarm_c = graph.get_constraint("alarm_grade")
        assert alarm_c.direction == "increase"
        excess_c = graph.get_constraint("voluntary_excess")
        assert excess_c.direction == "increase"
