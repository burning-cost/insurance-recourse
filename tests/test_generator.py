"""
Tests for insurance_recourse.generator.

Backend integration tests (DiCE, alibi) are skipped if the respective
package is not installed. Core logic (RecourseAction, description generation,
constraint propagation in _build_action, FOCUS tree extraction) is tested
with pure sklearn models — no external backend required.
"""

from __future__ import annotations

import sys
import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from insurance_recourse.constraints import ActionabilityGraph, FeatureConstraint, Mutability
from insurance_recourse.cost import InsuranceCostFunction
from insurance_recourse.generator import RecourseAction, RecourseGenerator, _generate_description


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_graph() -> ActionabilityGraph:
    constraints = {
        "age": FeatureConstraint(name="age", mutability=Mutability.IMMUTABLE, direction=None),
        "annual_mileage": FeatureConstraint(
            name="annual_mileage", direction="decrease", min_value=1000.0, max_value=50000.0
        ),
        "vehicle_security": FeatureConstraint(
            name="vehicle_security", direction="increase",
            allowed_values=[0, 1, 2, 3, 4],
        ),
        "pass_plus": FeatureConstraint(
            name="pass_plus", direction="increase", allowed_values=[0, 1],
        ),
        "postcode_risk": FeatureConstraint(
            name="postcode_risk",
            mutability=Mutability.CONDITIONALLY_MUTABLE,
            direction="decrease",
            min_value=1, max_value=10,
            causal_children=["crime_rate_decile"],
        ),
        "crime_rate_decile": FeatureConstraint(
            name="crime_rate_decile",
            mutability=Mutability.CONDITIONALLY_MUTABLE,
            direction=None,
        ),
    }

    def propagate_postcode(factual, interventions):
        new_pc = interventions.get("postcode_risk", factual.get("postcode_risk", 5))
        return {"crime_rate_decile": max(1, round(new_pc * 0.9))}

    return ActionabilityGraph(constraints, {"postcode_risk": propagate_postcode})


def _simple_cost() -> InsuranceCostFunction:
    return InsuranceCostFunction(
        monetary_costs={"vehicle_security": 250.0, "pass_plus": 150.0},
        time_costs_days={"vehicle_security": 7.0, "pass_plus": 90.0},
        feasibility_rates={"vehicle_security": 0.85, "pass_plus": 0.40},
        default_feasibility=0.5,
    )


def _simple_model():
    """Simple linear model: premium = 500 + 0.05 * mileage - 100 * security."""
    class LinearModel:
        def predict(self, df: pd.DataFrame) -> np.ndarray:
            result = np.zeros(len(df))
            if "annual_mileage" in df.columns:
                result += 0.05 * df["annual_mileage"].to_numpy(dtype=float)
            if "vehicle_security" in df.columns:
                result -= 100.0 * df["vehicle_security"].to_numpy(dtype=float)
            if "postcode_risk" in df.columns:
                result += 50.0 * df["postcode_risk"].to_numpy(dtype=float)
            if "crime_rate_decile" in df.columns:
                result += 20.0 * df["crime_rate_decile"].to_numpy(dtype=float)
            result += 500.0
            return result

    return LinearModel()


def _make_factual() -> pd.Series:
    return pd.Series({
        "age": 35,
        "annual_mileage": 12000.0,
        "vehicle_security": 1,
        "pass_plus": 0,
        "postcode_risk": 6,
        "crime_rate_decile": 5,
    })


# ---------------------------------------------------------------------------
# RecourseAction
# ---------------------------------------------------------------------------

class TestRecourseAction:
    def _make_effort(self):
        from insurance_recourse.cost import RecourseEffort
        return RecourseEffort(
            monetary_cost=250.0,
            time_days=7.0,
            feasibility_probability=0.85,
            feature_changes={"vehicle_security": (1, 2)},
        )

    def test_basic_construction(self):
        effort = self._make_effort()
        action = RecourseAction(
            feature_changes={"vehicle_security": (1, 2)},
            predicted_premium=900.0,
            premium_reduction=300.0,
            premium_reduction_pct=25.0,
            effort=effort,
            validity=True,
        )
        assert action.predicted_premium == 900.0
        assert action.premium_reduction == 300.0
        assert action.validity is True

    def test_as_dict_structure(self):
        effort = self._make_effort()
        action = RecourseAction(
            feature_changes={"vehicle_security": (1, 2)},
            predicted_premium=900.0,
            premium_reduction=300.0,
            premium_reduction_pct=25.0,
            effort=effort,
        )
        d = action.as_dict()
        assert "feature_changes" in d
        assert "predicted_premium" in d
        assert "premium_reduction" in d
        assert "premium_reduction_pct" in d
        assert "effort" in d
        assert "causal_effects" in d
        assert "validity" in d
        assert "description" in d

    def test_as_dict_feature_changes_format(self):
        effort = self._make_effort()
        action = RecourseAction(
            feature_changes={"mileage": (10000, 8000)},
            predicted_premium=900.0,
            premium_reduction=100.0,
            premium_reduction_pct=10.0,
            effort=effort,
        )
        d = action.as_dict()
        assert d["feature_changes"]["mileage"]["from"] == 10000
        assert d["feature_changes"]["mileage"]["to"] == 8000

    def test_causal_effects_default_empty(self):
        effort = self._make_effort()
        action = RecourseAction(
            feature_changes={},
            predicted_premium=900.0,
            premium_reduction=0.0,
            premium_reduction_pct=0.0,
            effort=effort,
        )
        assert action.causal_effects == {}

    def test_validity_defaults_true(self):
        effort = self._make_effort()
        action = RecourseAction(
            feature_changes={},
            predicted_premium=900.0,
            premium_reduction=0.0,
            premium_reduction_pct=0.0,
            effort=effort,
        )
        assert action.validity is True


# ---------------------------------------------------------------------------
# _generate_description
# ---------------------------------------------------------------------------

class TestGenerateDescription:
    def test_annual_mileage(self):
        d = _generate_description({"annual_mileage": (12000, 8000)})
        assert "mileage" in d.lower() or "12,000" in d or "8,000" in d

    def test_vehicle_security(self):
        d = _generate_description({"vehicle_security": (1, 3)})
        assert "security" in d.lower()

    def test_pass_plus(self):
        d = _generate_description({"pass_plus": (0, 1)})
        assert "Pass Plus" in d or "pass plus" in d.lower()

    def test_garaging(self):
        d = _generate_description({"garaging": (0, 2)})
        assert "parking" in d.lower() or "garaging" in d.lower()

    def test_voluntary_excess(self):
        d = _generate_description({"voluntary_excess": (250, 500)})
        assert "excess" in d.lower() or "£250" in d or "£500" in d

    def test_unknown_feature_fallback(self):
        d = _generate_description({"unknown_field_xyz": (1, 2)})
        assert "unknown_field_xyz" in d

    def test_empty_changes(self):
        d = _generate_description({})
        assert d  # non-empty string

    def test_multiple_changes_joined(self):
        d = _generate_description({
            "annual_mileage": (12000, 8000),
            "vehicle_security": (1, 2),
        })
        assert ";" in d or "\n" in d or len(d) > 20


# ---------------------------------------------------------------------------
# RecourseGenerator construction
# ---------------------------------------------------------------------------

class TestRecourseGeneratorConstruction:
    def test_valid_construction(self):
        model = _simple_model()
        graph = _simple_graph()
        cost_fn = _simple_cost()
        gen = RecourseGenerator(model, graph, cost_fn, backend="dice")
        assert gen.backend == "dice"
        assert gen.n_counterfactuals == 5

    def test_callable_model(self):
        graph = _simple_graph()
        cost_fn = _simple_cost()

        def my_predict(df: pd.DataFrame) -> np.ndarray:
            return np.ones(len(df)) * 1000.0

        gen = RecourseGenerator(my_predict, graph, cost_fn)
        assert gen._predict_fn is not None

    def test_invalid_model_raises(self):
        graph = _simple_graph()
        cost_fn = _simple_cost()
        with pytest.raises(TypeError, match="predict"):
            RecourseGenerator("not_a_model", graph, cost_fn)

    def test_repr(self):
        model = _simple_model()
        graph = _simple_graph()
        cost_fn = _simple_cost()
        gen = RecourseGenerator(model, graph, cost_fn)
        r = repr(gen)
        assert "RecourseGenerator" in r
        assert "backend" in r

    def test_custom_n_counterfactuals(self):
        model = _simple_model()
        graph = _simple_graph()
        cost_fn = _simple_cost()
        gen = RecourseGenerator(model, graph, cost_fn, n_counterfactuals=10)
        assert gen.n_counterfactuals == 10

    def test_focus_backend_accepted(self):
        model = _simple_model()
        graph = _simple_graph()
        cost_fn = _simple_cost()
        gen = RecourseGenerator(model, graph, cost_fn, backend="focus")
        assert gen.backend == "focus"

    def test_alibi_backend_accepted(self):
        model = _simple_model()
        graph = _simple_graph()
        cost_fn = _simple_cost()
        gen = RecourseGenerator(model, graph, cost_fn, backend="alibi_cfrl")
        assert gen.backend == "alibi_cfrl"

    def test_invalid_backend_raises_on_generate(self):
        model = _simple_model()
        graph = _simple_graph()
        cost_fn = _simple_cost()
        gen = RecourseGenerator(model, graph, cost_fn)
        gen.backend = "invalid_backend"
        factual = _make_factual()
        with pytest.raises(ValueError, match="Unknown backend"):
            gen.generate(factual, target_premium=900.0, current_premium=1200.0)


# ---------------------------------------------------------------------------
# RecourseGenerator._build_action (no backend needed)
# ---------------------------------------------------------------------------

class TestBuildAction:
    def _make_gen(self):
        return RecourseGenerator(
            _simple_model(), _simple_graph(), _simple_cost(), n_counterfactuals=3
        )

    def test_build_action_basic(self):
        gen = self._make_gen()
        factual = _make_factual()
        cf = factual.copy()
        cf["vehicle_security"] = 3
        cf["annual_mileage"] = 8000.0
        action = gen._build_action(factual, cf, current_premium=1200.0)
        assert action.predicted_premium > 0
        assert action.premium_reduction == pytest.approx(1200.0 - action.predicted_premium, abs=1.0)
        assert "vehicle_security" in action.feature_changes
        assert "annual_mileage" in action.feature_changes

    def test_build_action_causal_propagation(self):
        gen = self._make_gen()
        factual = _make_factual()
        cf = factual.copy()
        cf["postcode_risk"] = 3
        action = gen._build_action(factual, cf, current_premium=1200.0)
        assert "postcode_risk" in action.feature_changes
        # crime_rate_decile should appear in causal_effects
        assert "crime_rate_decile" in action.causal_effects

    def test_build_action_no_changes(self):
        gen = self._make_gen()
        factual = _make_factual()
        action = gen._build_action(factual, factual.copy(), current_premium=1200.0)
        assert action.feature_changes == {}
        assert action.premium_reduction == pytest.approx(0.0, abs=0.01)

    def test_build_action_validity_check(self):
        gen = self._make_gen()
        factual = _make_factual()
        cf = factual.copy()
        cf["vehicle_security"] = 2  # valid: increase
        action = gen._build_action(factual, cf, current_premium=1200.0)
        assert action.validity is True

    def test_build_action_validity_fails_immutable_change(self):
        gen = self._make_gen()
        factual = _make_factual()
        cf = factual.copy()
        cf["age"] = 40  # immutable change
        action = gen._build_action(factual, cf, current_premium=1200.0)
        assert action.validity is False

    def test_build_action_effort_computed(self):
        gen = self._make_gen()
        factual = _make_factual()
        cf = factual.copy()
        cf["vehicle_security"] = 2
        action = gen._build_action(factual, cf, current_premium=1200.0)
        assert action.effort.monetary_cost == 250.0  # Thatcham install

    def test_build_action_description_non_empty(self):
        gen = self._make_gen()
        factual = _make_factual()
        cf = factual.copy()
        cf["vehicle_security"] = 2
        action = gen._build_action(factual, cf, current_premium=1200.0)
        assert len(action.description) > 0


# ---------------------------------------------------------------------------
# RecourseGenerator.generate filters
# ---------------------------------------------------------------------------

class TestGenerateFilters:
    def _make_gen_with_mock_backend(self, cfs: list[pd.Series]):
        """Create a generator with _generate_dice mocked to return cfs."""
        gen = RecourseGenerator(
            _simple_model(), _simple_graph(), _simple_cost(), n_counterfactuals=3
        )
        gen._generate_dice = MagicMock(return_value=cfs)
        return gen

    def _make_cfs(self) -> list:
        factual = _make_factual()
        cf1 = factual.copy(); cf1["vehicle_security"] = 3  # big saving
        cf2 = factual.copy(); cf2["vehicle_security"] = 2  # medium saving
        cf3 = factual.copy(); cf3["annual_mileage"] = 5000.0  # small saving
        return [cf1, cf2, cf3]

    def test_generate_returns_list(self):
        cfs = self._make_cfs()
        gen = self._make_gen_with_mock_backend(cfs)
        factual = _make_factual()
        current_premium = 1200.0
        actions = gen.generate(factual, target_premium=900.0, current_premium=current_premium)
        assert isinstance(actions, list)

    def test_generate_sorted_by_reduction(self):
        cfs = self._make_cfs()
        gen = self._make_gen_with_mock_backend(cfs)
        factual = _make_factual()
        actions = gen.generate(factual, target_premium=900.0, current_premium=1200.0)
        if len(actions) > 1:
            for i in range(len(actions) - 1):
                assert actions[i].premium_reduction >= actions[i + 1].premium_reduction

    def test_generate_max_monetary_cost_filter(self):
        cfs = self._make_cfs()
        gen = self._make_gen_with_mock_backend(cfs)
        factual = _make_factual()
        actions = gen.generate(
            factual, target_premium=900.0, current_premium=1200.0,
            max_monetary_cost=0.0
        )
        for action in actions:
            assert action.effort.monetary_cost <= 0.0

    def test_generate_max_days_filter(self):
        cfs = self._make_cfs()
        gen = self._make_gen_with_mock_backend(cfs)
        factual = _make_factual()
        actions = gen.generate(
            factual, target_premium=900.0, current_premium=1200.0,
            max_days=3.0
        )
        for action in actions:
            assert action.effort.time_days <= 3.0

    def test_generate_min_premium_reduction_filter(self):
        cfs = self._make_cfs()
        gen = self._make_gen_with_mock_backend(cfs)
        factual = _make_factual()
        actions = gen.generate(
            factual, target_premium=900.0, current_premium=1200.0,
            min_premium_reduction=500.0
        )
        for action in actions:
            assert action.premium_reduction >= 500.0

    def test_generate_deduplicates(self):
        factual = _make_factual()
        cf = factual.copy(); cf["vehicle_security"] = 2
        # Duplicate counterfactuals
        gen = self._make_gen_with_mock_backend([cf, cf, cf])
        actions = gen.generate(factual, target_premium=900.0, current_premium=1200.0)
        assert len(actions) <= 1

    def test_generate_empty_cfs_returns_empty(self):
        gen = self._make_gen_with_mock_backend([])
        factual = _make_factual()
        actions = gen.generate(factual, target_premium=900.0, current_premium=1200.0)
        assert actions == []


# ---------------------------------------------------------------------------
# DiCE backend import check
# ---------------------------------------------------------------------------

class TestDiceBackendImport:
    def test_dice_import_error_message(self):
        """When dice-ml is not installed, ImportError mentions install command."""
        model = _simple_model()
        graph = _simple_graph()
        cost_fn = _simple_cost()
        gen = RecourseGenerator(model, graph, cost_fn, backend="dice")

        with patch.dict(sys.modules, {"dice_ml": None}):
            with pytest.raises(ImportError, match="dice-ml"):
                gen._generate_dice(_make_factual(), 900.0)


class TestAlibiBackendImport:
    def test_alibi_import_error_message(self):
        model = _simple_model()
        graph = _simple_graph()
        cost_fn = _simple_cost()
        gen = RecourseGenerator(model, graph, cost_fn, backend="alibi_cfrl")

        with patch.dict(sys.modules, {"alibi": None}):
            with pytest.raises(ImportError, match="alibi"):
                gen._generate_alibi(_make_factual(), 900.0)


# ---------------------------------------------------------------------------
# FOCUS backend (sklearn trees — no external dependency)
# ---------------------------------------------------------------------------

class TestFocusBackend:
    def _make_tree_gen(self, n_counterfactuals: int = 3):
        from sklearn.tree import DecisionTreeRegressor

        factual = _make_factual()
        feature_names = list(factual.index)

        # Fit a tiny tree on synthetic data
        rng = np.random.default_rng(42)
        X = pd.DataFrame(
            rng.uniform(0, 1, (100, len(feature_names))),
            columns=feature_names,
        )
        y = 500 + 0.05 * X["annual_mileage"] - 100 * X["vehicle_security"]
        tree = DecisionTreeRegressor(max_depth=3, random_state=42)
        tree.fit(X, y)

        graph = _simple_graph()
        cost_fn = _simple_cost()
        return RecourseGenerator(tree, graph, cost_fn, backend="focus",
                                 n_counterfactuals=n_counterfactuals, random_state=42)

    def test_extract_trees_single_tree(self):
        from sklearn.tree import DecisionTreeRegressor
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.uniform(0, 1, (50, 3)), columns=["a", "b", "c"])
        y = X["a"] + X["b"]
        tree = DecisionTreeRegressor(max_depth=2).fit(X, y)
        graph = ActionabilityGraph({"a": FeatureConstraint("a"), "b": FeatureConstraint("b"),
                                    "c": FeatureConstraint("c")})
        cost_fn = InsuranceCostFunction()
        gen = RecourseGenerator(tree, graph, cost_fn, backend="focus")
        trees = gen._extract_trees()
        assert len(trees) == 1
        assert trees[0][0] == 1.0

    def test_extract_trees_random_forest(self):
        from sklearn.ensemble import RandomForestRegressor
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.uniform(0, 1, (50, 3)), columns=["a", "b", "c"])
        y = X["a"] + X["b"]
        rf = RandomForestRegressor(n_estimators=5, random_state=42).fit(X, y)
        graph = ActionabilityGraph({"a": FeatureConstraint("a"), "b": FeatureConstraint("b"),
                                    "c": FeatureConstraint("c")})
        cost_fn = InsuranceCostFunction()
        gen = RecourseGenerator(rf, graph, cost_fn, backend="focus")
        trees = gen._extract_trees()
        assert len(trees) == 5
        assert all(abs(w - 0.2) < 1e-9 for w, _ in trees)

    def test_extract_trees_gradient_boosting(self):
        from sklearn.ensemble import GradientBoostingRegressor
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.uniform(0, 1, (50, 3)), columns=["a", "b", "c"])
        y = X["a"] + X["b"]
        gbm = GradientBoostingRegressor(n_estimators=3, random_state=42).fit(X, y)
        graph = ActionabilityGraph({"a": FeatureConstraint("a"), "b": FeatureConstraint("b"),
                                    "c": FeatureConstraint("c")})
        cost_fn = InsuranceCostFunction()
        gen = RecourseGenerator(gbm, graph, cost_fn, backend="focus")
        trees = gen._extract_trees()
        assert len(trees) == 3  # one per stage for single output

    def test_focus_single_tree_prediction_close_to_sklearn(self):
        """FOCUS soft prediction should approximate sklearn hard prediction."""
        from sklearn.tree import DecisionTreeRegressor
        rng = np.random.default_rng(42)
        n_features = 6
        feature_names = list(_make_factual().index)
        X = rng.uniform(0, 1, (80, n_features))
        y = X[:, 0] * 500 + X[:, 1] * 200
        tree = DecisionTreeRegressor(max_depth=4, random_state=42).fit(X, y)
        graph = _simple_graph()
        cost_fn = _simple_cost()
        gen = RecourseGenerator(tree, graph, cost_fn, backend="focus")
        # With high sigma, soft prediction approximates hard prediction
        x = rng.uniform(0, 1, n_features)
        soft_pred, _ = gen._focus_single_tree(x, tree, sigma=100.0, feature_names=feature_names)
        hard_pred = float(tree.predict(x.reshape(1, -1))[0])
        assert abs(soft_pred - hard_pred) < abs(hard_pred) * 0.05 + 1.0

    def test_focus_single_tree_gradient_shape(self):
        from sklearn.tree import DecisionTreeRegressor
        rng = np.random.default_rng(1)
        n_features = 6
        feature_names = list(_make_factual().index)
        X = rng.uniform(0, 1, (50, n_features))
        y = X[:, 0] * 100
        tree = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X, y)
        graph = _simple_graph()
        cost_fn = _simple_cost()
        gen = RecourseGenerator(tree, graph, cost_fn, backend="focus")
        x = rng.uniform(0, 1, n_features)
        _, grad = gen._focus_single_tree(x, tree, sigma=10.0, feature_names=feature_names)
        assert grad.shape == (n_features,)

    def test_clip_to_constraints(self):
        model = _simple_model()
        graph = _simple_graph()
        cost_fn = _simple_cost()
        gen = RecourseGenerator(model, graph, cost_fn, backend="focus")
        factual = _make_factual()
        feature_names = list(factual.index)
        all_constraints = graph.all_constraints()
        x_factual = factual.to_numpy(dtype=float)

        # Make mileage go below min_value
        x_modified = x_factual.copy()
        mileage_idx = feature_names.index("annual_mileage")
        x_modified[mileage_idx] = 0.0  # below min_value=1000
        x_clipped = gen._clip_to_constraints(x_modified, x_factual, feature_names, all_constraints)
        assert x_clipped[mileage_idx] >= 1000.0

    def test_clip_to_constraints_immutable_unchanged(self):
        model = _simple_model()
        graph = _simple_graph()
        cost_fn = _simple_cost()
        gen = RecourseGenerator(model, graph, cost_fn, backend="focus")
        factual = _make_factual()
        feature_names = list(factual.index)
        all_constraints = graph.all_constraints()
        x_factual = factual.to_numpy(dtype=float)

        x_modified = x_factual.copy()
        age_idx = feature_names.index("age")
        x_modified[age_idx] = 999.0  # attempt to change immutable
        x_clipped = gen._clip_to_constraints(x_modified, x_factual, feature_names, all_constraints)
        assert x_clipped[age_idx] == x_factual[age_idx]

    def test_focus_generate_returns_n_counterfactuals(self):
        gen = self._make_tree_gen(n_counterfactuals=3)
        factual = _make_factual()
        # Normalise factual to [0,1] range since tree was trained on uniform
        factual_norm = pd.Series({k: 0.5 for k in factual.index})
        cfs = gen._generate_focus(factual_norm, target_premium=0.3)
        assert len(cfs) == 3

    def test_focus_generate_series_index_preserved(self):
        gen = self._make_tree_gen()
        factual = pd.Series({k: 0.5 for k in _make_factual().index})
        cfs = gen._generate_focus(factual, target_premium=0.3)
        for cf in cfs:
            assert list(cf.index) == list(factual.index)

    def test_synthesise_train_df_shape(self):
        model = _simple_model()
        graph = _simple_graph()
        cost_fn = _simple_cost()
        gen = RecourseGenerator(model, graph, cost_fn)
        factual_df = _make_factual().to_frame().T.reset_index(drop=True)
        train_df = gen._synthesise_train_df(factual_df, n=20)
        assert len(train_df) == 20
        assert set(factual_df.columns).issubset(set(train_df.columns))

    def test_synthesise_train_df_respects_allowed_values(self):
        model = _simple_model()
        graph = _simple_graph()
        cost_fn = _simple_cost()
        gen = RecourseGenerator(model, graph, cost_fn)
        factual_df = _make_factual().to_frame().T.reset_index(drop=True)
        train_df = gen._synthesise_train_df(factual_df, n=50)
        # vehicle_security has allowed_values=[0,1,2,3,4]
        allowed = {0, 1, 2, 3, 4}
        assert set(train_df["vehicle_security"].astype(int).unique()).issubset(allowed)


# ---------------------------------------------------------------------------
# Integration: dice backend (skipped if dice-ml not installed)
# ---------------------------------------------------------------------------

try:
    import dice_ml  # noqa: F401
    _DICE_AVAILABLE = True
except ImportError:
    _DICE_AVAILABLE = False


@pytest.mark.skipif(not _DICE_AVAILABLE, reason="dice-ml not installed")
class TestDiceIntegration:
    def test_dice_generate_returns_list(self):
        from sklearn.linear_model import Ridge
        graph = ActionabilityGraph.from_template("motor")
        cost_fn = InsuranceCostFunction.motor_defaults()

        features = [
            "age", "annual_mileage", "vehicle_security", "pass_plus",
            "garaging", "telematics", "occupation_risk",
            "postcode_risk", "crime_rate_decile", "flood_zone_risk",
            "vehicle_age", "vehicle_value", "licence_years",
            "at_fault_claims_3yr", "years_no_claims", "gender",
        ]
        rng = np.random.default_rng(42)
        X = pd.DataFrame(
            rng.uniform(0, 1, (200, len(features))),
            columns=features,
        )
        y = 1000 + rng.normal(0, 100, 200)
        model = Ridge().fit(X, y)

        gen = RecourseGenerator(model, graph, cost_fn, backend="dice", n_counterfactuals=3)
        factual = pd.Series({f: 0.5 for f in features})
        current_premium = float(model.predict(factual.to_frame().T)[0])
        actions = gen.generate(factual, target_premium=current_premium * 0.8,
                               current_premium=current_premium)
        assert isinstance(actions, list)
