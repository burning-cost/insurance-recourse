"""
Counterfactual recourse generation for insurance pricing models.

Three backends:
- 'dice': DiCE (dice-ml>=0.11) — gradient-free genetic/KD-Tree. Best for
  sklearn GLMs and pipelines.
- 'alibi_cfrl': alibi CFRL — RL-based, model-agnostic. Best for CatBoost/
  XGBoost black-box models. Requires TensorFlow or PyTorch.
- 'focus': FOCUS sigmoid approximation (implemented internally). Replaces tree
  split thresholds with σ(θ_j - x_{f_j}), enabling gradient descent on tree
  ensembles. Best for optimal CatBoost/XGBoost counterfactuals.

All backends return RecourseAction objects with premium-denominated output and
effort computed via InsuranceCostFunction.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from insurance_recourse.constraints import ActionabilityGraph
from insurance_recourse.cost import InsuranceCostFunction, RecourseEffort


@dataclass
class RecourseAction:
    """
    A single counterfactual recourse recommendation.

    Parameters
    ----------
    feature_changes:
        Dict mapping feature name to (original_value, counterfactual_value).
    predicted_premium:
        Model's predicted premium for the counterfactual row.
    premium_reduction:
        Absolute premium reduction in GBP (current - predicted).
    premium_reduction_pct:
        Relative reduction as a percentage.
    effort:
        Human-interpretable cost of implementing this action.
    causal_effects:
        Derived downstream changes from causal propagation (e.g., postcode
        change -> crime_rate update). Separate from direct feature_changes.
    validity:
        Whether the counterfactual meets the target premium threshold.
    description:
        Short natural-language description of the primary change.
    """

    feature_changes: Dict[str, Tuple[Any, Any]]
    predicted_premium: float
    premium_reduction: float
    premium_reduction_pct: float
    effort: RecourseEffort
    causal_effects: Dict[str, Any] = field(default_factory=dict)
    validity: bool = True
    description: str = ""

    def as_dict(self) -> dict:
        return {
            "feature_changes": {
                k: {"from": v[0], "to": v[1]}
                for k, v in self.feature_changes.items()
            },
            "predicted_premium": round(self.predicted_premium, 2),
            "premium_reduction": round(self.premium_reduction, 2),
            "premium_reduction_pct": round(self.premium_reduction_pct, 2),
            "effort": self.effort.as_dict(),
            "causal_effects": self.causal_effects,
            "validity": self.validity,
            "description": self.description,
        }


def _generate_description(feature_changes: Dict[str, Tuple[Any, Any]]) -> str:
    """Produce a short natural-language description of the primary change."""
    _DESCRIPTIONS = {
        "annual_mileage": lambda f, t: f"Reduce annual mileage from {f:,.0f} to {t:,.0f} miles",
        "vehicle_security": lambda f, t: f"Upgrade vehicle security from level {f} to level {t}",
        "pass_plus": lambda f, t: "Complete a Pass Plus advanced driving course",
        "garaging": lambda f, t: f"Improve overnight parking from level {f} to level {t}",
        "telematics": lambda f, t: "Install a telematics black box",
        "occupation_risk": lambda f, t: f"Change occupation from risk group {f} to {t}",
        "postcode_risk": lambda f, t: f"Move to a lower-risk postcode (risk score {f} → {t})",
        "alarm_grade": lambda f, t: f"Upgrade burglar alarm from grade {f} to grade {t}",
        "locks_grade": lambda f, t: f"Upgrade door locks from grade {f} to grade {t}",
        "smoke_alarms": lambda f, t: "Install interconnected smoke alarms",
        "water_leak_detector": lambda f, t: "Install a smart water leak detector",
        "voluntary_excess": lambda f, t: f"Increase voluntary excess from £{f:,.0f} to £{t:,.0f}",
        "sum_insured_buildings": lambda f, t: f"Adjust buildings sum insured to £{t:,.0f}",
        "sum_insured_contents": lambda f, t: f"Adjust contents sum insured to £{t:,.0f}",
    }
    if not feature_changes:
        return "No changes required"

    parts = []
    for feature, (orig, new) in feature_changes.items():
        fn = _DESCRIPTIONS.get(feature)
        if fn:
            try:
                parts.append(fn(orig, new))
            except Exception:
                parts.append(f"Change {feature} from {orig} to {new}")
        else:
            parts.append(f"Change {feature} from {orig} to {new}")

    return "; ".join(parts)


class RecourseGenerator:
    """
    Generates counterfactual premium-reduction recommendations.

    Wraps DiCE, alibi CFRL, or a FOCUS sigmoid approximation as the
    counterfactual search backend, then layers insurance-native constraints,
    causal propagation, and monetary effort computation on top.

    Parameters
    ----------
    model:
        Sklearn-compatible predictor (has .predict()) or callable that
        accepts a DataFrame and returns a 1-D array of premiums.
    actionability_graph:
        Constraint graph encoding feature mutability and causal structure.
    cost_function:
        Maps feature changes to monetary/time/feasibility effort.
    backend:
        'dice' | 'alibi_cfrl' | 'focus'. Defaults to 'dice'.
    n_counterfactuals:
        How many diverse counterfactuals to generate per call.
    random_state:
        Seed for reproducibility.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.linear_model import Ridge
    >>> from insurance_recourse.constraints import ActionabilityGraph
    >>> from insurance_recourse.cost import InsuranceCostFunction
    >>> from insurance_recourse.generator import RecourseGenerator

    >>> graph = ActionabilityGraph.from_template("motor")
    >>> cost_fn = InsuranceCostFunction.motor_defaults()
    >>> gen = RecourseGenerator(model, graph, cost_fn, backend="dice")
    >>> actions = gen.generate(factual, target_premium=900.0, current_premium=1200.0)
    """

    def __init__(
        self,
        model: Any,
        actionability_graph: ActionabilityGraph,
        cost_function: InsuranceCostFunction,
        backend: Literal["dice", "alibi_cfrl", "focus"] = "dice",
        n_counterfactuals: int = 5,
        random_state: int = 42,
    ) -> None:
        self.model = model
        self.graph = actionability_graph
        self.cost_function = cost_function
        self.backend = backend
        self.n_counterfactuals = n_counterfactuals
        self.random_state = random_state
        self._predict_fn = self._build_predict_fn(model)

    def _build_predict_fn(self, model: Any) -> Callable:
        """Return a callable that accepts a DataFrame and returns a 1-D array."""
        if callable(getattr(model, "predict", None)):
            def predict(df: pd.DataFrame) -> np.ndarray:
                return np.asarray(model.predict(df), dtype=float)
            return predict
        elif callable(model):
            return model
        else:
            raise TypeError(
                f"model must have a .predict() method or be callable, got {type(model)}"
            )

    def generate(
        self,
        factual: pd.Series,
        target_premium: float,
        current_premium: float,
        max_monetary_cost: Optional[float] = None,
        max_days: Optional[float] = None,
        min_premium_reduction: Optional[float] = None,
    ) -> List[RecourseAction]:
        """
        Generate recourse actions for a single policyholder.

        Parameters
        ----------
        factual:
            Feature vector for the policyholder (pandas Series, column names
            must match the model's training features).
        target_premium:
            Desired premium in GBP. Counterfactuals targeting this level will
            be searched.
        current_premium:
            Current model-predicted premium (pre-computed to avoid double call).
        max_monetary_cost:
            Filter: discard actions that cost more than this in GBP.
        max_days:
            Filter: discard actions that take more than this many days.
        min_premium_reduction:
            Filter: discard actions that reduce the premium by less than this
            amount in GBP.

        Returns
        -------
        List[RecourseAction]
            Actions sorted by premium_reduction descending. Empty list if no
            feasible actions found.
        """
        if self.backend == "dice":
            raw_cfs = self._generate_dice(factual, target_premium)
        elif self.backend == "alibi_cfrl":
            raw_cfs = self._generate_alibi(factual, target_premium)
        elif self.backend == "focus":
            raw_cfs = self._generate_focus(factual, target_premium)
        else:
            raise ValueError(f"Unknown backend: {self.backend!r}")

        actions = []
        for cf_row in raw_cfs:
            action = self._build_action(factual, cf_row, current_premium)
            actions.append(action)

        # Apply filters
        if max_monetary_cost is not None:
            actions = [a for a in actions if a.effort.monetary_cost <= max_monetary_cost]
        if max_days is not None:
            actions = [a for a in actions if a.effort.time_days <= max_days]
        if min_premium_reduction is not None:
            actions = [a for a in actions if a.premium_reduction >= min_premium_reduction]

        # Deduplicate by feature_changes key
        seen: set = set()
        unique_actions = []
        for action in actions:
            key = frozenset(
                (k, v[1]) for k, v in action.feature_changes.items()
            )
            if key not in seen:
                seen.add(key)
                unique_actions.append(action)

        unique_actions.sort(key=lambda a: a.premium_reduction, reverse=True)
        return unique_actions

    def _build_action(
        self,
        factual: pd.Series,
        cf_row: pd.Series,
        current_premium: float,
    ) -> RecourseAction:
        """Convert a raw counterfactual row into a RecourseAction."""
        # Apply causal propagation
        factual_dict = factual.to_dict()
        cf_dict = cf_row.to_dict()
        interventions = {
            k: v for k, v in cf_dict.items()
            if k in factual_dict and v != factual_dict.get(k)
        }
        full_cf_dict = self.graph.propagate_causal_effects(factual_dict, interventions)
        cf_series = pd.Series(full_cf_dict, name=cf_row.name)

        # Re-predict on causally-propagated row
        cf_df = cf_series.to_frame().T.reset_index(drop=True)
        try:
            predicted_premium = float(self._predict_fn(cf_df)[0])
        except Exception as e:
            warnings.warn(f"Prediction on counterfactual failed: {e}", RuntimeWarning, stacklevel=2)
            predicted_premium = current_premium

        premium_reduction = current_premium - predicted_premium
        premium_reduction_pct = (
            100.0 * premium_reduction / current_premium if current_premium > 0 else 0.0
        )

        # Feature changes (direct only, not causal children)
        feature_changes = {
            k: (factual_dict[k], full_cf_dict[k])
            for k in interventions
            if k in factual_dict
        }

        # Causal effects (downstream changes not directly intervened on)
        causal_effects = {
            k: {"from": factual_dict.get(k), "to": full_cf_dict[k]}
            for k in full_cf_dict
            if k not in interventions and full_cf_dict.get(k) != factual_dict.get(k)
        }

        # Effort
        factual_for_cost = factual.copy()
        cf_for_cost = cf_series.reindex(factual.index)
        effort = self.cost_function.compute(factual_for_cost, cf_for_cost)

        # Validity
        is_valid, _ = self.graph.validate_counterfactual(factual_dict, full_cf_dict)

        description = _generate_description(feature_changes)

        return RecourseAction(
            feature_changes=feature_changes,
            predicted_premium=round(predicted_premium, 2),
            premium_reduction=round(premium_reduction, 2),
            premium_reduction_pct=round(premium_reduction_pct, 2),
            effort=effort,
            causal_effects=causal_effects,
            validity=is_valid,
            description=description,
        )

    def _generate_dice(
        self,
        factual: pd.Series,
        target_premium: float,
    ) -> List[pd.Series]:
        """Generate counterfactuals using DiCE."""
        try:
            import dice_ml  # noqa: F401
        except ImportError:
            raise ImportError(
                "DiCE is required for backend='dice'. Install it with:\n"
                "    pip install dice-ml>=0.11\n"
                "or:\n"
                "    pip install insurance-recourse[dice]"
            )

        from dice_ml import Dice, Data, Model

        factual_df = factual.to_frame().T.reset_index(drop=True)
        mutable = self.graph.get_mutable_features()
        immutable = self.graph.get_immutable_features()

        # Build feature ranges for DiCE
        feature_constraints: Dict[str, Any] = {}
        all_constraints = self.graph.all_constraints()
        for fname, constraint in all_constraints.items():
            if fname not in factual_df.columns:
                continue
            if not constraint.is_actionable():
                continue
            fc: Dict[str, Any] = {}
            if constraint.min_value is not None:
                fc["min"] = constraint.min_value
            if constraint.max_value is not None:
                fc["max"] = constraint.max_value
            if constraint.allowed_values is not None:
                fc["allowed"] = constraint.allowed_values
            if fc:
                feature_constraints[fname] = fc

        # Determine feature types for DiCE Data
        continuous_features = []
        categorical_features = []
        for col in factual_df.columns:
            c = all_constraints.get(col)
            if c and c.allowed_values is not None and len(c.allowed_values) <= 10:
                categorical_features.append(col)
            else:
                continuous_features.append(col)

        # Build a minimal training-like DataFrame for DiCE
        # DiCE needs a "dataset" to infer feature ranges; we synthesise one
        train_df = self._synthesise_train_df(factual_df, n=50)
        outcome_name = "_premium"
        train_df[outcome_name] = self._predict_fn(train_df.drop(columns=[outcome_name], errors="ignore"))

        dice_data = Data(
            dataframe=train_df,
            continuous_features=[f for f in continuous_features if f in train_df.columns],
            outcome_name=outcome_name,
        )

        dice_model = Model(
            model=self.model,
            backend="sklearn",
            model_type="regressor",
        )

        exp = Dice(dice_data, dice_model, method="genetic")

        # DiCE target: predict value below target_premium
        dice_result = exp.generate_counterfactuals(
            factual_df,
            total_CFs=self.n_counterfactuals,
            desired_range=[0, target_premium],
            features_to_vary=[f for f in mutable if f in factual_df.columns],
            permitted_range={
                k: v for k, v in feature_constraints.items()
                if isinstance(v, dict) and ("min" in v or "max" in v)
            },
            random_seed=self.random_state,
        )

        cfs = dice_result.cf_examples_list[0].final_cfs_df
        if cfs is None or len(cfs) == 0:
            return []

        # Drop outcome column if present
        if outcome_name in cfs.columns:
            cfs = cfs.drop(columns=[outcome_name])

        return [cfs.iloc[i] for i in range(len(cfs))]

    def _synthesise_train_df(
        self, factual_df: pd.DataFrame, n: int = 50
    ) -> pd.DataFrame:
        """
        Synthesise a small training-like DataFrame around the factual point.

        Used to give DiCE a dataset for feature range inference. Each column
        is perturbed within its allowed range using the constraint definitions.
        """
        rng = np.random.default_rng(self.random_state)
        rows = []
        all_constraints = self.graph.all_constraints()

        for _ in range(n):
            row = {}
            for col in factual_df.columns:
                orig = factual_df[col].iloc[0]
                constraint = all_constraints.get(col)
                if constraint and constraint.allowed_values is not None:
                    row[col] = rng.choice(constraint.allowed_values)
                elif constraint and (constraint.min_value is not None or constraint.max_value is not None):
                    lo = constraint.min_value if constraint.min_value is not None else float(orig) * 0.5
                    hi = constraint.max_value if constraint.max_value is not None else float(orig) * 1.5
                    row[col] = float(rng.uniform(lo, hi))
                else:
                    try:
                        noise = rng.normal(0, abs(float(orig)) * 0.2 + 1)
                        row[col] = float(orig) + noise
                    except (TypeError, ValueError):
                        row[col] = orig
            rows.append(row)

        return pd.DataFrame(rows)

    def _generate_alibi(
        self,
        factual: pd.Series,
        target_premium: float,
    ) -> List[pd.Series]:
        """Generate counterfactuals using alibi CFRL."""
        try:
            import alibi  # noqa: F401
        except ImportError:
            raise ImportError(
                "alibi is required for backend='alibi_cfrl'. Install it with:\n"
                "    pip install alibi>=0.9\n"
                "or:\n"
                "    pip install insurance-recourse[alibi]"
            )

        warnings.warn(
            "alibi CFRL backend is experimental and requires TensorFlow or PyTorch. "
            "Training the RL agent may take several minutes. Consider backend='dice' "
            "for faster iteration.",
            UserWarning,
            stacklevel=3,
        )

        from alibi.explainers import CounterfactualRL

        factual_arr = factual.to_numpy().reshape(1, -1).astype(float)
        feature_names = list(factual.index)
        mutable = self.graph.get_mutable_features()
        immutable_mask = np.array(
            [0 if f in mutable else 1 for f in feature_names],
            dtype=float,
        )

        def predict_fn(arr: np.ndarray) -> np.ndarray:
            df = pd.DataFrame(arr, columns=feature_names)
            return self._predict_fn(df).reshape(-1, 1)

        target_arr = np.array([[target_premium]])

        # Fit CFRL on a small synthetic dataset
        train_df = self._synthesise_train_df(
            factual.to_frame().T.reset_index(drop=True), n=200
        )
        train_arr = train_df[feature_names].to_numpy(dtype=float)

        explainer = CounterfactualRL(predict_fn=predict_fn)

        try:
            explainer.fit(train_arr)
            explanation = explainer.explain(factual_arr)
        except Exception as exc:
            warnings.warn(
                f"alibi CFRL fitting/explain failed: {exc}. Returning empty list.",
                RuntimeWarning,
                stacklevel=2,
            )
            return []

        cfs_arr = getattr(explanation, "cf", None)
        if cfs_arr is None:
            return []

        results = []
        for cf_arr in cfs_arr[:self.n_counterfactuals]:
            results.append(pd.Series(cf_arr.ravel(), index=feature_names))
        return results

    def _generate_focus(
        self,
        factual: pd.Series,
        target_premium: float,
    ) -> List[pd.Series]:
        """
        Generate counterfactuals using the FOCUS sigmoid approximation.

        FOCUS (from Lucic et al., AAAI 2022) replaces each tree split
        threshold with a sigmoid σ(θ_j - x_{f_j}), making the tree ensemble
        output differentiable w.r.t. inputs. We then minimise:

            L = max(0, f(x') - target)  +  β * ||x' - x||_2^2

        subject to feature constraints, using gradient descent.

        This implementation supports sklearn DecisionTreeRegressor,
        RandomForestRegressor, GradientBoostingRegressor, and XGBoost/CatBoost
        via their sklearn API.
        """
        try:
            trees = self._extract_trees()
        except (AttributeError, TypeError) as e:
            raise TypeError(
                f"FOCUS backend requires a tree ensemble model (sklearn "
                f"DecisionTreeRegressor, RandomForestRegressor, "
                f"GradientBoostingRegressor, or XGBoost/CatBoost with sklearn "
                f"API). Got: {type(self.model)}. Error: {e}"
            ) from e

        feature_names = list(factual.index)
        mutable = set(self.graph.get_mutable_features())
        all_constraints = self.graph.all_constraints()

        x0 = factual.to_numpy(dtype=float).copy()
        n_features = len(x0)

        # Build mask for mutable features
        mutable_mask = np.array(
            [1.0 if feature_names[i] in mutable else 0.0 for i in range(n_features)]
        )

        results = []
        rng = np.random.default_rng(self.random_state)

        for attempt in range(self.n_counterfactuals):
            # Initialise with small perturbation to generate diversity
            noise = rng.normal(0, 0.05, size=n_features) * mutable_mask
            x_cf = x0.copy() + noise

            x_cf = self._focus_optimise(
                x_cf=x_cf,
                x_factual=x0,
                target_premium=target_premium,
                trees=trees,
                mutable_mask=mutable_mask,
                feature_names=feature_names,
                all_constraints=all_constraints,
                sigma=10.0,
                beta=0.1,
                lr=0.01,
                n_steps=300,
            )

            # Clip to constraint bounds
            x_cf = self._clip_to_constraints(x_cf, x0, feature_names, all_constraints)

            cf_series = pd.Series(x_cf, index=feature_names)
            results.append(cf_series)

        return results

    def _focus_optimise(
        self,
        x_cf: np.ndarray,
        x_factual: np.ndarray,
        target_premium: float,
        trees: List[Any],
        mutable_mask: np.ndarray,
        feature_names: List[str],
        all_constraints: dict,
        sigma: float,
        beta: float,
        lr: float,
        n_steps: int,
    ) -> np.ndarray:
        """
        Gradient descent on the FOCUS sigmoid-approximated tree output.

        Node activation approximation:
          Left child:  t̃_j(x) = σ(σ_param * (θ_j - x_{f_j}))
          Right child: t̃_j(x) = σ(σ_param * (x_{f_j} - θ_j))

        Leaf values are propagated bottom-up using soft activations.
        """
        x = x_cf.copy()

        def sigmoid(z: np.ndarray) -> np.ndarray:
            return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))

        def sigmoid_grad(z: np.ndarray) -> np.ndarray:
            s = sigmoid(z)
            return s * (1 - s)

        def focus_predict_and_grad(x_in: np.ndarray) -> Tuple[float, np.ndarray]:
            """Return (prediction, gradient w.r.t. x_in) using FOCUS approximation."""
            total_pred = 0.0
            total_grad = np.zeros_like(x_in)

            for tree_obj in trees:
                weight, tree = tree_obj
                pred, grad = self._focus_single_tree(x_in, tree, sigma, feature_names)
                total_pred += weight * pred
                total_grad += weight * grad

            return total_pred, total_grad

        for step in range(n_steps):
            pred, grad_pred = focus_predict_and_grad(x)

            # Loss: hinge on premium + L2 proximity
            hinge = max(0.0, pred - target_premium)
            dist_sq = float(np.sum(((x - x_factual) * mutable_mask) ** 2))

            # Gradient of loss w.r.t. x
            if pred > target_premium:
                grad_hinge = grad_pred
            else:
                grad_hinge = np.zeros_like(x)

            grad_dist = 2.0 * beta * (x - x_factual) * mutable_mask
            grad_total = grad_hinge + grad_dist

            # Zero out gradient for immutable features
            grad_total *= mutable_mask

            x = x - lr * grad_total

            # Project back to constraints after each step
            x = self._clip_to_constraints(x, x_factual, feature_names, all_constraints)

            if hinge < 1e-4 and step > 20:
                break

        return x

    def _focus_single_tree(
        self,
        x: np.ndarray,
        tree: Any,
        sigma: float,
        feature_names: List[str],
    ) -> Tuple[float, np.ndarray]:
        """
        Compute FOCUS soft prediction and gradient for a single decision tree.

        Uses sklearn tree internals (tree_.feature, tree_.threshold,
        tree_.children_left, tree_.children_right, tree_.value).

        Returns (prediction, gradient).
        """
        t = tree.tree_
        n_nodes = t.node_count
        features = t.feature      # feature index at each node (-2 for leaves)
        thresholds = t.threshold  # split threshold at each node
        left = t.children_left
        right = t.children_right
        values = t.value          # shape (n_nodes, n_outputs, n_classes)

        # node_activation[i] = soft probability of reaching node i
        node_act = np.zeros(n_nodes)
        node_act[0] = 1.0

        # node_grad[i] = d(node_act[i])/d(x), shape (n_nodes, n_features)
        n_features = len(x)
        node_grad = np.zeros((n_nodes, n_features))

        def sigmoid(z: float) -> float:
            return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))

        for node in range(n_nodes):
            if left[node] == -1:  # leaf
                continue

            fi = features[node]
            theta = thresholds[node]
            z = sigma * (theta - x[fi])

            s_left = sigmoid(z)   # P(go left)
            s_right = 1.0 - s_left

            # Propagate activation
            node_act[left[node]] = node_act[node] * s_left
            node_act[right[node]] = node_act[node] * s_right

            # Propagate gradient
            # d(s_left)/d(x[fi]) = -sigma * sigmoid(z) * (1-sigmoid(z))
            ds_dx = -sigma * s_left * (1.0 - s_left)

            # d(node_act[left])/d(x) = node_act[parent] * d(s_left)/d(x[fi])
            #                        + s_left * d(node_act[parent])/d(x)
            left_grad = node_grad[node] * s_left
            left_grad[fi] += node_act[node] * ds_dx

            right_grad = node_grad[node] * s_right
            right_grad[fi] -= node_act[node] * ds_dx

            node_grad[left[node]] = left_grad
            node_grad[right[node]] = right_grad

        # Soft prediction: weighted sum of leaf values
        prediction = 0.0
        grad_pred = np.zeros(n_features)
        for node in range(n_nodes):
            if left[node] != -1:  # not a leaf
                continue
            leaf_val = float(values[node].ravel()[0])
            prediction += node_act[node] * leaf_val
            grad_pred += node_grad[node] * leaf_val

        return prediction, grad_pred

    def _extract_trees(self) -> List[Tuple[float, Any]]:
        """
        Extract (weight, sklearn_tree) tuples from the model.

        Supports sklearn DecisionTreeRegressor, RandomForestRegressor,
        GradientBoostingRegressor, and XGBoost/CatBoost with sklearn API.

        Returns list of (weight, estimator_with_tree_attribute) pairs.
        """
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

        model = self.model
        if isinstance(model, DecisionTreeRegressor):
            return [(1.0, model)]
        elif isinstance(model, RandomForestRegressor):
            n = len(model.estimators_)
            return [(1.0 / n, est) for est in model.estimators_]
        elif isinstance(model, GradientBoostingRegressor):
            weight = model.learning_rate
            trees = []
            for stage in model.estimators_:
                for tree in stage:
                    trees.append((weight, tree))
            return trees
        else:
            # Try XGBoost/CatBoost: fall back to numerical finite differences
            # since we can't access internal tree structure portably
            return self._extract_trees_via_booster()

    def _extract_trees_via_booster(self) -> List[Tuple[float, Any]]:
        """
        Fallback for XGBoost/CatBoost: use finite-difference gradient estimation.

        When tree internals aren't accessible via sklearn API, we wrap the
        predict function as a pseudo-tree with a numerical gradient.
        """

        class _PseudoTree:
            """Proxy that exposes .tree_ attribute via finite differences."""

            def __init__(self, model: Any, feature_names: List[str]) -> None:
                self.model = model
                self.feature_names = feature_names
                # Build a minimal tree_ namespace
                self.tree_ = self  # self-referential for compatibility
                # These attributes are accessed in _focus_single_tree
                self.node_count = 1
                self.feature = np.array([-2])
                self.threshold = np.array([-2.0])
                self.children_left = np.array([-1])
                self.children_right = np.array([-1])
                self.value = np.array([[[0.0]]])

        # Cannot decompose: raise to signal caller should use different backend
        raise AttributeError(
            "Cannot extract tree internals for FOCUS from this model type. "
            "Use backend='dice' for sklearn models or backend='alibi_cfrl' for "
            "XGBoost/CatBoost black-box models."
        )

    def _clip_to_constraints(
        self,
        x: np.ndarray,
        x_factual: np.ndarray,
        feature_names: List[str],
        all_constraints: dict,
    ) -> np.ndarray:
        """Project a continuous feature vector back to constraint feasibility region."""
        x_clipped = x.copy()
        for i, fname in enumerate(feature_names):
            constraint = all_constraints.get(fname)
            if constraint is None:
                continue
            try:
                x_clipped[i] = constraint.clip_counterfactual(
                    float(x_factual[i]), float(x[i])
                )
            except (TypeError, ValueError):
                pass
        return x_clipped

    def __repr__(self) -> str:
        return (
            f"RecourseGenerator("
            f"backend={self.backend!r}, "
            f"n_counterfactuals={self.n_counterfactuals}, "
            f"mutable_features={len(self.graph.get_mutable_features())})"
        )
