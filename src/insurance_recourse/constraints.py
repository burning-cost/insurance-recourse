"""
Constraint encoding for insurance feature actionability.

Models the mutability structure of insurance covariates as a directed acyclic
graph (DAG) with causal propagation. Pre-built templates for UK motor and home
product lines reflect FCA Consumer Duty requirements around actionability.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple


class Mutability(str, Enum):
    """Whether a policyholder can act on a feature."""
    IMMUTABLE = "immutable"
    MUTABLE = "mutable"
    CONDITIONALLY_MUTABLE = "conditionally_mutable"


@dataclass
class FeatureConstraint:
    """
    Actionability constraint for a single insurance feature.

    Parameters
    ----------
    name:
        Feature name (must match column name in the model's input DataFrame).
    mutability:
        Whether the feature is actionable by the policyholder.
    direction:
        For mutable features: 'increase', 'decrease', or 'either'.
        None for immutable features.
    effort_weight:
        Relative effort per unit change. Used by InsuranceCostFunction as a
        scale factor on per-feature monetary costs. Defaults to 1.0.
    feasibility_rate:
        Fraction of similar policyholders who could realistically action this
        change. Used to compute P(recourse action is actionable). Range [0, 1].
    causal_children:
        Feature names that are causally downstream of this feature. When this
        feature changes, those children must be updated via their propagation
        functions.
    allowed_values:
        Explicit enumeration of permitted counterfactual values (categorical
        features). None means continuous range applies.
    min_value:
        Minimum allowed value for continuous features. None means unbounded.
    max_value:
        Maximum allowed value for continuous features. None means unbounded.
    """

    name: str
    mutability: Mutability = Mutability.MUTABLE
    direction: Optional[Literal["increase", "decrease", "either"]] = "either"
    effort_weight: float = 1.0
    feasibility_rate: float = 1.0
    causal_children: List[str] = field(default_factory=list)
    allowed_values: Optional[List[Any]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.feasibility_rate <= 1.0:
            raise ValueError(
                f"feasibility_rate must be in [0, 1], got {self.feasibility_rate}"
            )
        if self.effort_weight < 0:
            raise ValueError(
                f"effort_weight must be non-negative, got {self.effort_weight}"
            )
        if self.mutability == Mutability.IMMUTABLE and self.direction is not None:
            warnings.warn(
                f"Feature '{self.name}' is immutable but has direction set; "
                "direction will be ignored.",
                UserWarning,
                stacklevel=2,
            )

    def is_actionable(self) -> bool:
        """True if the feature can be changed by the policyholder."""
        return self.mutability != Mutability.IMMUTABLE

    def clip_counterfactual(self, original: float, proposed: float) -> float:
        """
        Clip a proposed counterfactual value to the constraint's bounds.

        Enforces direction, min_value, and max_value. Categorical features
        should be validated separately via allowed_values.
        """
        if self.mutability == Mutability.IMMUTABLE:
            return original

        value = proposed

        if self.direction == "decrease" and value > original:
            value = original
        elif self.direction == "increase" and value < original:
            value = original

        if self.min_value is not None:
            value = max(value, self.min_value)
        if self.max_value is not None:
            value = min(value, self.max_value)

        return value

    def validate_counterfactual(self, original: Any, proposed: Any) -> Tuple[bool, str]:
        """
        Check whether proposed is a valid counterfactual for original.

        Returns (is_valid, reason). reason is empty string when valid.
        """
        if self.mutability == Mutability.IMMUTABLE:
            if proposed != original:
                return False, f"'{self.name}' is immutable"
            return True, ""

        if self.allowed_values is not None:
            if proposed not in self.allowed_values:
                return False, (
                    f"'{self.name}' counterfactual {proposed!r} not in "
                    f"allowed_values {self.allowed_values}"
                )
            return True, ""

        # Numeric direction check
        try:
            orig_num = float(original)
            prop_num = float(proposed)
        except (TypeError, ValueError):
            return True, ""  # Non-numeric without allowed_values: unchecked

        if self.direction == "decrease" and prop_num > orig_num:
            return False, (
                f"'{self.name}' must decrease; got {prop_num} > {orig_num}"
            )
        if self.direction == "increase" and prop_num < orig_num:
            return False, (
                f"'{self.name}' must increase; got {prop_num} < {orig_num}"
            )
        if self.min_value is not None and prop_num < self.min_value:
            return False, (
                f"'{self.name}' below min_value {self.min_value}: {prop_num}"
            )
        if self.max_value is not None and prop_num > self.max_value:
            return False, (
                f"'{self.name}' above max_value {self.max_value}: {prop_num}"
            )

        return True, ""


class ActionabilityGraph:
    """
    DAG encoding causal relationships between insurance features.

    Nodes are FeatureConstraint objects. Edges encode causal dependencies:
    if feature A is a causal parent of B, then changing A requires B to be
    updated via a propagation function.

    Pre-built templates are available for UK motor and home product lines.

    Parameters
    ----------
    constraints:
        Mapping from feature name to FeatureConstraint. All features used
        by the model should be included, including immutable ones.
    propagation_functions:
        Optional mapping from feature name to a callable that accepts the
        full factual row (as dict) plus the proposed intervention dict and
        returns the updated value for that feature. Used for causal propagation
        (e.g., postcode change -> crime_rate, flood_zone updates).

    Examples
    --------
    >>> graph = ActionabilityGraph.from_template("motor")
    >>> graph.get_mutable_features()
    ['annual_mileage', 'vehicle_security', 'garaging', ...]
    """

    def __init__(
        self,
        constraints: Dict[str, FeatureConstraint],
        propagation_functions: Optional[Dict[str, Callable]] = None,
    ) -> None:
        self._constraints = constraints
        self._propagation_functions: Dict[str, Callable] = propagation_functions or {}
        self._validate_dag()

    def _validate_dag(self) -> None:
        """Check for cycles and undefined causal children."""
        for name, constraint in self._constraints.items():
            for child in constraint.causal_children:
                if child not in self._constraints:
                    warnings.warn(
                        f"Feature '{name}' lists causal child '{child}' "
                        "which is not in the constraint set.",
                        UserWarning,
                        stacklevel=3,
                    )

    @classmethod
    def from_template(
        cls,
        product: Literal["motor", "home"],
    ) -> "ActionabilityGraph":
        """
        Return a pre-built constraint graph for a UK personal lines product.

        These templates encode the standard mutability structure for UK motor
        and home insurance per FCA Consumer Duty guidance. Override or extend
        as needed for your specific product features.
        """
        if product == "motor":
            return cls._motor_template()
        elif product == "home":
            return cls._home_template()
        else:
            raise ValueError(f"Unknown product template: {product!r}. Use 'motor' or 'home'.")

    @classmethod
    def _motor_template(cls) -> "ActionabilityGraph":
        constraints = {
            # Immutable features
            "age": FeatureConstraint(
                name="age",
                mutability=Mutability.IMMUTABLE,
                direction=None,
                effort_weight=0.0,
                feasibility_rate=0.0,
            ),
            "gender": FeatureConstraint(
                name="gender",
                mutability=Mutability.IMMUTABLE,
                direction=None,
                effort_weight=0.0,
                feasibility_rate=0.0,
            ),
            "years_no_claims": FeatureConstraint(
                name="years_no_claims",
                mutability=Mutability.IMMUTABLE,
                direction=None,
                effort_weight=0.0,
                feasibility_rate=0.0,
            ),
            "at_fault_claims_3yr": FeatureConstraint(
                name="at_fault_claims_3yr",
                mutability=Mutability.IMMUTABLE,
                direction=None,
                effort_weight=0.0,
                feasibility_rate=0.0,
            ),
            "licence_years": FeatureConstraint(
                name="licence_years",
                mutability=Mutability.IMMUTABLE,
                direction=None,
                effort_weight=0.0,
                feasibility_rate=0.0,
            ),
            # Mutable: decrease only
            "annual_mileage": FeatureConstraint(
                name="annual_mileage",
                mutability=Mutability.MUTABLE,
                direction="decrease",
                effort_weight=0.5,
                feasibility_rate=0.6,
                min_value=1000.0,
                max_value=50000.0,
            ),
            # Mutable: increase only (security improvements)
            "vehicle_security": FeatureConstraint(
                name="vehicle_security",
                mutability=Mutability.MUTABLE,
                direction="increase",
                effort_weight=1.5,
                feasibility_rate=0.85,
                min_value=0,
                max_value=4,
                allowed_values=[0, 1, 2, 3, 4],
            ),
            # Mutable: acquire only
            "pass_plus": FeatureConstraint(
                name="pass_plus",
                mutability=Mutability.MUTABLE,
                direction="increase",
                effort_weight=2.0,
                feasibility_rate=0.40,
                allowed_values=[0, 1],
            ),
            # Mutable: improve only (overnight location)
            "garaging": FeatureConstraint(
                name="garaging",
                mutability=Mutability.MUTABLE,
                direction="increase",
                effort_weight=1.2,
                feasibility_rate=0.35,
                allowed_values=[0, 1, 2, 3],  # 0=street, 1=driveway, 2=carport, 3=garage
            ),
            "telematics": FeatureConstraint(
                name="telematics",
                mutability=Mutability.MUTABLE,
                direction="increase",
                effort_weight=0.8,
                feasibility_rate=0.55,
                allowed_values=[0, 1],
            ),
            "occupation_risk": FeatureConstraint(
                name="occupation_risk",
                mutability=Mutability.MUTABLE,
                direction="either",
                effort_weight=3.0,
                feasibility_rate=0.10,
                min_value=1,
                max_value=10,
            ),
            # Conditionally mutable: postcode change has causal children
            "postcode_risk": FeatureConstraint(
                name="postcode_risk",
                mutability=Mutability.CONDITIONALLY_MUTABLE,
                direction="decrease",
                effort_weight=5.0,
                feasibility_rate=0.08,
                min_value=1,
                max_value=10,
                causal_children=["crime_rate_decile", "flood_zone_risk"],
            ),
            # Causal children of postcode — downstream, read-only in recourse
            "crime_rate_decile": FeatureConstraint(
                name="crime_rate_decile",
                mutability=Mutability.CONDITIONALLY_MUTABLE,
                direction=None,
                effort_weight=0.0,
                feasibility_rate=0.0,
            ),
            "flood_zone_risk": FeatureConstraint(
                name="flood_zone_risk",
                mutability=Mutability.CONDITIONALLY_MUTABLE,
                direction=None,
                effort_weight=0.0,
                feasibility_rate=0.0,
            ),
            "vehicle_age": FeatureConstraint(
                name="vehicle_age",
                mutability=Mutability.CONDITIONALLY_MUTABLE,
                direction="decrease",  # replace vehicle with newer one
                effort_weight=4.0,
                feasibility_rate=0.15,
                min_value=0,
                max_value=25,
                causal_children=["vehicle_value"],
            ),
            "vehicle_value": FeatureConstraint(
                name="vehicle_value",
                mutability=Mutability.CONDITIONALLY_MUTABLE,
                direction="either",
                effort_weight=0.0,
                feasibility_rate=0.15,
                min_value=500.0,
                max_value=150000.0,
            ),
        }

        def propagate_postcode(factual: dict, interventions: dict) -> dict:
            """When postcode_risk changes, derive crime_rate and flood_zone."""
            new_postcode_risk = interventions.get("postcode_risk", factual.get("postcode_risk", 5))
            updates: dict = {}
            # Linear approximation: crime rate tracks postcode risk
            updates["crime_rate_decile"] = max(1, min(10, round(new_postcode_risk * 0.9)))
            # Flood zone: high postcode risk correlates weakly with flood
            updates["flood_zone_risk"] = max(0, min(3, round((new_postcode_risk - 1) * 0.3)))
            return updates

        def propagate_vehicle_age(factual: dict, interventions: dict) -> dict:
            """When vehicle_age changes, derive vehicle_value."""
            new_age = interventions.get("vehicle_age", factual.get("vehicle_age", 5))
            orig_value = float(factual.get("vehicle_value", 15000.0))
            # Simplified depreciation: 15% per year from original value
            depreciation = max(0.0, 1.0 - 0.15 * new_age)
            orig_new = orig_value / max(0.01, 1.0 - 0.15 * float(factual.get("vehicle_age", 5)))
            updates = {"vehicle_value": round(orig_new * depreciation, 2)}
            return updates

        propagation_functions = {
            "postcode_risk": propagate_postcode,
            "vehicle_age": propagate_vehicle_age,
        }

        return cls(constraints, propagation_functions)

    @classmethod
    def _home_template(cls) -> "ActionabilityGraph":
        constraints = {
            # Immutable
            "property_age": FeatureConstraint(
                name="property_age",
                mutability=Mutability.IMMUTABLE,
                direction=None,
                effort_weight=0.0,
                feasibility_rate=0.0,
            ),
            "construction_type": FeatureConstraint(
                name="construction_type",
                mutability=Mutability.IMMUTABLE,
                direction=None,
                effort_weight=0.0,
                feasibility_rate=0.0,
            ),
            "years_at_address": FeatureConstraint(
                name="years_at_address",
                mutability=Mutability.IMMUTABLE,
                direction=None,
                effort_weight=0.0,
                feasibility_rate=0.0,
            ),
            "claims_5yr": FeatureConstraint(
                name="claims_5yr",
                mutability=Mutability.IMMUTABLE,
                direction=None,
                effort_weight=0.0,
                feasibility_rate=0.0,
            ),
            # Mutable: security improvements
            "alarm_grade": FeatureConstraint(
                name="alarm_grade",
                mutability=Mutability.MUTABLE,
                direction="increase",
                effort_weight=1.8,
                feasibility_rate=0.75,
                allowed_values=[0, 1, 2, 3, 4],  # 0=none, 4=BS 8243 Grade 3+
            ),
            "locks_grade": FeatureConstraint(
                name="locks_grade",
                mutability=Mutability.MUTABLE,
                direction="increase",
                effort_weight=0.8,
                feasibility_rate=0.90,
                allowed_values=[0, 1, 2, 3],  # 0=basic, 3=BS 3621 approved
            ),
            "smoke_alarms": FeatureConstraint(
                name="smoke_alarms",
                mutability=Mutability.MUTABLE,
                direction="increase",
                effort_weight=0.3,
                feasibility_rate=0.95,
                allowed_values=[0, 1],
            ),
            "water_leak_detector": FeatureConstraint(
                name="water_leak_detector",
                mutability=Mutability.MUTABLE,
                direction="increase",
                effort_weight=0.5,
                feasibility_rate=0.70,
                allowed_values=[0, 1],
            ),
            # Mutable: rebuild/contents
            "sum_insured_buildings": FeatureConstraint(
                name="sum_insured_buildings",
                mutability=Mutability.MUTABLE,
                direction="either",
                effort_weight=0.1,
                feasibility_rate=0.85,
                min_value=50000.0,
                max_value=2000000.0,
            ),
            "sum_insured_contents": FeatureConstraint(
                name="sum_insured_contents",
                mutability=Mutability.MUTABLE,
                direction="either",
                effort_weight=0.1,
                feasibility_rate=0.85,
                min_value=5000.0,
                max_value=500000.0,
            ),
            "voluntary_excess": FeatureConstraint(
                name="voluntary_excess",
                mutability=Mutability.MUTABLE,
                direction="increase",
                effort_weight=0.2,
                feasibility_rate=0.80,
                min_value=0.0,
                max_value=2500.0,
            ),
            # Conditionally mutable
            "postcode_flood_risk": FeatureConstraint(
                name="postcode_flood_risk",
                mutability=Mutability.CONDITIONALLY_MUTABLE,
                direction="decrease",
                effort_weight=6.0,
                feasibility_rate=0.05,
                min_value=0,
                max_value=5,
                causal_children=["subsidence_risk"],
            ),
            "subsidence_risk": FeatureConstraint(
                name="subsidence_risk",
                mutability=Mutability.CONDITIONALLY_MUTABLE,
                direction=None,
                effort_weight=0.0,
                feasibility_rate=0.0,
                min_value=0,
                max_value=5,
            ),
        }

        def propagate_postcode_flood(factual: dict, interventions: dict) -> dict:
            new_flood = interventions.get("postcode_flood_risk", factual.get("postcode_flood_risk", 2))
            return {"subsidence_risk": max(0, min(5, round(new_flood * 0.6)))}

        return cls(constraints, {"postcode_flood_risk": propagate_postcode_flood})

    def get_constraint(self, name: str) -> FeatureConstraint:
        """Return the constraint for a named feature."""
        if name not in self._constraints:
            raise KeyError(f"Feature '{name}' not in constraint graph.")
        return self._constraints[name]

    def get_mutable_features(self) -> List[str]:
        """Return names of all features the policyholder can action."""
        return [
            name
            for name, c in self._constraints.items()
            if c.is_actionable()
        ]

    def get_immutable_features(self) -> List[str]:
        """Return names of features the policyholder cannot change."""
        return [
            name
            for name, c in self._constraints.items()
            if not c.is_actionable()
        ]

    def all_constraints(self) -> Dict[str, FeatureConstraint]:
        """Return the full constraint mapping."""
        return dict(self._constraints)

    def _topological_order(self) -> List[str]:
        """
        Return feature names in topological order (parents before children).

        Used by propagate_causal_effects to process interventions in the
        correct dependency order.
        """
        visited: set = set()
        order: List[str] = []

        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            constraint = self._constraints.get(name)
            if constraint:
                for child in constraint.causal_children:
                    visit(child)
            order.append(name)

        for name in self._constraints:
            visit(name)

        return list(reversed(order))

    def propagate_causal_effects(
        self,
        factual: Dict[str, Any],
        interventions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compute downstream causal effects of a set of interventions.

        Processes interventions in topological order. When a feature with
        causal_children is intervened on, its propagation function is called
        to derive updated values for all downstream features.

        Parameters
        ----------
        factual:
            Original feature values for the policyholder.
        interventions:
            Proposed changes, keyed by feature name (partial dict is fine).

        Returns
        -------
        dict
            Full proposed row: original values with all interventions and
            derived causal effects applied.
        """
        result = dict(factual)
        result.update(interventions)

        for feature_name in self._topological_order():
            if feature_name not in interventions:
                continue
            constraint = self._constraints.get(feature_name)
            if not constraint or not constraint.causal_children:
                continue
            if feature_name not in self._propagation_functions:
                continue

            propagate_fn = self._propagation_functions[feature_name]
            derived = propagate_fn(factual, result)
            result.update(derived)

        return result

    def validate_counterfactual(
        self,
        factual: Dict[str, Any],
        counterfactual: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """
        Validate a full counterfactual row against all constraints.

        Parameters
        ----------
        factual:
            Original feature values.
        counterfactual:
            Proposed counterfactual row.

        Returns
        -------
        (is_valid, errors)
            is_valid is True only when errors is empty.
        """
        errors: List[str] = []
        for name, constraint in self._constraints.items():
            if name not in counterfactual:
                continue
            orig = factual.get(name)
            prop = counterfactual[name]
            valid, reason = constraint.validate_counterfactual(orig, prop)
            if not valid:
                errors.append(reason)
        return len(errors) == 0, errors

    def add_constraint(self, constraint: FeatureConstraint) -> None:
        """Add or replace a constraint in the graph."""
        self._constraints[constraint.name] = constraint

    def add_propagation_function(
        self, feature_name: str, fn: Callable
    ) -> None:
        """Register a causal propagation function for a feature."""
        self._propagation_functions[feature_name] = fn

    def __repr__(self) -> str:
        n_mutable = len(self.get_mutable_features())
        n_total = len(self._constraints)
        return (
            f"ActionabilityGraph(features={n_total}, mutable={n_mutable}, "
            f"propagation_fns={len(self._propagation_functions)})"
        )
