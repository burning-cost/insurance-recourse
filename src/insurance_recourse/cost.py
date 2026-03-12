"""
Human-interpretable effort metrics for insurance recourse actions.

Euclidean distance in feature space is meaningless to a policyholder. This
module translates feature changes into monetary costs (£), implementation
timelines (days), and feasibility probabilities — quantities that the FCA
Consumer Duty framework requires to be disclosed in plain terms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import pandas as pd


@dataclass
class RecourseEffort:
    """
    Human-interpretable cost of a recourse action.

    Parameters
    ----------
    monetary_cost:
        Estimated out-of-pocket cost to the policyholder in GBP to implement
        all feature changes in this action.
    time_days:
        Estimated calendar days to fully implement this action.
    feasibility_probability:
        P(average similar policyholder can execute this action). Product of
        per-feature feasibility rates.
    feature_changes:
        Dict mapping feature name to (original_value, counterfactual_value).
    """

    monetary_cost: float
    time_days: float
    feasibility_probability: float
    feature_changes: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.monetary_cost < 0:
            raise ValueError(f"monetary_cost must be non-negative, got {self.monetary_cost}")
        if self.time_days < 0:
            raise ValueError(f"time_days must be non-negative, got {self.time_days}")
        if not 0.0 <= self.feasibility_probability <= 1.0:
            raise ValueError(
                f"feasibility_probability must be in [0, 1], "
                f"got {self.feasibility_probability}"
            )

    @property
    def is_feasible(self) -> bool:
        """True if feasibility_probability is above a minimal threshold (5%)."""
        return self.feasibility_probability >= 0.05

    def as_dict(self) -> dict:
        return {
            "monetary_cost_gbp": round(self.monetary_cost, 2),
            "time_days": round(self.time_days, 1),
            "feasibility_probability": round(self.feasibility_probability, 4),
            "feature_changes": {
                k: {"from": v[0], "to": v[1]}
                for k, v in self.feature_changes.items()
            },
        }


class InsuranceCostFunction:
    """
    Maps feature changes to human-interpretable effort metrics.

    Each changed feature contributes a monetary cost, a time cost, and a
    feasibility rate. The total feasibility probability is the product of
    per-feature rates (independent assumption — conservative if actions are
    correlated).

    Parameters
    ----------
    monetary_costs:
        Mapping from feature name to cost in GBP per unit change (for
        continuous features) or per change event (for categorical features).
    time_costs_days:
        Mapping from feature name to calendar days per unit change or per
        change event.
    feasibility_rates:
        Mapping from feature name to [0, 1] probability that an average
        similar policyholder can action this change.
    default_monetary_cost:
        Fallback monetary cost for features not listed in monetary_costs.
    default_time_days:
        Fallback time cost for features not listed in time_costs_days.
    default_feasibility:
        Fallback feasibility rate for features not listed in feasibility_rates.

    Examples
    --------
    >>> cost_fn = InsuranceCostFunction.motor_defaults()
    >>> effort = cost_fn.compute(factual, counterfactual)
    >>> print(f"Cost: £{effort.monetary_cost:.0f}, {effort.time_days:.0f} days")
    """

    def __init__(
        self,
        monetary_costs: Optional[Dict[str, float]] = None,
        time_costs_days: Optional[Dict[str, float]] = None,
        feasibility_rates: Optional[Dict[str, float]] = None,
        default_monetary_cost: float = 0.0,
        default_time_days: float = 7.0,
        default_feasibility: float = 0.5,
    ) -> None:
        self.monetary_costs: Dict[str, float] = monetary_costs or {}
        self.time_costs_days: Dict[str, float] = time_costs_days or {}
        self.feasibility_rates: Dict[str, float] = feasibility_rates or {}
        self.default_monetary_cost = default_monetary_cost
        self.default_time_days = default_time_days
        self.default_feasibility = default_feasibility

    def compute(
        self,
        factual: "pd.Series",
        counterfactual: "pd.Series",
    ) -> RecourseEffort:
        """
        Compute effort required to move from factual to counterfactual.

        Only features that actually change contribute to the effort. The
        feasibility probability is the product of per-feature rates for
        all changed features.

        Parameters
        ----------
        factual:
            Original policyholder feature vector (pandas Series).
        counterfactual:
            Proposed counterfactual feature vector (pandas Series).

        Returns
        -------
        RecourseEffort
        """
        total_monetary = 0.0
        max_time_days = 0.0
        feasibility = 1.0
        feature_changes: Dict[str, Tuple[Any, Any]] = {}

        for feature in counterfactual.index:
            if feature not in factual.index:
                continue

            orig = factual[feature]
            prop = counterfactual[feature]

            # Skip unchanged features
            try:
                if float(orig) == float(prop):
                    continue
            except (TypeError, ValueError):
                if orig == prop:
                    continue

            feature_changes[feature] = (orig, prop)

            # Monetary cost
            base_cost = self.monetary_costs.get(feature, self.default_monetary_cost)
            try:
                magnitude = abs(float(prop) - float(orig))
                # For continuous features, scale by magnitude; for categoricals
                # (magnitude ~1 per step) this also works reasonably
                total_monetary += base_cost * magnitude
            except (TypeError, ValueError):
                # Categorical: flat cost per change event
                total_monetary += base_cost

            # Time cost (take max across features: parallel implementation)
            base_days = self.time_costs_days.get(feature, self.default_time_days)
            try:
                magnitude = abs(float(prop) - float(orig))
                days = base_days * max(1.0, magnitude)
            except (TypeError, ValueError):
                days = base_days
            max_time_days = max(max_time_days, days)

            # Feasibility (product of independent rates)
            rate = self.feasibility_rates.get(feature, self.default_feasibility)
            feasibility *= rate

        if not feature_changes:
            # No changes: trivially feasible at zero cost
            return RecourseEffort(
                monetary_cost=0.0,
                time_days=0.0,
                feasibility_probability=1.0,
                feature_changes={},
            )

        return RecourseEffort(
            monetary_cost=round(total_monetary, 2),
            time_days=round(max_time_days, 1),
            feasibility_probability=round(min(1.0, max(0.0, feasibility)), 6),
            feature_changes=feature_changes,
        )

    @classmethod
    def motor_defaults(cls) -> "InsuranceCostFunction":
        """
        Default cost function for UK motor insurance recourse.

        Costs are illustrative market-rate estimates (March 2026).
        Override with your own market data.
        """
        return cls(
            monetary_costs={
                # Per-event costs for categorical changes
                "vehicle_security": 250.0,   # Thatcham Cat 1 immobiliser ~£250
                "pass_plus": 150.0,          # Pass Plus course ~£150
                "garaging": 0.0,             # Behavioural change, no direct cost
                "telematics": 50.0,          # Black box installation ~£50
                "occupation_risk": 0.0,      # Occupation change: no direct cost
                # Per-unit continuous features
                "annual_mileage": 0.0,       # Mileage reduction: no direct cost
                "postcode_risk": 0.0,        # Moving costs handled separately
            },
            time_costs_days={
                "vehicle_security": 7.0,
                "pass_plus": 90.0,           # Course booking and completion
                "garaging": 30.0,            # May require renting garage
                "telematics": 7.0,
                "occupation_risk": 180.0,    # Job change realistically 6 months
                "annual_mileage": 1.0,       # Immediate behavioural change
                "postcode_risk": 90.0,       # Moving house
            },
            feasibility_rates={
                "vehicle_security": 0.85,
                "pass_plus": 0.40,
                "garaging": 0.35,
                "telematics": 0.55,
                "occupation_risk": 0.10,
                "annual_mileage": 0.60,
                "postcode_risk": 0.08,
            },
            default_monetary_cost=0.0,
            default_time_days=7.0,
            default_feasibility=0.5,
        )

    @classmethod
    def home_defaults(cls) -> "InsuranceCostFunction":
        """
        Default cost function for UK home insurance recourse.

        Costs are illustrative market-rate estimates (March 2026).
        """
        return cls(
            monetary_costs={
                "alarm_grade": 800.0,       # Professional alarm install ~£500-1200
                "locks_grade": 150.0,       # British Standard locks ~£100-200
                "smoke_alarms": 25.0,       # Interconnected smoke alarms ~£25-50
                "water_leak_detector": 120.0,  # Smart leak detector ~£80-160
                "voluntary_excess": 0.0,    # No direct cost
                "sum_insured_buildings": 0.0,
                "sum_insured_contents": 0.0,
            },
            time_costs_days={
                "alarm_grade": 14.0,
                "locks_grade": 3.0,
                "smoke_alarms": 1.0,
                "water_leak_detector": 2.0,
                "voluntary_excess": 1.0,
                "sum_insured_buildings": 1.0,
                "sum_insured_contents": 1.0,
            },
            feasibility_rates={
                "alarm_grade": 0.75,
                "locks_grade": 0.90,
                "smoke_alarms": 0.95,
                "water_leak_detector": 0.70,
                "voluntary_excess": 0.80,
                "sum_insured_buildings": 0.85,
                "sum_insured_contents": 0.85,
            },
            default_monetary_cost=0.0,
            default_time_days=7.0,
            default_feasibility=0.5,
        )

    def __repr__(self) -> str:
        return (
            f"InsuranceCostFunction("
            f"features_with_costs={len(self.monetary_costs)}, "
            f"features_with_time={len(self.time_costs_days)})"
        )
