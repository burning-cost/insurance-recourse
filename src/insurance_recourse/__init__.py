"""
insurance-recourse: Algorithmic recourse for FCA Consumer Duty in UK personal
lines insurance pricing.

Wraps DiCE and alibi to generate actionable premium-reduction recommendations
with insurance-native constraints and FCA-compliant reporting.

Quick start::

    from insurance_recourse.constraints import ActionabilityGraph
    from insurance_recourse.cost import InsuranceCostFunction
    from insurance_recourse.generator import RecourseGenerator
    from insurance_recourse.report import RecourseReport

    graph = ActionabilityGraph.from_template("motor")
    cost_fn = InsuranceCostFunction.motor_defaults()
    gen = RecourseGenerator(model, graph, cost_fn, backend="dice")
    actions = gen.generate(factual, target_premium=900.0, current_premium=1200.0)
    report = RecourseReport(factual, actions, {"model_version": "2024-Q4"})
    print(report.to_html())
"""

from insurance_recourse.constraints import (
    ActionabilityGraph,
    FeatureConstraint,
    Mutability,
)
from insurance_recourse.cost import (
    InsuranceCostFunction,
    RecourseEffort,
)
from insurance_recourse.generator import (
    RecourseAction,
    RecourseGenerator,
)
from insurance_recourse.report import RecourseReport

__version__ = "0.1.0"
__all__ = [
    "ActionabilityGraph",
    "FeatureConstraint",
    "Mutability",
    "InsuranceCostFunction",
    "RecourseEffort",
    "RecourseAction",
    "RecourseGenerator",
    "RecourseReport",
]
