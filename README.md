# insurance-recourse

Algorithmic recourse for FCA Consumer Duty in UK personal lines insurance pricing.

Answers the question: **"What can I change to lower my premium?"** — with insurance-native constraints, premium-denominated output, and audit-ready FCA reporting.

## The Problem

The FCA Consumer Duty (PS22/9) requires firms to produce clear explanations of how prices are set and what policyholders can do to obtain a better outcome. Existing algorithmic recourse libraries (DiCE, alibi) are research-grade tools that generate counterfactuals in abstract feature space. They have no concept of:

- **Mutability constraints specific to insurance** — age and claims history cannot be changed; mileage can only decrease; garaging can only improve
- **Causal structure** — changing postcode changes garaging risk, crime rate, and flood exposure simultaneously
- **Monetary terms** — a £250 immobiliser costs £250, takes 7 days, and 85% of policyholders can actually do it
- **FCA audit format** — regulators need a tamper-evident record with policyholder ID, model version, and SHA-256 hash

This library wraps DiCE and alibi with those insurance-specific layers.

## What It Produces

```
Premium Reduction Options — POL-123456
Current premium: £1,200.00/yr

Rank  Action                              New premium  Saving       Cost    Days  Feasibility
1     Add Thatcham Cat 1 immobiliser      £960.00      £240 (20%)   £250    7     85%
2     Reduce mileage from 12k to 8k      £1,050.00    £150 (12.5%) none    1     60%
3     Install telematics black box        £1,100.00    £100 (8.3%)  £50     7     55%
```

HTML output, JSON audit dict with SHA-256 hash, ranked by saving.

## Installation

```bash
pip install insurance-recourse                  # core + sklearn
pip install insurance-recourse[dice]            # + DiCE backend
pip install insurance-recourse[alibi]           # + alibi CFRL backend
pip install insurance-recourse[dice,alibi]      # both
```

**Python 3.9+ required.**

## Quick Start

```python
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from insurance_recourse.constraints import ActionabilityGraph
from insurance_recourse.cost import InsuranceCostFunction
from insurance_recourse.generator import RecourseGenerator
from insurance_recourse.report import RecourseReport

# 1. Build constraint graph from template
graph = ActionabilityGraph.from_template("motor")

# 2. Cost function (use defaults or override with your market data)
cost_fn = InsuranceCostFunction.motor_defaults()

# 3. Wrap your existing pricing model
#    model can be any sklearn-compatible estimator or callable
model = GradientBoostingRegressor(...)  # already fitted
gen = RecourseGenerator(model, graph, cost_fn, backend="dice", n_counterfactuals=5)

# 4. Generate recourse for a single policyholder
factual = pd.Series({
    "age": 28, "annual_mileage": 14000, "vehicle_security": 1,
    "pass_plus": 0, "garaging": 0, "telematics": 0, ...
})
current_premium = float(model.predict(factual.to_frame().T)[0])

actions = gen.generate(
    factual,
    target_premium=current_premium * 0.85,  # find 15%+ savings
    current_premium=current_premium,
    max_monetary_cost=500.0,                 # filter: max £500 upfront cost
    max_days=30,                             # filter: actionable within 30 days
)

# 5. Generate FCA report
report = RecourseReport(
    factual=factual,
    actions=actions,
    model_metadata={"model_version": "2024-Q4-motor-v3", "product": "motor"},
    policyholder_id="POL-123456",
    current_premium=current_premium,
)

html = report.to_html()           # customer-facing explanation
audit = report.to_dict()          # JSON audit record with SHA-256 hash
print(audit["audit_hash"])        # e.g. "a3f8c2d1..."
```

## Backends

### `dice` (recommended for GLMs and sklearn pipelines)

Uses DiCE's genetic algorithm or KD-Tree to search for counterfactuals. Works with any sklearn-compatible model. Fast iteration. Requires `pip install dice-ml>=0.11`.

### `focus` (optimal for tree ensembles, no extra deps)

Implements the FOCUS sigmoid approximation (Lucic et al., AAAI 2022) internally. Replaces each tree split threshold with σ(θ_j − x_{f_j}), making the forest output differentiable w.r.t. inputs, then runs gradient descent under constraint. Works with sklearn `DecisionTreeRegressor`, `RandomForestRegressor`, `GradientBoostingRegressor`. No extra installation beyond the core package.

### `alibi_cfrl` (model-agnostic RL for XGBoost/CatBoost)

Uses alibi's Counterfactual RL (CFRL) approach — trains an RL agent to find counterfactuals without requiring model differentiability. Requires `pip install alibi>=0.9` plus TensorFlow or PyTorch. Experimental; expect a few minutes for RL training.

## Custom Constraints

```python
from insurance_recourse.constraints import ActionabilityGraph, FeatureConstraint, Mutability

# Start from template and add product-specific features
graph = ActionabilityGraph.from_template("motor")

# Add a custom feature
graph.add_constraint(FeatureConstraint(
    name="advanced_driver_course",
    mutability=Mutability.MUTABLE,
    direction="increase",
    effort_weight=2.0,
    feasibility_rate=0.25,
    allowed_values=[0, 1],
))

# Add causal propagation for a conditionally mutable feature
def my_propagation(factual, interventions):
    new_val = interventions.get("my_parent_feature", factual["my_parent_feature"])
    return {"derived_child_feature": new_val * 0.8}

graph.add_propagation_function("my_parent_feature", my_propagation)
```

## Custom Cost Function

```python
from insurance_recourse.cost import InsuranceCostFunction

cost_fn = InsuranceCostFunction(
    monetary_costs={
        "vehicle_security": 350.0,   # your local market rate
        "pass_plus": 175.0,
        "telematics": 0.0,           # your insurer subsidises it
    },
    time_costs_days={
        "vehicle_security": 14.0,
        "pass_plus": 90.0,
        "telematics": 3.0,
    },
    feasibility_rates={
        "vehicle_security": 0.80,
        "pass_plus": 0.35,
        "telematics": 0.70,
    },
)
```

## FCA Consumer Duty Compliance

The `to_dict()` output includes:

- `policyholder_id`: reference for your records system
- `current_premium_gbp`: current annual premium
- `model_metadata`: model version, product, effective date
- `generated_at`: ISO 8601 timestamp
- `factual_features`: full feature vector at time of explanation
- `recourse_options`: ranked actions with saving £/%, cost, timeline, feasibility
- `audit_hash`: SHA-256 over all fields (tamper detection)

Store `audit_hash` alongside the policyholder record. On regulatory inspection, recompute from stored inputs to verify integrity.

## Modules

| Module | Purpose |
|--------|---------|
| `constraints.py` | `FeatureConstraint`, `ActionabilityGraph` — mutability, direction, causal DAG |
| `cost.py` | `InsuranceCostFunction`, `RecourseEffort` — monetary/time/feasibility effort |
| `generator.py` | `RecourseGenerator`, `RecourseAction` — counterfactual search + action assembly |
| `report.py` | `RecourseReport` — FCA-format HTML and JSON audit output |

## Notes on Design

DiCE requires a "dataset" to infer feature ranges. When you call `generate()`, the library synthesises a small training-like dataset (~50 rows) by perturbing the factual point within constraint bounds. This is a pragmatic workaround for inference-time use — you aren't expected to pass training data at prediction time.

The `alibi_cfrl` backend trains an RL agent, which is expensive. Use it only when you need model-agnostic recourse for black-box XGBoost/CatBoost and can afford the training time.

The FOCUS backend implements the sigmoid approximation from scratch using sklearn's `tree_` attribute. It requires the model to expose tree internals via the sklearn API. For models that don't (arbitrary callable predict functions), fall back to `backend="dice"`.

## Licence

MIT
