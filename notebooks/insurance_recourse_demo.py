# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-recourse: Algorithmic Recourse for FCA Consumer Duty
# MAGIC
# MAGIC This notebook demonstrates the full workflow of the `insurance-recourse` library
# MAGIC on a synthetic UK motor insurance dataset. It covers:
# MAGIC
# MAGIC 1. Generating a realistic synthetic motor portfolio
# MAGIC 2. Fitting a pricing model (GLM via sklearn)
# MAGIC 3. Configuring the constraint graph and cost function
# MAGIC 4. Generating premium-reduction recommendations for individual policyholders
# MAGIC 5. Producing FCA Consumer Duty audit reports (JSON + HTML)
# MAGIC 6. Batch processing across a portfolio

# COMMAND ----------

# MAGIC %pip install insurance-recourse dice-ml

# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from insurance_recourse.constraints import ActionabilityGraph, FeatureConstraint, Mutability
from insurance_recourse.cost import InsuranceCostFunction
from insurance_recourse.generator import RecourseGenerator, RecourseAction
from insurance_recourse.report import RecourseReport

print("insurance-recourse loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Synthetic Motor Portfolio

# COMMAND ----------

rng = np.random.default_rng(42)
N = 5000

# Feature definitions
age = rng.integers(17, 75, N).astype(float)
gender = rng.choice([0.0, 1.0], N)           # 0=female, 1=male
licence_years = np.clip(age - 17.0, 0, 50).astype(float) - rng.integers(0, 3, N)
at_fault_claims_3yr = rng.choice([0, 1, 2], N, p=[0.85, 0.12, 0.03]).astype(float)
years_no_claims = np.clip(rng.integers(0, 10, N), 0, 9).astype(float)

annual_mileage = rng.integers(3000, 25000, N).astype(float)
vehicle_security = rng.choice([0, 1, 2, 3, 4], N, p=[0.15, 0.45, 0.25, 0.10, 0.05]).astype(float)
pass_plus = rng.choice([0, 1], N, p=[0.85, 0.15]).astype(float)
garaging = rng.choice([0, 1, 2, 3], N, p=[0.30, 0.35, 0.20, 0.15]).astype(float)
telematics = rng.choice([0, 1], N, p=[0.70, 0.30]).astype(float)
occupation_risk = rng.integers(1, 11, N).astype(float)
postcode_risk = rng.integers(1, 11, N).astype(float)
crime_rate_decile = np.clip(np.round(postcode_risk * 0.9 + rng.normal(0, 0.5, N)), 1, 10)
flood_zone_risk = np.clip(np.round((postcode_risk - 1) * 0.3 + rng.normal(0, 0.3, N)), 0, 3)
vehicle_age = rng.integers(0, 20, N).astype(float)
vehicle_value = np.clip(20000.0 * np.exp(-0.15 * vehicle_age) + rng.normal(0, 2000, N), 500, 100000)

# Log-linear premium structure (GLM multiplicative rating)
log_premium = (
    7.0                                          # base
    + np.log1p(age) * 0.3                        # age curve (U-shaped)
    - np.log1p(age - 17) * 0.2                   # experience discount
    + gender * 0.05                              # gender loading (small)
    + at_fault_claims_3yr * 0.4                  # claims loading
    - years_no_claims * 0.05                     # NCD discount
    + np.log1p(annual_mileage / 1000) * 0.15     # mileage loading
    - vehicle_security * 0.08                    # security discount
    - pass_plus * 0.05                           # Pass Plus discount
    - garaging * 0.04                            # garaging discount
    - telematics * 0.07                          # telematics discount
    + occupation_risk * 0.03                     # occupation loading
    + postcode_risk * 0.04                       # postcode loading
    + crime_rate_decile * 0.02
    + rng.normal(0, 0.1, N)                      # residual noise
)
premium = np.exp(log_premium)

features = [
    "age", "gender", "licence_years", "at_fault_claims_3yr", "years_no_claims",
    "annual_mileage", "vehicle_security", "pass_plus", "garaging", "telematics",
    "occupation_risk", "postcode_risk", "crime_rate_decile", "flood_zone_risk",
    "vehicle_age", "vehicle_value",
]

df = pd.DataFrame({
    "age": age,
    "gender": gender,
    "licence_years": licence_years,
    "at_fault_claims_3yr": at_fault_claims_3yr,
    "years_no_claims": years_no_claims,
    "annual_mileage": annual_mileage,
    "vehicle_security": vehicle_security,
    "pass_plus": pass_plus,
    "garaging": garaging,
    "telematics": telematics,
    "occupation_risk": occupation_risk,
    "postcode_risk": postcode_risk,
    "crime_rate_decile": crime_rate_decile,
    "flood_zone_risk": flood_zone_risk,
    "vehicle_age": vehicle_age,
    "vehicle_value": vehicle_value,
    "premium": premium,
})

print(f"Portfolio: {len(df):,} policies")
print(f"Premium range: £{df.premium.min():.0f} — £{df.premium.max():.0f}")
print(f"Mean premium: £{df.premium.mean():.0f}")
display(df.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Fit Pricing Model

# COMMAND ----------

from sklearn.model_selection import train_test_split

X = df[features]
y = np.log(df["premium"])  # log-linear GLM

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge regression on log-premium (Tweedie GLM approximation)
model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0)),
])
model.fit(X_train, y_train)

# Wrapper: exponentiate back to premium space
class PremiumModel:
    def __init__(self, log_model):
        self._model = log_model

    def predict(self, X):
        return np.exp(self._model.predict(X))

premium_model = PremiumModel(model)

# Evaluate
y_pred = premium_model.predict(X_test)
y_actual = np.exp(y_test)
mae = np.mean(np.abs(y_pred - y_actual))
print(f"Test MAE: £{mae:.0f}")
print(f"Test MAPE: {np.mean(np.abs(y_pred - y_actual) / y_actual) * 100:.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Configure Constraint Graph

# COMMAND ----------

# Use the pre-built motor template
graph = ActionabilityGraph.from_template("motor")
print(graph)
print()
print("Mutable features:", graph.get_mutable_features())
print("Immutable features:", graph.get_immutable_features())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Configure Cost Function

# COMMAND ----------

# Use motor defaults — override with your own market data
cost_fn = InsuranceCostFunction.motor_defaults()
print(cost_fn)
print()
print("Monetary costs per event/unit:")
for k, v in cost_fn.monetary_costs.items():
    print(f"  {k}: £{v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Generate Recourse for One Policyholder

# COMMAND ----------

# Select a high-premium policyholder to demonstrate recourse
high_premium_idx = df["premium"].nlargest(100).index
# Pick one with some actionable features
test_idx = df.loc[high_premium_idx].query(
    "vehicle_security < 3 and annual_mileage > 10000 and pass_plus == 0"
).index[0]

factual = df.loc[test_idx, features]
current_premium = float(premium_model.predict(factual.to_frame().T)[0])

print(f"Policyholder index: {test_idx}")
print(f"Current premium: £{current_premium:,.2f}")
print()
print("Feature values:")
for feat, val in factual.items():
    print(f"  {feat}: {val}")

# COMMAND ----------

# Create the generator
gen = RecourseGenerator(
    model=premium_model,
    actionability_graph=graph,
    cost_function=cost_fn,
    backend="dice",
    n_counterfactuals=5,
    random_state=42,
)
print(gen)

# COMMAND ----------

# Generate recourse actions
actions = gen.generate(
    factual=factual,
    target_premium=current_premium * 0.85,  # target 15%+ reduction
    current_premium=current_premium,
    max_monetary_cost=500.0,                 # affordable upfront costs
    max_days=90,                             # actionable within 3 months
)

print(f"Found {len(actions)} recourse actions\n")
for i, action in enumerate(actions, 1):
    print(f"  Option {i}: {action.description}")
    print(f"    Premium after: £{action.predicted_premium:,.2f}")
    print(f"    Saving: £{action.premium_reduction:,.2f} ({action.premium_reduction_pct:.1f}%)")
    print(f"    Cost: £{action.effort.monetary_cost:,.0f} | "
          f"Time: {action.effort.time_days:.0f} days | "
          f"Feasibility: {action.effort.feasibility_probability*100:.0f}%")
    print(f"    Valid: {action.validity}")
    if action.causal_effects:
        print(f"    Causal effects: {action.causal_effects}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. FCA Consumer Duty Report

# COMMAND ----------

report = RecourseReport(
    factual=factual,
    actions=actions,
    model_metadata={
        "model_version": "2024-Q4-motor-v3",
        "product": "motor",
        "effective_date": "2024-10-01",
        "model_type": "GLM-Ridge",
    },
    policyholder_id=f"POL-{test_idx:06d}",
    current_premium=current_premium,
)

# JSON audit record
audit_dict = report.to_dict()
import json
print("Audit record:")
print(json.dumps(audit_dict, indent=2, default=str))

# COMMAND ----------

# HTML report (rendered in notebook)
html = report.to_html()
print(f"HTML report: {len(html):,} chars")

# Save to Databricks FileStore for inspection
with open("/dbfs/tmp/recourse_report.html", "w") as f:
    f.write(html)
print("Saved to /dbfs/tmp/recourse_report.html")

# Display inline
displayHTML(html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Audit Hash Verification

# COMMAND ----------

import hashlib

# Verify the audit hash is reproducible
stored_hash = audit_dict["audit_hash"]
# Remove audit_hash before recomputing
verification_dict = {k: v for k, v in audit_dict.items() if k != "audit_hash"}
recomputed = hashlib.sha256(
    json.dumps(verification_dict, sort_keys=True, default=str).encode()
).hexdigest()

print(f"Stored hash:   {stored_hash}")
print(f"Recomputed:    {recomputed}")
print(f"Match: {stored_hash == recomputed}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Custom Constraints: Extend the Motor Template

# COMMAND ----------

# Demonstrate adding a bespoke constraint
custom_graph = ActionabilityGraph.from_template("motor")

# Add an advanced driver course (IAM/RoSPA)
custom_graph.add_constraint(FeatureConstraint(
    name="advanced_driving_course",
    mutability=Mutability.MUTABLE,
    direction="increase",
    effort_weight=2.5,
    feasibility_rate=0.20,
    allowed_values=[0, 1],  # 0=none, 1=IAM or RoSPA
))

print("Custom graph:", custom_graph)
print("Mutable features:", custom_graph.get_mutable_features())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. FOCUS Backend (tree ensemble, no extra deps)

# COMMAND ----------

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# Fit a gradient boosting model
gbm_log = GradientBoostingRegressor(
    n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42
)
gbm_log.fit(X_train, y_train)

class GBMPremiumModel:
    def __init__(self, model):
        self._model = model

    def predict(self, X):
        return np.exp(self._model.predict(X))

gbm_model = GBMPremiumModel(gbm_log)

# Use FOCUS backend (internal sigmoid approximation, no dice-ml needed)
focus_gen = RecourseGenerator(
    model=gbm_log,  # FOCUS uses the underlying sklearn model directly
    actionability_graph=graph,
    cost_function=cost_fn,
    backend="focus",
    n_counterfactuals=3,
    random_state=42,
)
print(focus_gen)
print()

# Generate counterfactuals for same policyholder
# Note: FOCUS works in the log-premium space here
factual_norm = factual.copy()
focus_actions = focus_gen.generate(
    factual=factual_norm,
    target_premium=np.log(current_premium * 0.85),  # log-space target
    current_premium=np.log(current_premium),
)
print(f"FOCUS found {len(focus_actions)} actions")
for a in focus_actions:
    print(f"  Premium change: {a.premium_reduction:.3f} log-units | {a.description[:60]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Batch Recourse Across Portfolio Segment

# COMMAND ----------

# Process the top-100 highest-premium policyholders
batch_idx = df["premium"].nlargest(100).index
batch_df = df.loc[batch_idx, features]
batch_premiums = premium_model.predict(batch_df)

results = []
for i, (idx, row) in enumerate(batch_df.iterrows()):
    current = float(batch_premiums[i])
    try:
        acts = gen.generate(
            factual=row,
            target_premium=current * 0.85,
            current_premium=current,
            max_monetary_cost=300.0,
        )
        best = acts[0] if acts else None
        results.append({
            "policyholder_idx": idx,
            "current_premium": round(current, 2),
            "best_premium_after": round(best.predicted_premium, 2) if best else current,
            "best_saving_gbp": round(best.premium_reduction, 2) if best else 0.0,
            "best_saving_pct": round(best.premium_reduction_pct, 1) if best else 0.0,
            "best_action": best.description[:80] if best else "No action found",
            "n_options": len(acts),
        })
    except Exception as e:
        results.append({
            "policyholder_idx": idx,
            "current_premium": round(current, 2),
            "best_saving_gbp": 0.0,
            "best_saving_pct": 0.0,
            "best_action": f"Error: {e}",
            "n_options": 0,
        })

results_df = pd.DataFrame(results)
print(f"Batch recourse complete: {len(results_df)} policyholders")
print(f"Mean best saving: £{results_df.best_saving_gbp.mean():.0f}")
print(f"Policyholders with at least 1 option: {(results_df.n_options > 0).sum()}")
display(results_df.head(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Constraint Validation Demo

# COMMAND ----------

from insurance_recourse.constraints import FeatureConstraint

# Demonstrate constraint validation
test_constraint = graph.get_constraint("annual_mileage")
print("annual_mileage constraint:")
print(f"  direction: {test_constraint.direction}")
print(f"  min_value: {test_constraint.min_value}")
print(f"  feasibility: {test_constraint.feasibility_rate}")

# Valid: decrease
valid, reason = test_constraint.validate_counterfactual(12000, 8000)
print(f"\nDecrease 12k->8k: valid={valid}")

# Invalid: increase
valid, reason = test_constraint.validate_counterfactual(12000, 15000)
print(f"Increase 12k->15k: valid={valid}, reason={reason}")

# Invalid: below min
valid, reason = test_constraint.validate_counterfactual(12000, 500)
print(f"Below min 12k->500: valid={valid}, reason={reason}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Causal Propagation Demo

# COMMAND ----------

# Demonstrate postcode change cascading to crime_rate and flood_zone
factual_dict = factual.to_dict()
print("Original values:")
print(f"  postcode_risk: {factual_dict.get('postcode_risk', 'N/A')}")
print(f"  crime_rate_decile: {factual_dict.get('crime_rate_decile', 'N/A')}")
print(f"  flood_zone_risk: {factual_dict.get('flood_zone_risk', 'N/A')}")

interventions = {"postcode_risk": 2}  # move to low-risk postcode
propagated = graph.propagate_causal_effects(factual_dict, interventions)

print("\nAfter moving to postcode_risk=2:")
print(f"  postcode_risk: {propagated.get('postcode_risk')}")
print(f"  crime_rate_decile: {propagated.get('crime_rate_decile')} (was {factual_dict.get('crime_rate_decile', '?')})")
print(f"  flood_zone_risk: {propagated.get('flood_zone_risk')} (was {factual_dict.get('flood_zone_risk', '?')})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC The `insurance-recourse` library provides:
# MAGIC
# MAGIC - **Constraint-aware recourse** — immutability, direction constraints, allowed values, causal DAG
# MAGIC - **Premium-denominated output** — all savings in £, not abstract distance metrics
# MAGIC - **Effort metrics** — monetary cost, implementation timeline, feasibility probability
# MAGIC - **Three backends** — DiCE (GLMs), FOCUS (tree ensembles), alibi CFRL (black-box)
# MAGIC - **FCA Consumer Duty reports** — structured JSON audit + customer-facing HTML with SHA-256 hash
# MAGIC
# MAGIC **Install**: `pip install insurance-recourse[dice]`
# MAGIC
# MAGIC **Source**: https://github.com/burning-cost/insurance-recourse
