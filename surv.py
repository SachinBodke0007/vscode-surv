"""
Survival analysis example using a pseudo (synthetic) dataset.

Scenario: patients enrolled in a trial, some on a new "drug" treatment,
some on placebo. We track time-to-event (e.g. relapse/death) and whether
the event was observed or the patient was censored (lost to follow-up /
study ended before the event happened).

Requires: numpy, pandas, lifelines, matplotlib
    pip install numpy pandas lifelines matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test


def make_pseudo_data(n=200, seed=42):
    rng = np.random.default_rng(seed)

    age = rng.normal(60, 10, n)
    treatment = rng.integers(0, 2, n)  # 0 = placebo, 1 = drug

    # True hazard depends on age and treatment: drug + younger age -> longer survival
    baseline_hazard = 0.02
    risk_score = 0.03 * (age - 60) - 0.8 * treatment
    event_time = rng.exponential(1 / (baseline_hazard * np.exp(risk_score)))

    # Censoring: study ends at 60 months, plus random dropout
    censor_time = rng.uniform(10, 60, n)
    time = np.minimum(event_time, censor_time)
    event_observed = (event_time <= censor_time).astype(int)

    df = pd.DataFrame({
        "time": time,
        "event": event_observed,
        "treatment": treatment,
        "age": age,
    })
    return df


def kaplan_meier_analysis(df):
    kmf = KaplanMeierFitter()

    fig, ax = plt.subplots(figsize=(8, 6))
    for group, label in [(0, "Placebo"), (1, "Drug")]:
        mask = df["treatment"] == group
        kmf.fit(df.loc[mask, "time"], df.loc[mask, "event"], label=label)
        kmf.plot_survival_function(ax=ax)
        print(f"\n{label} median survival time: {kmf.median_survival_time_:.2f}")

    ax.set_title("Kaplan-Meier Survival Curves by Treatment Group")
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival probability")
    plt.tight_layout()
    plt.savefig("km_survival_curves.png")
    print("\nSaved plot to km_survival_curves.png")

    # Log-rank test: is the difference between groups statistically significant?
    placebo = df[df["treatment"] == 0]
    drug = df[df["treatment"] == 1]
    result = logrank_test(
        placebo["time"], drug["time"],
        event_observed_A=placebo["event"], event_observed_B=drug["event"],
    )
    print(f"\nLog-rank test p-value: {result.p_value:.4f}")


def cox_ph_analysis(df):
    cph = CoxPHFitter()
    cph.fit(df, duration_col="time", event_col="event")
    print("\nCox Proportional Hazards model summary:")
    cph.print_summary()

    # Hazard ratios: exp(coef) > 1 means higher risk, < 1 means lower risk
    print("\nHazard ratios:")
    print(np.exp(cph.params_))


if __name__ == "__main__":
    df = make_pseudo_data()
    print(df.head())

    kaplan_meier_analysis(df)
    cox_ph_analysis(df)
