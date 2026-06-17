# vscode-surv

Small Python scripts.

## Files

- [hello.py](hello.py) — basic hello world script.
- [surv.py](surv.py) — survival analysis example on a synthetic (pseudo) patient dataset, using Kaplan-Meier estimation, a log-rank test, and a Cox Proportional Hazards model.

## Usage

```bash
pip install numpy pandas lifelines matplotlib
python surv.py
```

This generates `km_survival_curves.png` with Kaplan-Meier survival curves by treatment group, and prints median survival times, log-rank test p-value, and Cox model hazard ratios to the console.
