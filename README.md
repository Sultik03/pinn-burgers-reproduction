# Physics-Informed Neural Network for Burgers' Equation

A PyTorch reproduction and sensitivity analysis of the continuous-time Burgers' equation example from Raissi, Perdikaris, and Karniadakis (2017), *Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations*.

This repository does more than reproduce one headline error. It evaluates how supervised data quantity and network architecture affect a PINN's ability to recover a nonlinear PDE solution with a sharp internal layer.

## Key results

| Result | Relative L2 error | Interpretation |
|---|---:|---|
| Original paper, reported configuration | 6.70e-04 | Reference result |
| Baseline PyTorch reproduction | 4.95e-03 | Captures the solution qualitatively, but is 7.4× higher than the paper's error |
| Best supervised-data sweep run, `N_u = 200` | 1.18e-3 | Best result in the completed data-size sweep |
| Best architecture sweep run, `8×40` | 7.48e-4 | Within about 11.6% of the paper's reported error, under a modified architecture |

> The `8×40` result is a sensitivity-experiment result, not an exact reproduction of the paper's baseline configuration.

![Baseline prediction](results/burgers_pinn_result.png)

## Problem formulation

The one-dimensional viscous Burgers' equation is

```text
u_t + u u_x - (0.01/pi) u_xx = 0,    x in [-1, 1], t in [0, 1]
u(0, x) = -sin(pi x)
u(t, -1) = u(t, 1) = 0
```

The neural network approximates `u(t, x)`. Automatic differentiation provides `u_t`, `u_x`, and `u_xx`, allowing the PDE residual

```text
f(t, x) = u_t + u u_x - (0.01/pi) u_xx
```

to be included in the training objective:

```text
MSE = MSE_u + MSE_f
```

- `MSE_u` fits initial and boundary observations.
- `MSE_f` penalizes violations of Burgers' equation at collocation points.

The method avoids an explicit finite-difference or finite-element discretization during training, although it still uses finitely sampled collocation points and a reference grid for evaluation.

## Baseline reproduction

- Framework: PyTorch
- Precision: `float64`
- Architecture: 8 hidden layers, 20 neurons per layer, `tanh`
- Trainable parameters: 3,021
- Supervised points: `N_u = 100`
- Collocation points: `N_f = 10,000`, sampled with Latin Hypercube Sampling
- Optimizers: Adam followed by L-BFGS
- Relative L2 error: `0.004951`
- Paper's reported relative L2 error: `6.70e-04`

The baseline error is higher than the paper's result, but the predicted field and time snapshots reproduce the principal qualitative behavior, including the sharp internal layer near `x = 0`.

## Completed experiments

### 1. Supervised-data sweep

The completed sweep varied `N_u` across `20, 40, 60, 80, 100, 200`.

![Data sweep](results/experiments/data_sweep_relative_l2.png)

Main findings:

- Accuracy improved dramatically once the supervised set reached roughly 80 points.
- The trend was not monotonic: `N_u = 40` performed worse than `N_u = 20`.
- Increasing `N_u` from 20 to 200 reduced relative L2 error by about 60.4×.
- The non-monotonic behavior shows that PINN training is affected by optimization and sampling, not only by data count.

Full results: [`results/experiments/data_sweep_results.csv`](results/experiments/data_sweep_results.csv)

### 2. Architecture sweep

The architecture experiment fixed `N_u = 100` and `N_f = 5,000`, used 500 Adam steps followed by 500 L-BFGS steps, and compared six networks.

![Architecture sweep](results/experiments/architecture_sweep_relative_l2.png)

Main findings:

- Increasing depth alone did not improve accuracy consistently.
- The `6×20` network performed worse than both `2×20` and `4×20`.
- At a fixed depth of eight hidden layers, increasing width from 10 to 20 to 40 neurons improved accuracy consistently.
- The `8×40` model achieved `7.48e-4`, the best result among the completed experiments.
- Wider models increased parameter count and runtime, but the accuracy gain from `8×20` to `8×40` was much larger than the runtime increase.

Full results: [`results/experiments/architecture_sweep_results.csv`](results/experiments/architecture_sweep_results.csv)

### 3. Collocation-point sweep

A collocation-point sweep was designed to compare PINNs with different `N_f` values, including `N_f = 0` as a data-only neural-network baseline. The experiment code exists, but a verified completed result table was not available when this README was generated. No values are invented here. Science remains inconveniently dependent on evidence.

## Repository structure

```text
pinn-burgers-reproduction/
├── README.md
├── TECHNICAL_REPORT.md
├── requirements.txt
├── data/
│   └── burgers_shock.mat
├── notebook/
│   ├── pinn_burgers.ipynb
│   ├── experiments_pinn_burgers.ipynb
│   └── 03_architecture_sweep.ipynb
└── results/
    ├── burgers_pinn_result.png
    └── experiments/
        ├── data_sweep_results.csv
        ├── architecture_sweep_results.csv
        ├── data_sweep_relative_l2.png
        └── architecture_sweep_relative_l2.png
```

## Installation and execution

```bash
git clone https://github.com/Sultik03/pinn-burgers-reproduction
cd pinn-burgers-reproduction

python3 -m venv venv
source venv/bin/activate
# Windows: venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook
```

Run the baseline notebook first, then the experiment notebooks. Runtime depends strongly on hardware, collocation count, architecture, and L-BFGS iteration budget.

## Reproducibility checklist

- Fix NumPy and PyTorch random seeds.
- Record the exact sampled training and collocation points.
- Use `float64` for stable second derivatives.
- Record optimizer settings and stopping criteria.
- Save raw CSV results, not only screenshots.
- Report hardware and runtime.
- Repeat important configurations with multiple seeds before making strong claims.
- Keep failed or anomalous runs in the analysis rather than silently deleting them.

## Interpretation

This project supports three conclusions:

1. A continuous-time PINN can recover the Burgers solution from sparse initial and boundary observations.
2. More supervised data generally helps, but the optimization landscape can produce non-monotonic results.
3. Width was more effective than depth in the completed architecture sweep, with `8×40` approaching the paper's reported accuracy.

The project therefore reproduces the central qualitative claim and demonstrates meaningful experimental analysis beyond one L2 error.

## Limitations

- The baseline does not numerically match the paper's headline error.
- The best architecture result uses a modified network and must not be presented as the exact paper configuration.
- Most configurations were run once, so seed sensitivity is not quantified.
- A verified collocation-sweep table is still missing.
- Relative L2 error can hide localized failures near the shock; future reporting should retain shock-region error and residual statistics.

## Technical report

See [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) for the complete methodology, results, analysis, limitations, and conclusions.

## Citation

```bibtex
@article{raissi2017physics,
  title={Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations},
  author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em},
  journal={arXiv preprint arXiv:1711.10561},
  year={2017}
}
```
