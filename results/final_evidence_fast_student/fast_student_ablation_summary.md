# Fast student ablation summary

- Device: `cpu`
- Input: `[1, 2, 2048]`
- Calibration: `inverse_hyperbola: x = a/(W-b)+c, a=2485.85238, b=0.489855695, c=92.7040181`

| Model | Params | MACs(M) | Pos MAE (mm) | Pos RMSE (mm) | Pos MaxAE (mm) | Mean ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| plain_cnn_w32 | 105857 | 24.707 | 0.104577 | 0.145060 | 0.615766 | 1.1756 | 1.4881 |

## Environment

- Python: `3.13.5`
- Platform: `Windows-10-10.0.19045-SP0`
- PyTorch: `2.10.0+cpu`
- CPU threads: `1`