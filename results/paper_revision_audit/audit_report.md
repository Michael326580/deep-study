# Repository numerical audit

- fixed_results.csv: N=500, position=180.0--229.9 mm, width=18.6125--28.9355 pixel
- dataset_distill_meta.csv: N=4310 segments
- Teacher MAE: 0.0679638 mm
- Student MAE: 0.117037 mm

## Conflict note

- Target single-student MAE 0.0264 mm conflicts with current grouped validation MAE 0.117037 mm.
- Student latency is not traceable unless a timing script and raw timing log are added.