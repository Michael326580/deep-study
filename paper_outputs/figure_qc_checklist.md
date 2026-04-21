# Figure QC checklist

## fig01_position_error_curves
- [x] no clipped labels
- [x] no overlapping tick labels
- [x] legend does not occlude major data (single-column legend moved above)
- [x] readable at final single-column size
- [x] readable at final double-column size
- [x] consistent encoding across all figures
- [x] status: PASS for main-text use (recommended: double-column)

## fig02_error_distribution
- [x] no clipped labels
- [x] no overlapping tick labels
- [x] legend moved to figure-level top, avoids panel occlusion
- [x] single-column uses vertical layout to avoid crowding
- [x] double-column panel (b) widened to avoid x-label crowding
- [x] double-column panel (b) uses compact x-label size and 2-line `Plain FFT`
- [x] consistent encoding across all figures
- [x] status: PASS for main-text use (recommended: double-column)

## fig04_summary_bars
- [x] compact layout and restrained styling
- [x] method set restricted to Teacher/Student/Plain FFT
- [x] status: PASS as supplementary figure
- [ ] recommended for main text (default recommendation: supplementary)

## fig03_accuracy_latency_tradeoff
- [x] intentionally not generated when any latency input is missing
- [x] reason wording fixed: missing traceable latency inputs
- [ ] status: BLOCKED for main-text use until latency inputs are provided
