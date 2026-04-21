# Figure style notes (Teacher / Student / Plain FFT)

## Unified visual encoding
- Teacher: color #1f3b73, solid line, marker o
- Student: color #a0512d, dashed line, marker s
- Plain FFT: color #2d6a4f, dotted line, marker ^

Design rationale:
1. Low-saturation, journal-friendly colors with clear contrast.
2. Triple encoding (color + linestyle + marker) preserves grayscale readability.
3. Sparse markers reduce clutter in dense curves.
4. Light grid and thin zero-reference line avoid overpowering data.

## Recommended manuscript placement
- Main text (final freeze): Fig.1 double-column + Fig.2 double-column.
- Main text (conditional): Fig.3 only when traceable latency inputs are available for Teacher/Student/Plain FFT.
- Supplementary: Fig.4 (single-column preferred for compact appendix layout; keep double-column as reserve only).

## Symbol meaning
- Fig.1/2/3 line/marker encoding:
  - Teacher = deep blue, solid, circle
  - Student = brown-orange, dashed, square
  - Plain FFT = dark green, dotted, triangle
- Fig.2 double-column panel (b): `Plain FFT` is split as `Plain` + `FFT` for readability.
- Fig.4 bar textures:
  - // = MAE
  - \ = RMSE
  - .. = MaxAE
- Fig.2 boxplot elements:
  - median line = central tendency
  - box = interquartile range (Q1--Q3)
  - whiskers = non-outlier spread (fliers hidden)

## Old multi-baseline figures
- `benchmark_error_plot.png` and `benchmark_boxplot_or_hist.png` are not recommended for the main text
  after reducing methods to Teacher/Student/Plain FFT. Keep for supplementary only if needed.
