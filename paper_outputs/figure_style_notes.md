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
- Main text: Fig.1 (position error curves), Fig.2 (ECDF+boxplot), Fig.3 (if latency available).
- Optional/supplementary: Fig.4 grouped bars if space is limited.

## Old multi-baseline figures
- `benchmark_error_plot.png` and `benchmark_boxplot_or_hist.png` are not recommended for the main text
  after reducing methods to Teacher/Student/Plain FFT. Keep for supplementary only if needed.
