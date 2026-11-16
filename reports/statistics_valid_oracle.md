# Statistics against the annotated positions -- outputs/stats/valid_oracle/positions.parquet

549 ground-truth tracks of 51 clips. Noise (robust std of position minus position smoothed over 0.4 s): reference 0.015 m, pipeline 0.107 m (medians over tracks).

| smoothing (s) | step (s) | reference (m/min) | pipeline (m/min) | pipeline vs reference |
|---|---|---|---|---|
| 0 | 0 | 121.2 | 177.2 | +60.0% |
| 0 | 0.2 | 120.6 | 149.5 | +23.1% |
| 0.4 | 0.2 | 119.5 | 125.2 | +6.5% |
| 1 | 0.5 | 112.5 | 115.9 | +3.4% |
| 2 | 1 | 104.5 | 107.4 | +2.6% |
