# Frozen figure Source Data

These are small reference files copied without changing their numerical values. Larger arrays needed for reproduction are in the separate data bundle. Newly generated statistics go to `outputs/`; the files here remain reference inputs for comparison.

| Folder | Files and interpretation |
|---|---|
| `figure2/` | Six-degree zonal mean, event-level normal interval and counts |
| `figure3/` | Hemispheric profile Source Data and final v3 summary |
| `figure4/` | Observation-stratum mid-latitude effects, full/core latitude curves, hemisphere-specific high-latitude effects and final summary |
| `figure5/` | Terrain-scale alignment statistics, conditional source-slope odds ratios and frozen terrain-analysis summary |
| `figure6/` | Frozen field NPZ and cell-level Source Data CSV |

General conventions:

- Angular values are in degrees. Trajectory bearing is clockwise from north; directed rockfall–downslope separation spans 0–180°.
- `effect` / poleward component is positive poleward and negative equatorward. It is dimensionless.
- `ci_low` and `ci_high` are the interval endpoints from the particular figure's implementation; see the figure-specific guides for clustering and interval methods.
- `n_events` counts retained direction indicators. `n_images` counts contributing NAC products, not distinct physical failures.
- Figure 5's `odds_ratio` is a conditional observed-case comparison, not an absolute activity rate. A confidence interval spanning one does not establish a conclusive slope preference.
- Figure 6 support/classification fields describe cell eligibility, field construction and displayed regimes; consult `docs/figure6.md` for thresholds. Its frozen `n_hq` is 456,253.

Legacy `track_m`, `length_m` and `minimum_track_length_m` cache/summary keys refer to the span between direction-marker endpoints. They do not measure travelled distance or physical track length. Public preparation code uses marker-span terminology where applicable, while frozen reference bytes remain preserved.
