# Figure 6: supported global field and synthesis framework

This package renders the approved Mollweide map with the equatorial dashed locator and dagger. The selected source PDF is byte-identical to Figure6.pdf in both the current manuscript derivative and the author-designated English Nature Communications source project (SHA-256: 6328d98b4841090a7f60392a2f73edb0e239a60d807dfa6c11002714135eca85).

## Run

From the repository root, configure LRVF_DATA_DIR to the reproduction bundle's data directory as described in the main README. LRVF_OUTPUT_DIR optionally changes the output directory.

```text
python scripts/figure6/plot_figure6.py
python scripts/figure6/build_field.py
```

The first command renders the frozen approved field. The second rebuilds the field and CSV from event-quality arrays and event-to-NAC identifiers, writing outputs under figure6 within LRVF_OUTPUT_DIR. To render that rebuilt NPZ, set LRVF_FIGURE6_FIELD to its path and rerun the plot command. This does not replace the bundled frozen input.

Plot inputs:

- figure6/fig6v4_field.npz

Additional analysis inputs:

- figure6/fig4_data.npz
- figure2/cache/nac_ids.npy

The frozen NPZ and accompanying CSV are also available under source_data/figure6 in the code repository for inspection. The CSV contains a per-cell section and a separate zonal-profile section; comment rows and the intervening blank row are intentional.

## Frozen population and analysis

The high-quality core contains 456,253 mapped features from 76,194 contributing NAC products out of the 1,026,786-feature catalogue. The intersection of quality criteria is: matched image metadata; incidence angle 35–75 degrees inclusive; pixel scale at most 1.0 m per pixel; non-summed acquisition; detection confidence at least 0.5713289. The figure annotation rounds the last threshold to 0.571 only for display; substituting that rounded value in the analysis would select 457,053 features and would not reproduce the approved field.

A mapped feature is a retained directed record after cross-image deduplication; it is not necessarily a temporally distinct occurrence. Bearing is clockwise from north. The poleward component is cos(bearing) north of the equator and -cos(bearing) south of it. Encoded line separation is not a physical travel distance.

The grid has 6-degree longitude and latitude cells (30 latitude rows and 60 longitude columns). Cell intervals use lower-inclusive, upper-exclusive bin assignment, with endpoint clipping at the global bounds. Cell means weight mapped features equally. Per-cell 95% intervals resample contributing NAC products as clusters, with 2,000 ordinary cluster-bootstrap replicates and seed 20260715; a cluster's event sum and count are resampled together.

| Setting | Ordinary cells | Sparse polar cells: centre absolute latitude at least 66 degrees | Outer sparse cells: centre absolute latitude at least 72 degrees |
|---|---:|---:|---:|
| Minimum mapped features | 25 | 10 | 5 |
| Minimum contributing NAC products | 3 | 2 | 2 |
| Maximum share from a single NAC product | 0.90 | 0.90 | 0.90 |
| Maximum interval width | 0.50 | 0.80 | 0.80 |
| Minimum reliable image-centre count for the polar support test | 100 | 10 | 5 |
| Minimum reliable fraction for the polar support test | 0.10 | 0 | 0 |

The image-centre support test applies only to rows with centre absolute latitude at least 60 degrees. Image reliability uses the same incidence, resolution and non-summed criteria, without the event detection-confidence criterion. The frozen cache contains 1,053,697 image-centre metadata rows. These rows are separate from the 395,071 author-confirmed surveyed NAC products; do not interpret their count as the survey-image count, or the centre-based support test as exact footprint coverage.

A cell is classified as poleward or equatorward only if its cluster-bootstrap interval excludes zero and the mean magnitude clears 0.03 in that direction. Otherwise an eligible cell is marked as having no resolved directional preference. Ineligible cells remain insufficient. Sparse-tier eligible cells carry the CSV quality flag ok_sparse_polar.

## Zonal intervals and display operations

The marginal profile aggregates raw high-quality mapped features in 6-degree latitude bins, retaining bins with at least 200 features. Its intervals are event-level normal approximations, mean +/- 1.96 times sample standard deviation divided by the square root of the number of features. They are not NAC-cluster-bootstrap intervals. Its regime labels use these normal intervals plus the same 0.03 effect threshold. This inherited method is preserved for reproduction; interpretation should retain the manuscript's qualification that shared-image dependence is not represented by these profile intervals.

The map uses a display smoothing layer with Gaussian standard deviations 0.55 latitude cells and 2.2 longitude cells, wrapping in longitude and clipping at latitude boundaries. A normalized weight above 0.25 is required. The plotted layer subsequently fills missing cells by 400 iterations of longitude-wrapped Laplace relaxation. These operations are display-only; they do not create analysed observations or replace raw cell statistics. Fifteen contour levels span -0.25 to +0.25. Polar caps are hatched beyond the last row having at least 15% finite display cells; for the frozen field those boundaries are 78 degrees north and south.

The anomaly display box spans 180–112 degrees W and 5 degrees S–13 degrees N. The dagger is at 106 degrees W, 11.5 degrees N; the explanatory note identifies strongest cells at approximately 120–165 degrees W. Panel b is schematic and retains the approved artwork.

## Portability and validation

The analysis script adapts the original 01_field.py by resolving input/output paths from the repository configuration and explicitly using UTF-8 for CSV output. The original path to figure4/fig4_data.npz did not exist in this supplied tree; the preserved quality cache is correctly located under figure6. Scientific constants and algorithms are unchanged. The plotting script adapts the latest approved Mollweide-locator derivative with portable paths, an optional field override and concise output names.

An in-memory replay of the original analysis with its input locations corrected independently recovered the 456,253-feature, 76,194-product core. The field, display field, support mask, classes, latitude/longitude centres, counts, regimes and scalar settings exactly matched the frozen NPZ. Profile means and interval bounds matched to floating-point roundoff (maximum absolute difference below 2.8e-17). Event-quality latitudes exactly matched the Figure 3 event cache, and its NAC identifier lookup exactly matched the Figure 2 per-event NAC identifier array.

Dependencies are NumPy for computation, and Matplotlib, Cartopy and Pillow for rendering. The approved design requests Arial; install that font for the closest typography. The original NAC identifier NPY is an object array and is loaded with allow_pickle=True; use the checksum-verified bundled file. Destination hashes and source selections are recorded in provenance_figure6.json. All original project files are preserved.
