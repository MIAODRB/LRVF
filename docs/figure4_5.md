# Figures 4 and 5: analysis and reproduction

Run the commands below from the repository root after configuring `LRVF_DATA_DIR` to the reproduction data folder (or importing that folder into `data/`). Generated files go to `outputs/` by default; `LRVF_OUTPUT_DIR` overrides that location. The scripts read frozen inputs and write generated outputs to separate folders.

## Figure 4

```console
python scripts/figure4/plot_figure4.py --bootstrap 800
```

This is the production entry point. It reuses the validated base implementation and retains the later sub-solar azimuth East/West stratum, the 30–60° label and the 150 mm production figure height. Running `plot_figure4_base.py` directly would omit those production patches.

Required frozen inputs:

- `figure3/figure3_event_cache.npz`: catalogue directions, poleward components, latitudes and NAC cluster indices.
- `figure4/figure4_outputs/nac_metadata_cache.npz`: observing metadata, indexed by the same NAC products.
- `figure6/fig4_data.npz`: event score array; this array was verified to equal every raw DBF score after float32 conversion.
- `solar_per_image.npz`: ground-frame bearing toward the sub-solar point. The production entry point checks its NAC IDs against the event cache before using it.

The final statistics are also archived in `source_data/figure4/`. Frozen settings are 800 Poisson bootstrap replicates, NAC-product clusters, base seed 20260714 (individual effects use the original offsets), 2° absolute-latitude profile bins, Gaussian smoothing with sigma 1 bin, and displayed support of at least 500 records and 10 NAC products per bin. The frozen effect bands are 30° ≤ |latitude| < 60° and 68° ≤ |latitude| < 76°. The reference reversal interval is 63.48970201322201–66.66960727698721°.

The high-quality intersection contains 456,253 records in 76,194 NAC products: incidence 35–75°, pixel scale ≤ 1.0 m/pixel, 5,064 line samples and score ≥ 0.5713289. Observation splits retain incidence 52.18°, pixel scale 0.652 m/pixel, score 0.5713289, LEFT/RIGHT NAC frames, 5,064/2,532 samples, and solar-bearing half-planes 0–180°/180–360°. The solar split is an illumination-direction control; it is not a pixel-level shadow correction.

Optional preparation commands:

```console
python scripts/figure4/prepare_figure4_data.py
python scripts/figure4/prepare_solar_geometry.py
```

The first command checks or prepares observing metadata and produces a summary using the frozen inputs. The second rebuilds the per-image solar geometry from `cumindex_solar_extract.csv.gz` and writes `outputs/solar_per_image.npz` for comparison. It preserves the original first-product-match join and prefers the great-circle bearing from image centre toward the sub-solar point, falling back to the image-frame estimate where needed. It does not replace the frozen input file. Rebuilding observing metadata after removing its frozen cache additionally requires the exact raw `CUMINDEX.TAB` snapshot in `LRVF_DATA_DIR` (902-byte fixed records). A raw DBF is required only when the compact frozen score cache is unavailable.

## Figure 5

```console
python scripts/figure5/plot_figure5.py
```

This is the production entry point and retains the annotation explaining the expected peak near 180° under a systematic direction-polarity swap. It reads the frozen `figure5/figure5_outputs/` panel-a terrain cache, panel-b distribution, conditional-odds CSV and summary. The bundle also archives the event-terrain cache, per-NAC footprint cache and terrain-scale statistics. The final CSV/JSON reference outputs are copied to `source_data/figure5/` without changing their values or legacy metadata keys.

Frozen primary settings are the preselected `NAC_DTM_CHAPLYGIN` v1.9 5 m/pixel DTM, Moon radius 1,737,400 m, central longitude 180°, standard parallel −4°, 25 m normalized Gaussian terrain scale, valid-data support ≥ 0.98, slope ≥ 5°, detection score ≥ 0.50, incidence 35–75°, pixel scale ≤ 1.20 m/pixel, and 5,064 samples. Terrain-scale sensitivity uses 15/25/50 m. There are 2,407 retained records in 14 NAC products, with primary directed downslope alignment D = 0.7642036676, median angle 15.54296875°, and 74.28334026% within 30°. The frozen output explicitly records 3,000 NAC-cluster Poisson bootstrap replicates and 1,500 independent NAC-cluster rotations, base seed 20260714 with the original offsets.

The conditional source-slope comparison uses nominal footprint opportunity sampled at 50 m, matched by NAC product, slope bin, elevation tercile and curvature tercile. The primary aspect half-window is 60°; 45° and 75° are sensitivity checks. It has 1,520 observed cases in 139 primary matched strata. Its confidence interval spans one, so it remains an uncertain observation-conditioned odds comparison. Nominal footprint opportunity is not effective searchable area and is not an absolute activity-rate denominator. The independently selected DTM footprint catalogue is archived for selection provenance; this preparation script starts with the frozen selected DTM and does not repeat the original global site ranking.

### Stored marker span

The legacy preparation code filters on a minimum 10 m great-circle span between the two stored direction-marker endpoints. These catalogue lines encode direction only; their endpoint span is **not measured track length or physical runout**. The numerical mask is retained to reproduce the frozen 2,407-record population. Public copied-code names and new preparation summaries now call it `marker_span_m` / `minimum_marker_span_m`; byte-identical frozen reference files retain the original `track_m` / `minimum_track_length_m` keys. This naming clarification changes no direction, event selection, estimate, bootstrap or null calculation. Example arrows are fixed-length display symbols.

### Optional rebuild from external raw inputs

```console
python scripts/figure5/prepare_figure5_data.py --bootstrap 3000 --null-replicates 1500
python scripts/figure5/plot_figure5.py --input-dir outputs/figure5
```

The first command additionally requires `GLOBAL_rockfall_directions_dedup.shp`, its matching `.dbf` and the exact `CUMINDEX.TAB` snapshot directly under `LRVF_DATA_DIR`. The complete Chaplygin GeoTIFF/detached label are provided in the separate reproduction bundle under `figure5/data/`. The code validates the DTM mode/dimensions and two-point PolyLine structure. These fixed binary readers are specific to the frozen archive and are not generic readers for arbitrary shapefiles or metadata versions. Raw inputs and the ~3.64 GB index are kept outside the GitHub code folder. The interrupted CHAPLYGIN2 download is not included or used.

## Provenance and scope

The production Figure 4 and Figure 5 source PDFs under `paper/fig_v2/` are byte-identical to the uncapitalized current manuscript Figure4.pdf and Figure5.pdf assets. The SA copies separately changed panel-label case only; the public entry points retain the NC production artwork. `docs/provenance_figure4_5.json` records every source path, copied/adapted destination and source/destination SHA-256. Path, input/output separation and naming adaptations are confined to the new delivery folder; the original source scripts and data are preserved.
