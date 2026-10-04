# Figures 2 and 3

These scripts reproduce the displays approved for the submitted manuscript. Figure 2 follows the September 2026 Mollweide version with uniform longitude thinning, the zonal strip restored and no blue direction overlay on the NAC close-up. Figure 3 follows the final v3 script; the earlier v2 wrapper and its legacy plotting/bootstrap algorithms are excluded. Scientific computations and display settings are preserved.

## Run the frozen figures

Install the repository dependencies first, then run from the repository root. If the companion reproduction data are still in the sibling folder:

```sh
python scripts/figure2/plot_figure2.py --cache-dir ../reproduction_data/data/figure2/cache --zonal-source ../reproduction_data/data/figure2/figure2_zonal_strip_source_data.csv
python scripts/figure3/plot_figure3.py --event-cache ../reproduction_data/data/figure3/figure3_event_cache.npz
```

With the data staged in `data/`, omit these arguments. The shared path module also accepts `LRVF_DATA_DIR` and `LRVF_OUTPUT_DIR`. Outputs default to `outputs/figure2/` and `outputs/figure3/`. Figure 2 exports PDF, 600-dpi PNG and a 200-dpi preview. Figure 3 exports PDF, 600-dpi PNG/TIFF, its source CSV and summary JSON.

## Inputs and dependencies

| Figure | Mandatory input below the data directory | Libraries |
|---|---|---|
| 2 | `figure2/cache/density_0p25.npy`, `grid_6deg.npz`, `events.npz`, `panelb_dem.npy`, `nac_crop_0.npy`, `nac_crop_0_origin.npy`; `figure2/figure2_zonal_strip_source_data.csv` | NumPy, SciPy, Matplotlib, Cartopy |
| 3 | `figure3/figure3_event_cache.npz` | NumPy, SciPy, Matplotlib |

All mandatory binary inputs are copied without changes into the companion data folder. Small frozen source CSV/JSON files are also retained in `source_data/figure2/` and `source_data/figure3/`. The global catalogue and full CUMINDEX are not bundled with the code. Figure 2 no longer loads `nac_ids.npy`: the panel-c direction overlay using those IDs was removed from the approved display.

## Frozen settings

Figure 2 uses a 0-degree-centred Mollweide projection, the full 1,026,786-record catalogue, a 6-degree direction grid, regular support of at least 30 records and relaxed support of at least 5 records poleward of 60 degrees. Arrow length represents grid-cell resultant length, and colour represents the cell mean poleward component. Longitude display stride is at least two cells, with additional thinning toward the poles. Density is aggregated to 0.5-degree cells and smoothed for display with sigma 1.2 cells. Hercules G glyphs have fixed display length and are thinned on a 0.9-mm occupancy grid with random seed 42. These display symbols do not represent physical displacement. The six-degree zonal strip uses the original event-level normal approximation `mean +/- 1.96 * sample_sd / sqrt(n)` with minimum bin count 200 and four-decimal CSV serialization; these intervals are not NAC-cluster intervals.

Figure 3 uses the full catalogue cache with float32 latitude, bearing, relative angle and poleward component, int32 NAC codes and 169,857 NAC products. Panel a uses 36 ten-degree bearing bins. Panel b uses two-degree absolute-latitude bins, Gaussian smoothing of numerator/denominator with sigma one bin, 800 NAC-product Poisson-bootstrap replicates and seeds 20260715 (north) and 20260716 (south). Support requires at least 500 records and 10 NAC products; the profile is truncated at the first failing bin at or beyond 70 degrees. Panel c normalizes the raw five-degree-angle by one-degree-latitude histogram within each latitude column, then applies display smoothing with sigma `(1.1, 1.0)` in angle/latitude order. It truncates unsupported polar columns.

The Figure 3 reversal annotations retain the frozen run values 63.48970201322201 degrees N and 66.66960727698721 degrees S (midpoint 65.07965464510461 degrees). They were inherited from the earlier persistent-crossing calculation and are constants in this final plotting script; they are not newly estimated by this run. The final Figure 3 colour limits are also frozen. Do not substitute the legacy v2 heatmap construction or bootstrap seeds.

## Optional upstream preparation

These preparation scripts write to `outputs/prepared/` by default, preserving the frozen input data. Supply your downloaded two-point direction catalogue and optional raster files explicitly:

```sh
python scripts/figure2/extract_events.py --input /path/to/GLOBAL_rockfall_directions_dedup.shp
python scripts/figure2/grid_statistics.py --events outputs/prepared/figure2/cache/events.npz
python scripts/figure3/build_event_cache.py --input /path/to/GLOBAL_rockfall_directions_dedup.shp
python scripts/figure2/prepare_zonal_strip.py --event-cache outputs/prepared/figure3/figure3_event_cache.npz
python scripts/figure2/prepare_dem_crop.py --dem /path/to/Lunar_LRO_LOLAKaguya_DEMmerge_60N60S_512ppd.tif
python scripts/figure2/prepare_nac_crop.py --nac /path/to/M142007158LC_pyr.tif
```

`extract_events.py` additionally needs Pyogrio, GeoPandas and Shapely 2. The two regional raster preparation scripts need Rasterio/GDAL. Their defaults retain the original upstream GDAL-readable remote URLs; network availability and remote raster versions can change. The frozen cropped arrays allow the submitted plots to run offline. The NAC preparation script retains only the selected first crop; candidate crops and debug overlays are excluded. The DEM is used for display hillshade in Figure 2, not for the local slope statistics in Figure 5.

Install the additional raw-catalogue dependencies with `python -m pip install -r requirements-upstream.txt` before running `extract_events.py`.

To draw newly generated Figure 2 inputs, pass `--cache-dir outputs/prepared/figure2/cache --zonal-source outputs/prepared/figure2/figure2_zonal_strip_source_data.csv`. To draw a rebuilt Figure 3 cache, pass `--event-cache outputs/prepared/figure3/figure3_event_cache.npz`. Preparation output folders are replaceable generated results; choose an explicit `--output-dir` to retain multiple runs.

`build_event_cache.py` adapts only the original strict SHP/DBF binary reader, not the obsolete Figure 3 visualization or bootstrap algorithms. It expects one-part, two-point PolyLine records and the original DBF schema (`nac_id` C80, `score` N24,15, `lat_zone` C80). The source reader was recovered from the original Figure 3 analysis project because the supplied v2 wrapper lacked that module. Its inputs are supplied through arguments; it has no dependency on the original project location.

The legacy Figure 2 gridding calculates an arithmetic midpoint after normalizing each endpoint longitude separately. A direction indicator crossing the longitude seam can therefore be assigned an inappropriate longitude. This inherited implementation is preserved for exact reproduction of the submitted frozen grids; correcting it would constitute a separate analysis revision. The Figure 3 cache reader wraps longitude differences for bearing and uses mean endpoint latitude. Stored endpoints encode polarity and position; endpoint separation, including the legacy `length_m` field, has no physical trajectory-length or displacement meaning.

## Verification and provenance

All eight new Python scripts passed syntax parsing. The curated grid/density builder reproduces every frozen grid array exactly. The zonal-strip builder reproduces the frozen CSV byte for byte. The strict Figure 3 cache reader reproduces all eight frozen cache arrays exactly from the current catalogue. These checks use the local catalogue without adding it to the upload folder. Full plotting verification is recorded in the repository validation report. Font availability and library versions may change PDF metadata and raster appearance without changing scientific statistics.

The approved Figure 2 PDF is byte-identical to the corresponding formal manuscript asset, with SHA-256 `d5b71440baa07d6d2cbde4f366bb24c4c80831f63fa021029534e6006b3468f1`. The approved Figure 3 PDF similarly has SHA-256 `fb07d1d5e23f0d6dc5627d8b3401d6a12086c6c4a531beb98a3adf0fe981cbe9`. `provenance_figure2_3.json` records the source path, destination and original source SHA-256 for each adapted/copied file. All pre-existing project files remain unchanged.
