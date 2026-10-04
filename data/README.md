# Reproduction inputs

The code repository intentionally contains no large event or raster inputs. The companion `LRVF_reproduction_data_20261001.zip` contains a `data/` tree with the exact frozen inputs, plus selected data for upstream preparation. All files have sizes, SHA-256 checksums and figure-use tags in [data_manifest.json](../docs/data_manifest.json).

NPZ field names, shapes and dtypes are listed in [data_array_schema.json](../docs/data_array_schema.json).

Import from the repository root:

```bash
python scripts/import_reproduction_data.py ../LRVF_reproduction_data_20261001.zip
```

The importer checks every file before copying and refuses to overwrite different existing data. Inputs remain separate from generated outputs. Alternatively, pass `--data-dir` to `reproduce.py` and use the extracted bundle directly.

| Data group | Purpose |
|---|---|
| `figure2/cache/` | Frozen event/density/grid arrays, Hercules G DEM crop and NAC crop/origin |
| `figure3/figure3_event_cache.npz` | Event bearings, latitude, poleward components and NAC cluster identifiers |
| `figure4/figure4_outputs/nac_metadata_cache.npz` | NAC quality and observation metadata aligned to event-cache NAC IDs |
| `solar_per_image.npz` | Solar bearing aligned to the same NAC IDs |
| `figure5/figure5_outputs/` | Frozen terrain display, alignment distribution, conditional-model results and provenance |
| `figure6/` | Frozen supported field and full-precision event-quality cache |
| `figure5/data/` | Complete Chaplygin DTM/label and DTM footprint catalogue for optional upstream terrain preparation |
| `cumindex_solar_extract.csv.gz` | Extracted solar metadata for optional rebuilding of the compact solar cache |
| `CUMINDEX_reference.LBL` | Original metadata-column definitions |

The complete global catalogue Shapefile, full `CUMINDEX.TAB`, the global SLDEM2015 raster and original NAC images remain external. The figure-specific guides explain where optional raw inputs are expected. The incomplete CHAPLYGIN2 download is excluded.

The catalogue DOI supplied by the author is [10.5281/zenodo.22805805](https://doi.org/10.5281/zenodo.22805805); this does not imply the companion analysis caches are already deposited there. Exact bytes in this companion bundle are verified locally. No access tokens are included.

Large data and third-party assets retain their original provenance and licensing terms. Archival distribution of this bundle is separate from uploading the code directory.
