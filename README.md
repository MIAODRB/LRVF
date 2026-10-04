# Lunar Rockfall Vector Field (LRVF): Figures 2–6

Code accompanying *A global vector field of lunar rockfalls reveals latitude-organized mass wasting* by Dingruibo Miao, Jianguo Yan, Zhigang Tu, Zhiyong Xiao, Sho Sasaki and Jean-Pierre Barriot.

This package assembles the analysis and plotting code used for the current manuscript's Figures 2–6. It selects the approved Mollweide/no-overlay Figure 2, the v3 Figure 3, the final Figure 4/5 drivers and the Mollweide/equatorial-locator Figure 6. Machine-specific paths have been replaced with configurable locations. Numerical filters, random seeds and statistical methods are preserved. Source Data tables are in `source_data/`.

## Quick start

Use Python 3.12 for the validated environment. From this repository folder:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/import_reproduction_data.py ../LRVF_reproduction_data_20261001.zip
python reproduce.py --check --verify-data
python reproduce.py --figure all
```

The companion ZIP is supplied separately from the code repository. Alternatively, point directly to its extracted data folder:

```bash
python reproduce.py --data-dir ../reproduction_data/data --figure 3
```

Figure files and generated statistics are written to `outputs/figure2/` through `outputs/figure6/`. Run any one figure with `--figure 2` (or 3, 4, 5, 6). `--data-dir` and `--output-dir` accept absolute or relative paths. Direct scripts use the equivalent environment variables `LRVF_DATA_DIR` and `LRVF_OUTPUT_DIR`. Figure 3/4 bootstrap analysis can take several minutes.

## Contents and workflows

| Figure | Main subject                                                   | Production entry point            | Details                         |
| ------ | -------------------------------------------------------------- | --------------------------------- | ------------------------------- |
| 2      | Global distribution, gridded directions and Hercules G example | `scripts/figure2/plot_figure2.py` | [Figure 2/3](docs/figure2_3.md) |
| 3      | Latitude-organized directional structure                       | `scripts/figure3/plot_figure3.py` | [Figure 2/3](docs/figure2_3.md) |
| 4      | Observation-condition controls and high-quality core           | `scripts/figure4/plot_figure4.py` | [Figure 4/5](docs/figure4_5.md) |
| 5      | Terrain alignment and conditional slope comparison             | `scripts/figure5/plot_figure5.py` | [Figure 4/5](docs/figure4_5.md) |
| 6      | Supported high-quality global direction field                  | `scripts/figure6/plot_figure6.py` | [Figure 6](docs/figure6.md)     |

The figure-specific guides describe upstream preparation separately from reproduction with frozen inputs. The main runner renders from frozen analysis caches; it does not download the LROC archive, rerun the detector or reconstruct the global deduplicated catalogue. Raw catalogue and image-metadata inputs remain external. See [data inventory](data/README.md), [data checksums](docs/data_manifest.json), [reproducibility limits](docs/reproducibility.md) and [validation](docs/validation.md).

## Scientific definitions

The global catalogue contains 1,026,786 retained direction indicators from 169,857 NAC products. A record encodes a polarity-resolved indicator of inferred trajectory direction. Its stored line span is not a physical travel distance, runout or velocity. Bearings are clockwise from north. The poleward component is cos(bearing) in the north and −cos(bearing) in the south.

The high-quality core contains 456,253 records from 76,194 NAC products, using detection confidence ≥0.5713289, incidence 35–75°, pixel scale ≤1.0 m/pixel and non-summed acquisitions. The exact confidence threshold must not be replaced by the rounded display label 0.571.

These observations concern the detected illuminated-terrain sample. They do not by themselves identify a unique physical cause.

## Data access and distribution

The author-provided catalogue archive identifier is [10.5281/zenodo.22805805](https://doi.org/10.5281/zenodo.22805805). The identifier alone does not establish that this repository's companion caches have been published there. This package contains no private preview credentials. LROC imagery and cumulative metadata originate from the [LROC PDS archive](https://pds.lroc.im-ldi.com/data/). Exact local inputs and checksums are recorded in the manifests.

Keep large raw data and the companion ZIP outside the Git repository; `data/` inputs and generated outputs are ignored by Git. GitHub's browser uploads are limited to 25 MiB per file, while ordinary Git blocks files above 100 MiB. See [GitHub's official size guidance](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github). Distribute companion data as a separate archival deposit or release asset when publishing the code.

This packaging does not assign a new software or third-party data license. No publication DOI or software-release DOI is invented.

# 
