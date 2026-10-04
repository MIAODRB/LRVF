# Reproducibility scope and inherited limits

This is a packaging and portability pass. It preserves the statistical implementation that generated the submitted display figures. It does not revise the scientific methods or replace any existing manuscript asset.

## Frozen and reconstructed components

Figures 2 and 6 render frozen gridded fields and profiles. Figure 3 recalculates its bootstrap profiles and directional-density display from the frozen event cache. Figure 4 recalculates its observation-control statistics from aligned event, quality, NAC metadata and solar caches. Figure 5 renders frozen terrain-alignment and conditional-model outputs. Optional upstream scripts are provided separately; raw-data regeneration requires the external catalogue and metadata described in the figure guides.

The final production Figure 2, 3, 4, 5 and 6 PDFs were matched by SHA-256 to their selected source output before packaging. Figures 2/6 originate from approved later Mollweide directories; Figure 3 and the final Figure 4/5 drivers originate from `paper/fig_v2/scripts`. Selecting those versions avoids obsolete displays and a missing Figure 3 v2 import.

## Statistical limits retained

- Figure 2's zonal-strip confidence intervals use an event-level normal approximation. They are not NAC-cluster bootstrap intervals.
- Figure 3 uses the production v3 bootstrap and heatmap algorithms. Its reversal markers (63.48970201322201° N and 66.66960727698721° S) are frozen values inherited from the preceding persistent-crossing analysis. The production script does not independently recompute these markers.
- Figure 4 preserves its 800-replicate NAC-cluster bootstrap and its final solar-azimuth strata.
- Figure 5 preserves the frozen 3,000 bootstrap replicates and 1,500 rotations. No population or bootstrap setting is changed during packaging.
- Figure 6's zonal profile and regime classification retain the existing event-level normal-theory intervals. These should not be described as NAC-cluster bootstrap uncertainty. The gridded field's support and effective-sample handling are separate from that profile interval calculation.

## Geometry and metadata limits

The stored direction geometry starts at the transformed oriented-box centre and points toward the polarity-defining boulder endpoint. It is a direction indicator, not a measurement of the true physical start/end of motion. Its span has no physical displacement or runout meaning. The Figure 5 10 m span filter is retained solely as an angular-stability criterion; public preparation code calls it `marker_span_m`. Frozen files with inherited `track_m` or `minimum_track_length_m` keys remain byte-identical and have this clarified interpretation.

Legacy Figure 2 gridding normalizes the two endpoint longitudes independently before taking their arithmetic midpoint. Longitude-seam geometries can consequently be assigned differently if that legacy rule is corrected. The frozen submitted grid is preserved; this pass does not silently correct it.

The Figure 6 support cache contains 1,053,697 archive metadata image-centre entries. These are not the author-selected 395,071 surveyed products, nor the 169,857 products contributing catalogue records. The inherited spatial support mask is documented in the Figure 6 guide.

## Output equivalence

Numerical comparisons use the frozen CSV/JSON/NPZ outputs. PDF hashes from a new software environment need not match because font metrics, PDF metadata and rendering libraries can differ. The figure documents record exact inputs, thresholds and seeds, while the validation report distinguishes numerical reproduction from byte-identical artwork.

This repository contains figure analysis, not the detector deployment, image selection or cross-image deduplication pipeline. It should not be cited as the complete global catalogue-construction code.
