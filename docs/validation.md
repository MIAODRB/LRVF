# Validation of the packaged code

Validated locally on 1 October 2026, using Python 3.12.14 on Windows and the versions pinned in `requirements.txt`. Validation dependencies and generated outputs were stored outside the GitHub upload directory.

## Executed checks

| Check | Result |
|---|---|
| Syntax parsing | All 21 Python files passed |
| Main runner input/dependency check | Figures 2–6 passed, including SHA-256 verification |
| Production rendering | All five production entry points exited successfully; PDF/PNG outputs were generated and visually inspected |
| Figure 2 grid/density preparation | Every frozen grid/density array reproduced exactly |
| Figure 2 raw catalogue extraction | All seven event arrays and the per-record NAC identifiers reproduced exactly |
| Figure 2 zonal-strip preparation | Frozen CSV reproduced byte for byte |
| Figure 3 raw cache reconstruction | All eight frozen arrays reproduced exactly from the local direction catalogue |
| Figure 3 statistical output | Source Data CSV reproduced byte for byte |
| Figure 4 statistical outputs | Curve and high-latitude CSVs reproduced byte for byte; two forest effect values differed only in the final floating-point digits (≤2×10⁻¹⁷), with all other values unchanged |
| Solar geometry reconstruction | NAC IDs and two source azimuth arrays identical; solar bearings differed by at most 5.7×10⁻¹⁴ degrees |
| Figure 5 full raw-data preparation | 3,000 bootstraps and 1,500 rotations completed; all event/footprint/terrain/distribution arrays identical after the marker-span key alias; conditional-model CSV byte-identical and summary identical after the key alias |
| Figure 6 field reconstruction | Field, support, classification, population and regime arrays identical; three zonal arrays differed by at most 2.8×10⁻¹⁷ |
| Public upload scan | No old machine-specific execution paths or access-token URLs found |
| Companion ZIP import | All 32 files imported and checked successfully; checksum/dependency checking also passed when invoked from another working directory |

The regenerated core is 456,253 records from 76,194 NAC products. The independently reconstructed terrain set is 2,407 records from 14 NAC products; its primary alignment is 0.764203667640686 and its conditional model has 1,520 observed cases.

The full global raw catalogue and metadata index were read locally for the upstream checks without copying them into the GitHub repository. Remote NAC/SLDEM downloads were not exercised; frozen crops support offline plotting.

New PDFs are not asserted byte-identical to the earlier PDFs: font, rendering-library and PDF metadata differences are expected. Scientific results are compared against the frozen numerical files, with exact or explicitly stated floating-point equivalence.

The accompanying preparation/package audit verifies that pre-existing selected sources and frozen copies remain unchanged. The manuscript and existing figure assets were not overwritten.
