# Data Provenance: DeepLift Attribution Analysis

## Chain of Custody

```
source_data/formatted_real_PPIs.csv   (1,084 real Y2H pairs)
source_data/formatted_random_PPIs.csv (1,084 random MED4 pairs)
          │
          ▼
code/captum_deeplift_proper_analysis.py
  Method: Captum DeepLift (reference-based backpropagation)
  Model:  ppiGPT (GPT-2, 12L/12H/768D, ~85M params)
  Ckpt:   /home/drn2/Desktop/PPIs/MED4/ppiGPT_MED4_solo/out_3e/ckpt.pt
  Date:   2025-06-29
          │
          ▼
results/deeplift_motif_analysis_results.pkl  (78 MB)
  Contains: per-residue attributions, interaction probabilities,
            motif discovery, position-wise statistics for all 2,168 pairs
          │
          ├──▶ code/create_explainability_dashboard.py
          │      Reads pkl → computes t-test, Cohen's d, Mann-Whitney U,
          │      KS test, correlations, motif counts
          │      Output: figures/DeepLift-gptPPI-sixPanel.pdf
          │      Date: 2026-01-26
          │
          └──▶ code/create_pair_attribution_heatmap.py
                 Reads pkl → selects 50 diverse pairs per dataset
                 (evenly spaced rank positions from prediction-sorted results)
                 Output: figures/DeepLift-gptPPI-FIGURE.pdf
                 Date: 2026-01-26
```

## Original File Locations

| Supplement Path | Original Location |
|----------------|-------------------|
| `code/captum_deeplift_proper_analysis.py` | `JUN2025/sureLLM-based/captum_deeplift_proper_analysis.py` |
| `code/create_explainability_dashboard.py` | `sureLLM-based/twoGuardsGPTexplainability/create_explainability_dashboard.py` |
| `code/create_pair_attribution_heatmap.py` | `sureLLM-based/create_pair_attribution_heatmap.py` |
| `code/figure_config.py` | `sureLLM-based/twoGuardsGPTexplainability/visualization/figure_config.py` |
| `source_data/formatted_real_PPIs.csv` | `sureLLM-based/twoGuardsGPTexplainability/data/formatted_real_PPIs.csv` |
| `source_data/formatted_random_PPIs.csv` | `sureLLM-based/twoGuardsGPTexplainability/data/formatted_random_PPIs.csv` |
| `results/deeplift_motif_analysis_results.pkl` | `JUN2025/sureLLM-based/deeplift_motif_analysis_20250629_091944/deeplift_motif_analysis_results.pkl` |
| `figures/DeepLift-gptPPI-FIGURE.pdf` | `INTERACTOMICS_EXPLAINABILITY/DeepLift-gptPPI-FIGURE.pdf` |
| `figures/DeepLift-gptPPI-sixPanel.pdf` | `INTERACTOMICS_EXPLAINABILITY/DeepLift-gptPPI-sixPanel.pdf` |
| `documentation/results_and_figure_legends.txt` | `INTERACTOMICS_EXPLAINABILITY/DeepLift_RESULTS_and_LEGENDS_20260126_162252.txt` |

All paths are relative to `/Users/drn2/Documents/CYANO-INTERACTOME/`.

## Excluded Materials and Rationale

| File | Reason for Exclusion |
|------|---------------------|
| `sureLLM-based/full_deeplift_analysis.py` | Superseded pipeline using input x gradient (not Captum DeepLift); produced different statistics (mean predictions 0.7252/0.6968 vs authoritative 0.718/0.207) |
| `JUN2025/sureLLM-based/full_deeplift_results/` | Output from superseded pipeline |
| `sureLLM-based/deeplift_motif_discovery_full.py` | Motif discovery already performed inside `captum_deeplift_proper_analysis.py` |
| `sureLLM-based/DEEPLIFT_ATTRIBUTION_CORRECTED_EXPLANATION.md` | Internal lab notes, not supplement material |
| `sureLLM-based/PRESENTATION_DEEPLIFT_SUMMARY.md` | Presentation aid; contains outdated Cohen's d value (3.11 vs authoritative 1.91) |
| `JUN2025/sureLLM-based/deeplift_motif_analysis_20250629_091442/` | Earlier pkl (timestamp 091442 vs authoritative 091944) |
| `sureLLM-based-2/` | Backup copies |
| Test/debug scripts (`test_*.py`, `inspect_*.py`) | Development artifacts |

## Data Integrity Certification

- NO synthetic, simulated, or randomly generated data was used as a substitute
  for real experimental results in any analysis or visualization.
- All attribution values derive from Captum DeepLift applied to the trained
  ppiGPT model checkpoint on real protein sequence inputs.
- Random protein pairs (RRS) are genuine random pairings of MED4 proteins,
  not simulated interaction data.

## Date

Provenance document generated: 2026-01-26
