# Prochlorococcus Interactome Model Explainability

Interpretability and explainability analysis of ppiGPT protein-protein interaction predictions in *Prochlorococcus* MED4.

This repository contains all code, data, and results for the model interpretability analyses described in the accompanying manuscript.

## Overview

ppiGPT is a 12-layer, 12-head, 768-dimensional GPT-2 architecture (84.98M parameters) trained from scratch with a 29-token character-level vocabulary (20 amino acids + 9 special tokens) to predict protein-protein interactions. This repository provides the interpretability pipeline used to understand what sequence features drive the model's predictions.

## Repository Structure

```
.
├── model/                          # ppiGPT model architecture
│   └── model.py                    # GPT-2 based PPI prediction model (GPTConfig, GPT)
│
├── data/                           # Input datasets
│   ├── formatted_real_PPIs.csv     # 1,084 Y2H-validated interactions (PRS)
│   └── formatted_random_PPIs.csv   # 1,084 randomly paired MED4 proteins (RRS)
│
├── analysis/                       # Interpretability methods
│   ├── deeplift/                   # Captum DeepLift attribution analysis
│   ├── integrated_gradients/       # Captum Integrated Gradients
│   ├── gradient_attribution/       # Gradient-based attribution
│   ├── lrp/                        # Layer-wise Relevance Propagation
│   ├── perturbation/               # Alanine substitution scanning
│   ├── attention/                  # Attention pattern extraction and analysis
│   ├── counterfactual/             # Counterfactual sequence generation
│   ├── probes/                     # Linear probing of internal representations
│   ├── motif_discovery/            # Attribution-guided motif discovery
│   └── uncertainty_quantification.py
│
├── af3_structural_analysis/        # AlphaFold3 N-terminal ablation experiments
│   ├── submit_to_af3_batch.py      # AF3 batch submission
│   ├── analyze_af3_results.py      # AF3 output parsing
│   └── analyze_med4_results_20251011.py  # ipTM analysis of N20A ablations
│
├── visualization/                  # Figure generation
│   ├── create_explainability_dashboard.py  # Six-panel summary dashboard
│   ├── create_pair_attribution_heatmap.py  # Per-residue heatmaps
│   └── figure_config.py            # Matplotlib style configuration
│
├── results/                        # Analysis outputs
│   └── integrated_gradients_*.csv           # IG attribution tables
│
├── figures/                        # Publication figures
│   ├── DeepLift-gptPPI-FIGURE.pdf           # Per-residue attribution heatmaps
│   ├── DeepLift-gptPPI-sixPanel.pdf         # Summary dashboard
│   └── *.svg                                # Vector figure components
│
└── documentation/
    ├── data_provenance.md           # Full chain of custody for all outputs
    └── results_and_figure_legends.txt
```

## Key Results

| Metric | PRS (Real PPIs) | RRS (Random) |
|--------|-----------------|--------------|
| Mean prediction | 0.718 +/- 0.347 | 0.207 +/- 0.152 |
| Mean \|attribution\| | 0.0081 +/- 0.0047 | 0.0083 +/- 0.0047 |
| Unique motifs (3-5 mers) | 80,119 | 270,576 |

| Test | Statistic | p-value |
|------|-----------|---------|
| Two-sample t-test | t = 44.36 | 2.76 x 10^-306 |
| Mann-Whitney U | U = 1,018,862 | 1.64 x 10^-192 |
| Cohen's d | 1.91 (large) | -- |

## Large Files Not Included

The following files are excluded due to size but can be regenerated from the analysis scripts:

- `results/deeplift_motif_analysis_results.pkl` (78 MB) — complete Captum DeepLift attribution arrays for all 2,168 protein pairs, motif discovery results, and position-wise statistics. Regenerate with `analysis/deeplift/captum_deeplift_proper_analysis.py`.
- `results/integrated_gradients_random_ppi_per_token_attributions.csv` (174 MB) — per-token IG attributions. Regenerate with `analysis/integrated_gradients/captum_integrated_gradients_enhanced.py`.

## Model Checkpoint

The trained ppiGPT checkpoint (`out_3e/ckpt.pt`) is not included in this repository due to size. Place the checkpoint at `model/out_3e/ckpt.pt` and the tokenizer metadata at `model/data/meta.pkl` before running analysis scripts.

## Reproduction

```bash
pip install -r requirements.txt

# 1. DeepLift attribution analysis (requires GPU + model checkpoint)
python analysis/deeplift/captum_deeplift_proper_analysis.py

# 2. Generate summary dashboard from results
python visualization/create_explainability_dashboard.py

# 3. Generate per-pair heatmaps
python visualization/create_pair_attribution_heatmap.py
```

## Software

- Python 3.10+
- PyTorch >= 2.0.0
- Captum (DeepLift, Integrated Gradients)
- scipy, numpy, matplotlib, seaborn, pandas

## License

MIT License. See [LICENSE](LICENSE).
