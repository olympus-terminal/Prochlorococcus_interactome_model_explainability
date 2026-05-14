# Explainability Analysis of ppiGPT Interaction Predictions in *Prochlorococcus* MED4

This repository contains code, data, and results for the model explainability analyses described in Daakour et al., "Topological entrenchment of adaptive proteins in the streamlined interactome of *Prochlorococcus* MED4." These analyses investigate what sequence features drive the predictions of ppiGPT, a protein-protein interaction model created by Kourosh Salehi-Ashtiani.

## Overview

The explainability pipeline applies multiple interpretability methods to ppiGPT predictions across 1,084 experimentally validated Y2H interactions (PRS) and 1,084 randomly paired MED4 proteins (RRS). Methods include DeepLift attribution, Integrated Gradients, gradient attribution, Layer-wise Relevance Propagation, alanine substitution scanning, attention analysis, counterfactual sequence generation, linear probing, and AlphaFold3 N-terminal ablation experiments.

## Repository Structure

```
.
├── model/                          # ppiGPT model architecture (created by K. Salehi-Ashtiani)
│   └── model.py                    # GPT-2 architecture definition, included for reproducibility
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

## Large Files (Hugging Face)

Large result files and the ppiGPT checkpoint (included for reproducibility) are hosted on Hugging Face:

**https://huggingface.co/GreenGenomicsLab/Prochlorococcus_interactome_model_explainability**

Download and place them locally before running analysis scripts:

| HF Path | Size | Description |
|---------|------|-------------|
| `results/deeplift_motif_analysis_results.pkl` | 78 MB | DeepLift attribution scores for all 2,168 protein pairs |
| `results/integrated_gradients_random_ppi_per_token_attributions.csv` | 174 MB | Per-token IG attributions for 1,084 RRS pairs |
| `model/out_3e/ckpt.pt` | 1.0 GB | ppiGPT checkpoint (K. Salehi-Ashtiani), included for reproducibility |
| `model/data/meta.pkl` | 343 B | Tokenizer metadata |

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

## Citation

This repository is part of:

> Daakour et al., "Topological entrenchment of adaptive proteins in the streamlined interactome of *Prochlorococcus* MED4."

## License

MIT License. See [LICENSE](LICENSE).
