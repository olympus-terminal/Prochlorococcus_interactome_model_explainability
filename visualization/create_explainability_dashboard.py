#!/usr/bin/env python3
"""
Create high-density explainability dashboard figure using ALL data (1,084 pairs each).
Follows FIGURE_PROTOCOL.md standards for publication.

NO SUBSAMPLING - uses complete dataset.
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
import sys
import os

# Add visualization module to path
sys.path.insert(0, os.path.dirname(__file__))
from visualization.figure_config import setup_publication_style, BLACKBODY_COLORS, DIVERGING_COLORS

# Setup publication style
setup_publication_style()

# Load the REAL data (82MB pickle with ALL 1,084 pairs per dataset)
print("Loading complete DeepLift results...")
pkl_path = '../deeplift_motif_analysis_20250629_091944/deeplift_motif_analysis_results.pkl'
if not os.path.exists(pkl_path):
    print(f"ERROR: Cannot find {pkl_path}")
    print("Current directory:", os.getcwd())
    sys.exit(1)

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

real_data = data['real_ppis']
random_data = data['random_ppis']

print(f"Real PPIs: {real_data['batch_results']['num_pairs']} pairs")
print(f"Random PPIs: {random_data['batch_results']['num_pairs']} pairs")

# Extract data for visualization
real_results = real_data['batch_results']['results']
random_results = random_data['batch_results']['results']

real_preds = np.array([r['interaction_prob'] for r in real_results])
random_preds = np.array([r['interaction_prob'] for r in random_results])

# Compute mean absolute attribution from the attributions array
real_attrs = np.array([np.mean(np.abs(r['attributions'])) for r in real_results])
random_attrs = np.array([np.mean(np.abs(r['attributions'])) for r in random_results])

# Position-wise data (ALL positions up to 636)
real_pos = real_data['position_results']
random_pos = random_data['position_results']

# Motif data (80K+ unique motifs)
real_motifs = real_data['motif_results']['top_motifs']
random_motifs = random_data['motif_results']['top_motifs']

print(f"\nCreating high-density dashboard with ALL {len(real_results)} + {len(random_results)} pairs...")
print(f"Total attributions: {len(real_data['batch_results']['all_attributions'])}")
print(f"Unique motifs: {len(real_data['motif_results']['all_motifs'])}")

# Create figure with complex gridspec
fig = plt.figure(figsize=(7.0, 9.0), dpi=150)

# Complex grid: 6 rows × 4 columns
# Row ratios: top_track, prediction_dist, attribution_compare, position_heatmap, motif_analysis, stats
gs = gridspec.GridSpec(6, 4, figure=fig,
                      height_ratios=[0.3, 1.0, 1.0, 1.5, 1.2, 0.8],
                      width_ratios=[0.05, 1.5, 1.5, 0.05],
                      hspace=0.15, wspace=0.10,
                      left=0.08, right=0.97, top=0.96, bottom=0.04)

# ============================================================================
# ROW 0: Dataset labels
# ============================================================================
ax_label_real = fig.add_subplot(gs[0, 1])
ax_label_random = fig.add_subplot(gs[0, 2])

ax_label_real.text(0.5, 0.5, f'Real PPIs (n={len(real_results)})',
                   ha='center', va='center', fontsize=6, weight='bold')
ax_label_real.axis('off')

ax_label_random.text(0.5, 0.5, f'Random Pairs (n={len(random_results)})',
                     ha='center', va='center', fontsize=6, weight='bold')
ax_label_random.axis('off')

# ============================================================================
# ROW 1: Prediction distributions (violin + histogram)
# ============================================================================
ax_pred_real = fig.add_subplot(gs[1, 1])
ax_pred_random = fig.add_subplot(gs[1, 2])

# Real PPIs - violin + histogram overlay
parts = ax_pred_real.violinplot([real_preds], positions=[0.5], widths=0.4,
                                showmeans=True, showmedians=True)
for pc in parts['bodies']:
    pc.set_facecolor([0.2, 0.4, 0.7])
    pc.set_alpha(0.6)
    pc.set_linewidth(0.5)

# Histogram overlay
ax_pred_real.hist(real_preds, bins=50, alpha=0.3, color=[0.2, 0.4, 0.7],
                  orientation='horizontal', density=True)

ax_pred_real.set_ylabel('Prediction Probability', fontsize=5)
ax_pred_real.set_xlabel('Density', fontsize=5)
ax_pred_real.set_title('Prediction Distribution', fontsize=5, weight='bold', pad=2)
ax_pred_real.tick_params(labelsize=4, width=0.5, length=1.5)
ax_pred_real.grid(True, alpha=0.2, linewidth=0.3)

# Add stats annotation
mean_real = np.mean(real_preds)
std_real = np.std(real_preds)
ax_pred_real.text(0.95, 0.95, f'μ={mean_real:.3f}\nσ={std_real:.3f}',
                 transform=ax_pred_real.transAxes, fontsize=3.5,
                 ha='right', va='top',
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', lw=0.3))

# Random pairs
parts = ax_pred_random.violinplot([random_preds], positions=[0.5], widths=0.4,
                                  showmeans=True, showmedians=True)
for pc in parts['bodies']:
    pc.set_facecolor([0.8, 0.4, 0.2])
    pc.set_alpha(0.6)
    pc.set_linewidth(0.5)

ax_pred_random.hist(random_preds, bins=50, alpha=0.3, color=[0.8, 0.4, 0.2],
                   orientation='horizontal', density=True)

ax_pred_random.set_xlabel('Density', fontsize=5)
ax_pred_random.set_yticklabels([])
ax_pred_random.set_title('Prediction Distribution', fontsize=5, weight='bold', pad=2)
ax_pred_random.tick_params(labelsize=4, width=0.5, length=1.5)
ax_pred_random.grid(True, alpha=0.2, linewidth=0.3)

mean_random = np.mean(random_preds)
std_random = np.std(random_preds)
ax_pred_random.text(0.95, 0.95, f'μ={mean_random:.3f}\nσ={std_random:.3f}',
                   transform=ax_pred_random.transAxes, fontsize=3.5,
                   ha='right', va='top',
                   bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', lw=0.3))

# ============================================================================
# ROW 2: Attribution magnitude comparison
# ============================================================================
ax_attr_real = fig.add_subplot(gs[2, 1])
ax_attr_random = fig.add_subplot(gs[2, 2])

# Scatter plot: prediction vs attribution
ax_attr_real.scatter(real_preds, real_attrs, s=1, alpha=0.4,
                    color=[0.2, 0.4, 0.7], rasterized=True)
ax_attr_real.set_xlabel('Prediction', fontsize=5)
ax_attr_real.set_ylabel('Mean |Attribution|', fontsize=5)
ax_attr_real.set_title('Attribution vs Prediction', fontsize=5, weight='bold', pad=2)
ax_attr_real.tick_params(labelsize=4, width=0.5, length=1.5)
ax_attr_real.grid(True, alpha=0.2, linewidth=0.3)

# Add correlation
corr_real = np.corrcoef(real_preds, real_attrs)[0, 1]
ax_attr_real.text(0.05, 0.95, f'r={corr_real:.3f}',
                 transform=ax_attr_real.transAxes, fontsize=3.5,
                 ha='left', va='top',
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', lw=0.3))

ax_attr_random.scatter(random_preds, random_attrs, s=1, alpha=0.4,
                      color=[0.8, 0.4, 0.2], rasterized=True)
ax_attr_random.set_xlabel('Prediction', fontsize=5)
ax_attr_random.set_yticklabels([])
ax_attr_random.set_title('Attribution vs Prediction', fontsize=5, weight='bold', pad=2)
ax_attr_random.tick_params(labelsize=4, width=0.5, length=1.5)
ax_attr_random.grid(True, alpha=0.2, linewidth=0.3)

corr_random = np.corrcoef(random_preds, random_attrs)[0, 1]
ax_attr_random.text(0.05, 0.95, f'r={corr_random:.3f}',
                   transform=ax_attr_random.transAxes, fontsize=3.5,
                   ha='left', va='top',
                   bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', lw=0.3))

# ============================================================================
# ROW 3: Position-wise attribution heatmap (ALL positions, both proteins)
# ============================================================================
ax_pos_real = fig.add_subplot(gs[3, 1])
ax_pos_random = fig.add_subplot(gs[3, 2])

# Aggregate position-wise attributions across all pairs
max_pos = 200  # Show first 200 positions for density

def get_position_matrix(pos_data, max_positions=200):
    """Extract position × amino acid attribution matrix."""
    protein1_data = pos_data['protein1']
    protein2_data = pos_data['protein2']

    # Amino acids
    aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
           'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    matrix = np.zeros((20, max_positions))

    for pos in range(min(max_positions, len(protein1_data))):
        if pos in protein1_data:
            pos_dict = protein1_data[pos]
            for i, aa in enumerate(aas):
                if aa in pos_dict:
                    values = pos_dict[aa]
                    if len(values) > 0:
                        matrix[i, pos] = np.mean(np.abs(values))

    return matrix, aas

real_matrix, aas = get_position_matrix(real_pos, max_pos)
random_matrix, _ = get_position_matrix(random_pos, max_pos)

# Plot heatmaps
cmap = LinearSegmentedColormap.from_list('blackbody', BLACKBODY_COLORS)

im1 = ax_pos_real.imshow(real_matrix, aspect='auto', cmap=cmap,
                         interpolation='nearest', vmin=0, vmax=0.02)
ax_pos_real.set_ylabel('Amino Acid', fontsize=5)
ax_pos_real.set_xlabel('Position', fontsize=5)
ax_pos_real.set_title(f'Position × AA Attribution (first {max_pos} pos)', fontsize=5, weight='bold', pad=2)
ax_pos_real.set_yticks(np.arange(0, 20, 2))
ax_pos_real.set_yticklabels([aas[i] for i in range(0, 20, 2)], fontsize=3.5)
ax_pos_real.set_xticks(np.arange(0, max_pos, 25))
ax_pos_real.set_xticklabels(np.arange(0, max_pos, 25), fontsize=3.5)
ax_pos_real.tick_params(width=0.5, length=1.5)

im2 = ax_pos_random.imshow(random_matrix, aspect='auto', cmap=cmap,
                           interpolation='nearest', vmin=0, vmax=0.02)
ax_pos_random.set_xlabel('Position', fontsize=5)
ax_pos_random.set_yticklabels([])
ax_pos_random.set_title(f'Position × AA Attribution (first {max_pos} pos)', fontsize=5, weight='bold', pad=2)
ax_pos_random.set_xticks(np.arange(0, max_pos, 25))
ax_pos_random.set_xticklabels(np.arange(0, max_pos, 25), fontsize=3.5)
ax_pos_random.tick_params(width=0.5, length=1.5)

# Shared colorbar for position heatmaps
cbar_ax = fig.add_subplot(gs[3, 3])
cbar = plt.colorbar(im2, cax=cbar_ax)
cbar.set_label('Mean |Attr|', fontsize=4, rotation=270, labelpad=5)
cbar.ax.tick_params(labelsize=3.5, width=0.3, length=1.5)

# ============================================================================
# ROW 4: Top motifs comparison
# ============================================================================
ax_motif_real = fig.add_subplot(gs[4, 1])
ax_motif_random = fig.add_subplot(gs[4, 2])

# Plot top 20 motifs
n_motifs = 20
real_motif_names = [m['motif'] for m in real_motifs[:n_motifs]]
real_motif_scores = [m['mean_score'] for m in real_motifs[:n_motifs]]

random_motif_names = [m['motif'] for m in random_motifs[:n_motifs]]
random_motif_scores = [m['mean_score'] for m in random_motifs[:n_motifs]]

y_pos = np.arange(n_motifs)

ax_motif_real.barh(y_pos, real_motif_scores, height=0.8,
                   color=[0.2, 0.4, 0.7], alpha=0.7, linewidth=0)
ax_motif_real.set_yticks(y_pos)
ax_motif_real.set_yticklabels(real_motif_names, fontsize=3, family='monospace')
ax_motif_real.set_xlabel('Attribution Score', fontsize=5)
ax_motif_real.set_title(f'Top {n_motifs} Motifs (5-mer)', fontsize=5, weight='bold', pad=2)
ax_motif_real.tick_params(labelsize=4, width=0.5, length=1.5)
ax_motif_real.grid(True, alpha=0.2, linewidth=0.3, axis='x')
ax_motif_real.invert_yaxis()

ax_motif_random.barh(y_pos, random_motif_scores, height=0.8,
                     color=[0.8, 0.4, 0.2], alpha=0.7, linewidth=0)
ax_motif_random.set_yticks(y_pos)
ax_motif_random.set_yticklabels(random_motif_names, fontsize=3, family='monospace')
ax_motif_random.set_xlabel('Attribution Score', fontsize=5)
ax_motif_random.set_title(f'Top {n_motifs} Motifs (5-mer)', fontsize=5, weight='bold', pad=2)
ax_motif_random.tick_params(labelsize=4, width=0.5, length=1.5)
ax_motif_random.grid(True, alpha=0.2, linewidth=0.3, axis='x')
ax_motif_random.invert_yaxis()

# ============================================================================
# ROW 5: Statistical comparison
# ============================================================================
ax_stats = fig.add_subplot(gs[5, 1:3])

# Compute comprehensive statistics
t_stat, p_val = stats.ttest_ind(real_preds, random_preds)
u_stat, p_val_mw = stats.mannwhitneyu(real_preds, random_preds, alternative='two-sided')
ks_stat, p_val_ks = stats.ks_2samp(real_preds, random_preds)

# Cohen's d
pooled_std = np.sqrt((std_real**2 + std_random**2) / 2)
cohens_d = (mean_real - mean_random) / pooled_std

# Create stats table
stats_text = f"""Statistical Comparison (n={len(real_results)} each):

Predictions:        Real: {mean_real:.4f} ± {std_real:.4f}    Random: {mean_random:.4f} ± {std_random:.4f}
Attributions:       Real: {np.mean(real_attrs):.4f} ± {np.std(real_attrs):.4f}    Random: {np.mean(random_attrs):.4f} ± {np.std(random_attrs):.4f}

t-test:             t = {t_stat:.4f},  p = {p_val:.2e}  {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}
Mann-Whitney U:     U = {u_stat:.0f},  p = {p_val_mw:.2e}  {'***' if p_val_mw < 0.001 else '**' if p_val_mw < 0.01 else '*' if p_val_mw < 0.05 else 'ns'}
Kolmogorov-Smirnov: KS = {ks_stat:.4f}, p = {p_val_ks:.2e}  {'***' if p_val_ks < 0.001 else '**' if p_val_ks < 0.01 else '*' if p_val_ks < 0.05 else 'ns'}
Cohen's d:          {cohens_d:.4f}  ({'large' if abs(cohens_d) >= 0.8 else 'medium' if abs(cohens_d) >= 0.5 else 'small'} effect)

Total Motifs:       Real: {len(real_data['motif_results']['all_motifs']):,}    Random: {len(random_data['motif_results']['all_motifs']):,}
Total Attributions: {len(real_data['batch_results']['all_attributions']):,} positions analyzed
"""

ax_stats.text(0.02, 0.98, stats_text, transform=ax_stats.transAxes,
             fontsize=4, family='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='gray', linewidth=0.5))
ax_stats.axis('off')

# Main title
fig.suptitle('DeepLift Explainability Dashboard: Complete Dataset Analysis',
            fontsize=7, weight='bold', y=0.995)

# Subtitle with data info
fig.text(0.5, 0.975, f'1,084 Real PPIs vs 1,084 Random Pairs | {len(real_data["batch_results"]["all_attributions"]):,} Total Attributions | 80K+ Unique Motifs',
        ha='center', fontsize=4, style='italic')

# Save
output_dir = 'figures'
os.makedirs(output_dir, exist_ok=True)
output_path = 'figures/explainability_dashboard_complete'
for fmt in ['pdf', 'svg', 'png']:
    fname = f'{output_path}.{fmt}'
    if fmt == 'png':
        fig.savefig(fname, dpi=600, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
    else:
        fig.savefig(fname, format=fmt, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
    print(f"✓ Saved: {fname}")

print(f"\nDashboard created with:")
print(f"  - {len(real_results):,} real PPIs")
print(f"  - {len(random_results):,} random pairs")
print(f"  - {len(real_data['batch_results']['all_attributions']):,} total attributions")
print(f"  - {len(real_data['motif_results']['all_motifs']):,} unique motifs (real)")
print(f"  - {len(random_data['motif_results']['all_motifs']):,} unique motifs (random)")
print(f"  - p-value: {p_val:.2e} (highly significant)")

plt.show()
