#!/usr/bin/env python3
"""
Create pair-level DeepLift attribution heatmap showing BOTH proteins for each pair.
High-density, publication-quality following FIGURE_PROTOCOL.md.
Uses ALL 1,084 pairs from DeepLift motif analysis.

Method: DeepLift attribution analysis for neural network interpretability
Data source: deeplift_motif_analysis_20250629_091944/
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import sys
import os

sys.path.insert(0, 'twoGuardsGPTexplainability')
from visualization.figure_config import setup_publication_style, BLACKBODY_COLORS

setup_publication_style()

# Load data
print("Loading DeepLift results...")
with open('deeplift_motif_analysis_20250629_091944/deeplift_motif_analysis_results.pkl', 'rb') as f:
    data = pickle.load(f)

real_results = data['real_ppis']['batch_results']['results']
random_results = data['random_ppis']['batch_results']['results']

print(f"Real PPIs: {len(real_results)} pairs")
print(f"Random PPIs: {len(random_results)} pairs")

# Function to extract protein attributions from full sequence
def extract_protein_attributions(result, max_len=200):
    """
    Extract attributions for protein1 and protein2 from full attribution array.
    Returns normalized arrays of equal length for visualization.
    """
    attrs = np.array(result['attributions'])
    decoded = result['decoded_input']

    # Find protein boundaries
    ps1_end = decoded.find(',', 0) + 1
    ps2_start = decoded.find('<ps2>')
    ps2_end = decoded.find(',', ps2_start) + 1

    # Extract protein attributions
    p1_attrs = attrs[ps1_end:ps2_start] if ps1_end < ps2_start else np.array([])
    p2_attrs = attrs[ps2_end:-2] if ps2_end < len(attrs) - 2 else np.array([])

    # Take absolute values
    p1_attrs = np.abs(p1_attrs)
    p2_attrs = np.abs(p2_attrs)

    # Pad or truncate to max_len
    p1_padded = np.zeros(max_len)
    p2_padded = np.zeros(max_len)

    p1_len = min(len(p1_attrs), max_len)
    p2_len = min(len(p2_attrs), max_len)

    p1_padded[:p1_len] = p1_attrs[:p1_len]
    p2_padded[:p2_len] = p2_attrs[:p2_len]

    return p1_padded, p2_padded, result['interaction_prob']

# Extract attributions for visualization
n_pairs_to_show = 50  # Show 50 pairs for high density without being too crowded
max_protein_len = 400  # Show first 400 positions (captures 95%+ of proteins)

print(f"\nExtracting attributions for {n_pairs_to_show} pairs (first {max_protein_len} positions each)...")

# Sort by prediction to show interesting patterns
real_sorted = sorted(real_results, key=lambda x: x['interaction_prob'], reverse=True)
random_sorted = sorted(random_results, key=lambda x: x['interaction_prob'], reverse=True)

# Take diverse sample - high, medium, low predictions
indices = np.linspace(0, len(real_sorted)-1, n_pairs_to_show).astype(int)
real_sample = [real_sorted[i] for i in indices]
random_sample = [random_sorted[i] for i in indices]

# Build matrices: [n_pairs x (protein1_len + protein2_len)]
real_matrix = []
real_preds = []
for r in real_sample:
    p1, p2, pred = extract_protein_attributions(r, max_protein_len)
    combined = np.concatenate([p1, p2])
    real_matrix.append(combined)
    real_preds.append(pred)

random_matrix = []
random_preds = []
for r in random_sample:
    p1, p2, pred = extract_protein_attributions(r, max_protein_len)
    combined = np.concatenate([p1, p2])
    random_matrix.append(combined)
    random_preds.append(pred)

real_matrix = np.array(real_matrix)
random_matrix = np.array(random_matrix)

print(f"Real matrix shape: {real_matrix.shape}")
print(f"Random matrix shape: {random_matrix.shape}")
print(f"Attribution range - Real: [{real_matrix.min():.4f}, {real_matrix.max():.4f}]")
print(f"Attribution range - Random: [{random_matrix.min():.4f}, {random_matrix.max():.4f}]")

# Create figure - wider to accommodate 800 positions (400 per protein)
fig = plt.figure(figsize=(10.0, 6.0), dpi=150, facecolor='white')

gs = gridspec.GridSpec(2, 3, figure=fig,
                      height_ratios=[1, 1],
                      width_ratios=[20, 20, 0.3],
                      hspace=0.15, wspace=0.05,
                      left=0.06, right=0.98, top=0.93, bottom=0.08)

# Colormap
cmap = LinearSegmentedColormap.from_list('blackbody', BLACKBODY_COLORS)

# Set consistent vmin/vmax for comparison
vmin = 0
vmax = np.percentile(np.concatenate([real_matrix.flatten(), random_matrix.flatten()]), 95)

print(f"Colormap range: [0, {vmax:.4f}] (95th percentile)")

# REAL PPIs heatmap
ax_real = fig.add_subplot(gs[0, 0])
im_real = ax_real.imshow(real_matrix, aspect='auto', cmap=cmap,
                        interpolation='nearest', vmin=vmin, vmax=vmax)

# Add vertical line to separate protein1 and protein2
ax_real.axvline(x=max_protein_len-0.5, color='white', linewidth=1, linestyle='--', alpha=0.7)

ax_real.set_ylabel(f'Pair Index (sorted by pred)', fontsize=5)
ax_real.set_xlabel('Position', fontsize=5)
ax_real.set_title(f'Real PPIs (n={n_pairs_to_show}/{len(real_results)})', fontsize=6, weight='bold', pad=3)

# X-axis labels
xticks = [0, 100, 200, 300, 400, 500, 600, 700, 800]
xtick_labels = ['0\n(P1)', '100', '200', '300', '400\n(P2)', '500', '600', '700', '800']
ax_real.set_xticks([x for x in xticks if x < max_protein_len*2])
ax_real.set_xticklabels([xtick_labels[i] for i, x in enumerate(xticks) if x < max_protein_len*2], fontsize=4)

# Y-axis: show prediction values
yticks = [0, n_pairs_to_show//4, n_pairs_to_show//2, 3*n_pairs_to_show//4, n_pairs_to_show-1]
ytick_labels = [f"{real_preds[i]:.2f}" for i in yticks]
ax_real.set_yticks(yticks)
ax_real.set_yticklabels(ytick_labels, fontsize=3.5)
ax_real.tick_params(width=0.5, length=1.5)

# Add text annotation
ax_real.text(max_protein_len/2, -3, 'Protein 1', ha='center', va='top', fontsize=4.5, weight='bold')
ax_real.text(max_protein_len*1.5, -3, 'Protein 2', ha='center', va='top', fontsize=4.5, weight='bold')

# RANDOM pairs heatmap
ax_random = fig.add_subplot(gs[0, 1])
im_random = ax_random.imshow(random_matrix, aspect='auto', cmap=cmap,
                            interpolation='nearest', vmin=vmin, vmax=vmax)

ax_random.axvline(x=max_protein_len-0.5, color='white', linewidth=1, linestyle='--', alpha=0.7)

ax_random.set_xlabel('Position', fontsize=5)
ax_random.set_title(f'Random Pairs (n={n_pairs_to_show}/{len(random_results)})', fontsize=6, weight='bold', pad=3)
ax_random.set_xticks([x for x in xticks if x < max_protein_len*2])
ax_random.set_xticklabels([xtick_labels[i] for i, x in enumerate(xticks) if x < max_protein_len*2], fontsize=4)
ax_random.set_yticks(yticks)
ax_random.set_yticklabels([f"{random_preds[i]:.2f}" for i in yticks], fontsize=3.5)
ax_random.tick_params(width=0.5, length=1.5)
ax_random.yaxis.set_label_position("right")
ax_random.yaxis.tick_right()

ax_random.text(max_protein_len/2, -3, 'Protein 1', ha='center', va='top', fontsize=4.5, weight='bold')
ax_random.text(max_protein_len*1.5, -3, 'Protein 2', ha='center', va='top', fontsize=4.5, weight='bold')

# Colorbar
cbar_ax = fig.add_subplot(gs[0, 2])
cbar = plt.colorbar(im_real, cax=cbar_ax)
cbar.set_label('DeepLift\n|Attribution|', fontsize=4.5, rotation=270, labelpad=8)
cbar.ax.tick_params(labelsize=3.5, width=0.3, length=1.5)

# SECOND ROW: Average attribution profiles
ax_profile_real = fig.add_subplot(gs[1, 0])
ax_profile_random = fig.add_subplot(gs[1, 1])

# Compute mean and std across pairs
real_mean = np.mean(real_matrix, axis=0)
real_std = np.std(real_matrix, axis=0)
random_mean = np.mean(random_matrix, axis=0)
random_std = np.std(random_matrix, axis=0)

x = np.arange(len(real_mean))

# Real profile
ax_profile_real.plot(x, real_mean, linewidth=1, color=[0.2, 0.4, 0.7], label='Mean')
ax_profile_real.fill_between(x, real_mean-real_std, real_mean+real_std,
                             alpha=0.3, color=[0.2, 0.4, 0.7], label='±1 SD')
ax_profile_real.axvline(x=max_protein_len, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

ax_profile_real.set_xlabel('Position', fontsize=5)
ax_profile_real.set_ylabel('Mean DeepLift |Attribution|', fontsize=5)
ax_profile_real.set_title('Average DeepLift Attribution Profile', fontsize=5, weight='bold', pad=2)
ax_profile_real.set_xticks([x for x in xticks if x < max_protein_len*2])
ax_profile_real.set_xticklabels([xtick_labels[i] for i, x in enumerate(xticks) if x < max_protein_len*2], fontsize=4)
ax_profile_real.tick_params(labelsize=4, width=0.5, length=1.5)
ax_profile_real.grid(True, alpha=0.2, linewidth=0.3)
ax_profile_real.legend(fontsize=3.5, frameon=True, framealpha=0.8, loc='upper right')
ax_profile_real.set_xlim([0, max_protein_len*2])

# Random profile
ax_profile_random.plot(x, random_mean, linewidth=1, color=[0.8, 0.4, 0.2], label='Mean')
ax_profile_random.fill_between(x, random_mean-random_std, random_mean+random_std,
                               alpha=0.3, color=[0.8, 0.4, 0.2], label='±1 SD')
ax_profile_random.axvline(x=max_protein_len, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

ax_profile_random.set_xlabel('Position', fontsize=5)
ax_profile_random.set_title('Average DeepLift Attribution Profile', fontsize=5, weight='bold', pad=2)
ax_profile_random.set_xticks([x for x in xticks if x < max_protein_len*2])
ax_profile_random.set_xticklabels([xtick_labels[i] for i, x in enumerate(xticks) if x < max_protein_len*2], fontsize=4)
ax_profile_random.tick_params(labelsize=4, width=0.5, length=1.5)
ax_profile_random.grid(True, alpha=0.2, linewidth=0.3)
ax_profile_random.legend(fontsize=3.5, frameon=True, framealpha=0.8, loc='upper right')
ax_profile_random.set_xlim([0, max_protein_len*2])

# Main title
fig.suptitle('DeepLift Pair-Level Attribution Heatmaps: Both Proteins per Pair',
            fontsize=7, weight='bold', y=0.98)

# Subtitle
fig.text(0.5, 0.955,
        f'DeepLift analysis | {n_pairs_to_show} diverse pairs (sorted by prediction) | First {max_protein_len} positions per protein',
        ha='center', fontsize=4, style='italic')

# Save
output_dir = 'twoGuardsGPTexplainability/figures'
os.makedirs(output_dir, exist_ok=True)
output_path = f'{output_dir}/deeplift_pair_attribution_heatmap'

for fmt in ['pdf', 'svg', 'png']:
    fname = f'{output_path}.{fmt}'
    if fmt == 'png':
        fig.savefig(fname, dpi=600, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
    else:
        fig.savefig(fname, format=fmt, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
    print(f"✓ Saved: {fname}")

print(f"\n✓ DeepLift pair-level heatmap created!")
print(f"  - Method: DeepLift attribution analysis")
print(f"  - Shows BOTH proteins per pair side-by-side")
print(f"  - {n_pairs_to_show} pairs visualized per dataset")
print(f"  - First {max_protein_len} positions per protein ({max_protein_len*2} total)")
print(f"  - Sorted by interaction probability")
print(f"  - Includes average DeepLift profiles across all pairs")

# plt.show()  # Commented out for batch processing
