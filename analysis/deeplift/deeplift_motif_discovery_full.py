#!/usr/bin/env python3
"""
DeepLift Motif Discovery - Full Analysis with Detailed Logging
Analyzes all sequences to identify important amino acid motifs and patterns
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from scipy import stats
import logomaker
from matplotlib.patches import Rectangle
import os
import logging
from datetime import datetime
import time

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('seaborn-darkgrid')
sns.set_palette("husl")

# Global logger instances
logger = None
master_logger = None

def setup_logging():
    """Set up logging configuration."""
    global logger, master_logger
    
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Set up run-specific logger
    log_filename = f'logs/deeplift_motif_analysis_log_{timestamp}.txt'
    logger = logging.getLogger('deeplift_analysis')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers
    
    # File handler for this run
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    # Set up master logger that tracks all runs
    master_log_filename = 'logs/deeplift_master_log.txt'
    master_logger = logging.getLogger('deeplift_master')
    master_logger.setLevel(logging.INFO)
    master_logger.handlers = []  # Clear any existing handlers
    
    # File handler for master log
    mfh = logging.FileHandler(master_log_filename, mode='a')  # Append mode
    mfh.setLevel(logging.INFO)
    mfh.setFormatter(formatter)
    master_logger.addHandler(mfh)
    
    # Log header
    logger.info("="*60)
    logger.info("DeepLift Motif Discovery Analysis Log")
    logger.info("="*60)
    logger.info(f"Script: deeplift_motif_discovery_full.py")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("Analysis Type: Full dataset motif discovery")
    logger.info("")
    
    # Also log to master
    master_logger.info("\n" + "="*60)
    master_logger.info(f"New Analysis Run: {timestamp}")
    master_logger.info(f"Script: deeplift_motif_discovery_full.py")
    master_logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return timestamp

def log_figure_creation(filename, description, dataset_name=None):
    """Log when a figure is created."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if dataset_name:
        msg = f"[{timestamp}] {filename} - {description} for {dataset_name}"
    else:
        msg = f"[{timestamp}] {filename} - {description}"
    logger.info("FIGURE CREATED: " + msg)
    master_logger.info("FIGURE: " + msg)

def load_deeplift_results(filepath):
    """Load DeepLift results from pickle file."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def extract_amino_acid_attributions(results):
    """Extract attributions mapped to amino acids."""
    all_attributions = []
    
    for result in results:
        if not result['success']:
            continue
            
        sequence = result['sequence']
        attributions = result['attributions']
        
        # Parse the sequence format: <ps1>,PROTEIN1,<ps2>,PROTEIN2,<
        parts = sequence.split(',')
        if len(parts) >= 4:
            protein1 = parts[1]
            protein2 = parts[3].rstrip('<')
            
            # Map attributions to amino acids
            # Skip special tokens: <ps1>, comma, <ps2>, comma, final <
            idx = 0
            aa_attributions = []
            
            # Skip <ps1>,
            idx += len('<ps1>,')
            
            # Protein 1
            for i, aa in enumerate(protein1):
                if idx < len(attributions):
                    aa_attributions.append({
                        'position': i,
                        'amino_acid': aa,
                        'attribution': attributions[idx],
                        'protein': 'protein1',
                        'relative_position': i / len(protein1)
                    })
                idx += 1
            
            # Skip ,<ps2>,
            idx += len(',<ps2>,')
            
            # Protein 2
            for i, aa in enumerate(protein2):
                if idx < len(attributions):
                    aa_attributions.append({
                        'position': i,
                        'amino_acid': aa,
                        'attribution': attributions[idx],
                        'protein': 'protein2',
                        'relative_position': i / len(protein2)
                    })
                idx += 1
            
            all_attributions.append({
                'sequence_id': result['pair_idx'],
                'prediction': result['prediction'],
                'protein1_len': len(protein1),
                'protein2_len': len(protein2),
                'aa_attributions': aa_attributions,
                'protein1_seq': protein1,
                'protein2_seq': protein2
            })
    
    return all_attributions

def analyze_position_importance(attributions_data, dataset_name):
    """Analyze position-specific importance across all sequences."""
    # Collect attributions by relative position
    position_attributions = defaultdict(list)
    
    for seq_data in attributions_data:
        for aa_attr in seq_data['aa_attributions']:
            # Use bins for relative positions
            rel_pos_bin = int(aa_attr['relative_position'] * 20) / 20  # 20 bins
            position_attributions[rel_pos_bin].append(abs(aa_attr['attribution']))
    
    # Calculate statistics
    positions = sorted(position_attributions.keys())
    means = [np.mean(position_attributions[p]) for p in positions]
    stds = [np.std(position_attributions[p]) for p in positions]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(positions, means, 'b-', linewidth=2, label='Mean |Attribution|')
    ax.fill_between(positions, 
                    np.array(means) - np.array(stds), 
                    np.array(means) + np.array(stds), 
                    alpha=0.3, color='blue', label='±1 STD')
    
    ax.set_xlabel('Relative Position in Protein', fontsize=12)
    ax.set_ylabel('Mean Absolute Attribution', fontsize=12)
    ax.set_title(f'Position-Specific Attribution Importance - {dataset_name}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'{dataset_name}_position_importance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.svg'
    plt.savefig(filename, format='svg', bbox_inches='tight', dpi=300)
    plt.close()
    
    log_figure_creation(filename, "Position-specific attribution importance analysis", dataset_name)
    
    return positions, means

def analyze_amino_acid_preferences(attributions_data, dataset_name):
    """Analyze which amino acids have highest attributions."""
    # Collect attributions by amino acid
    aa_attributions = defaultdict(list)
    
    for seq_data in attributions_data:
        for aa_attr in seq_data['aa_attributions']:
            aa_attributions[aa_attr['amino_acid']].append(abs(aa_attr['attribution']))
    
    # Calculate statistics
    aa_stats = {}
    for aa, attrs in aa_attributions.items():
        aa_stats[aa] = {
            'mean': np.mean(attrs),
            'std': np.std(attrs),
            'count': len(attrs),
            'max': np.max(attrs),
            'percentile_95': np.percentile(attrs, 95)
        }
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame(aa_stats).T
    df = df.sort_values('mean', ascending=False)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Mean attributions
    ax1.bar(range(len(df)), df['mean'], yerr=df['std'], capsize=5, color='steelblue')
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels(df.index, fontsize=10)
    ax1.set_ylabel('Mean Absolute Attribution', fontsize=12)
    ax1.set_title(f'Amino Acid Attribution Importance - {dataset_name}', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 95th percentile attributions
    ax2.bar(range(len(df)), df['percentile_95'], color='darkred')
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels(df.index, fontsize=10)
    ax2.set_ylabel('95th Percentile Attribution', fontsize=12)
    ax2.set_xlabel('Amino Acid', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filename = f'{dataset_name}_amino_acid_preferences_{datetime.now().strftime("%Y%m%d_%H%M%S")}.svg'
    plt.savefig(filename, format='svg', bbox_inches='tight', dpi=300)
    plt.close()
    
    log_figure_creation(filename, "Amino acid attribution preferences (mean and 95th percentile)", dataset_name)
    
    return df

def find_high_attribution_regions(attributions_data, threshold_percentile=90):
    """Find regions with high attributions."""
    high_attr_regions = []
    
    for seq_data in attributions_data:
        attrs = [aa['attribution'] for aa in seq_data['aa_attributions']]
        threshold = np.percentile(np.abs(attrs), threshold_percentile)
        
        # Find consecutive high attribution regions
        current_region = []
        for i, aa_attr in enumerate(seq_data['aa_attributions']):
            if abs(aa_attr['attribution']) >= threshold:
                current_region.append(aa_attr)
            else:
                if len(current_region) >= 3:  # Minimum region length
                    # Extract sequence
                    start_pos = current_region[0]['position']
                    end_pos = current_region[-1]['position']
                    protein = current_region[0]['protein']
                    
                    if protein == 'protein1':
                        seq = seq_data['protein1_seq'][start_pos:end_pos+1]
                    else:
                        seq = seq_data['protein2_seq'][start_pos:end_pos+1]
                    
                    high_attr_regions.append({
                        'sequence': seq,
                        'protein': protein,
                        'start': start_pos,
                        'end': end_pos,
                        'mean_attribution': np.mean([abs(aa['attribution']) for aa in current_region]),
                        'sequence_id': seq_data['sequence_id']
                    })
                current_region = []
        
        # Check last region
        if len(current_region) >= 3:
            start_pos = current_region[0]['position']
            end_pos = current_region[-1]['position']
            protein = current_region[0]['protein']
            
            if protein == 'protein1':
                seq = seq_data['protein1_seq'][start_pos:end_pos+1]
            else:
                seq = seq_data['protein2_seq'][start_pos:end_pos+1]
            
            high_attr_regions.append({
                'sequence': seq,
                'protein': protein,
                'start': start_pos,
                'end': end_pos,
                'mean_attribution': np.mean([abs(aa['attribution']) for aa in current_region]),
                'sequence_id': seq_data['sequence_id']
            })
    
    return high_attr_regions

def create_sequence_logo(sequences, dataset_name, title_suffix=""):
    """Create sequence logo from aligned sequences."""
    if not sequences:
        return
    
    # Find the most common length
    lengths = [len(seq) for seq in sequences]
    most_common_length = Counter(lengths).most_common(1)[0][0]
    
    # Filter sequences of the most common length
    filtered_seqs = [seq for seq in sequences if len(seq) == most_common_length]
    
    if len(filtered_seqs) < 5:  # Need at least 5 sequences
        return
    
    # Create position frequency matrix
    matrix = np.zeros((most_common_length, 20))
    aa_order = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_idx = {aa: i for i, aa in enumerate(aa_order)}
    
    for seq in filtered_seqs:
        for pos, aa in enumerate(seq):
            if aa in aa_to_idx:
                matrix[pos, aa_to_idx[aa]] += 1
    
    # Normalize to frequencies
    matrix = matrix / len(filtered_seqs)
    
    # Create DataFrame for logomaker
    df = pd.DataFrame(matrix, columns=list(aa_order))
    
    # Create logo
    fig, ax = plt.subplots(figsize=(12, 4))
    logo = logomaker.Logo(df, ax=ax, color_scheme='chemistry')
    
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Sequence Logo - {dataset_name}{title_suffix}', fontsize=14)
    
    plt.tight_layout()
    filename = f'{dataset_name}_logo{title_suffix.replace(" ", "_")}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.svg'
    plt.savefig(filename, format='svg', bbox_inches='tight', dpi=300)
    plt.close()
    
    log_figure_creation(filename, f"Sequence logo{title_suffix}", dataset_name)
    logger.info(f"  - Created from {len(filtered_seqs)} sequences of length {most_common_length}")

def analyze_motif_patterns(high_attr_regions, dataset_name):
    """Analyze common motifs in high attribution regions."""
    # Extract all k-mers (3-5 length)
    kmer_counts = defaultdict(lambda: {'count': 0, 'mean_attr': []})
    
    for region in high_attr_regions:
        seq = region['sequence']
        for k in range(3, 6):  # 3-mers to 5-mers
            for i in range(len(seq) - k + 1):
                kmer = seq[i:i+k]
                kmer_counts[kmer]['count'] += 1
                kmer_counts[kmer]['mean_attr'].append(region['mean_attribution'])
    
    # Calculate average attribution for each k-mer
    kmer_stats = []
    for kmer, data in kmer_counts.items():
        if data['count'] >= 5:  # Minimum occurrence
            kmer_stats.append({
                'motif': kmer,
                'count': data['count'],
                'mean_attribution': np.mean(data['mean_attr']),
                'std_attribution': np.std(data['mean_attr'])
            })
    
    # Sort by mean attribution
    kmer_stats.sort(key=lambda x: x['mean_attribution'], reverse=True)
    
    # Plot top motifs
    if kmer_stats:
        top_motifs = kmer_stats[:20]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        motifs = [m['motif'] for m in top_motifs]
        means = [m['mean_attribution'] for m in top_motifs]
        counts = [m['count'] for m in top_motifs]
        
        # Color by count
        colors = plt.cm.viridis(np.array(counts) / max(counts))
        
        bars = ax.barh(range(len(motifs)), means, color=colors)
        ax.set_yticks(range(len(motifs)))
        ax.set_yticklabels(motifs, fontsize=10)
        ax.set_xlabel('Mean Attribution Score', fontsize=12)
        ax.set_title(f'Top High-Attribution Motifs - {dataset_name}', fontsize=14)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                   norm=plt.Normalize(vmin=min(counts), vmax=max(counts)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Occurrence Count', fontsize=10)
        
        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'n={count}', va='center', fontsize=8)
        
        plt.tight_layout()
        filename = f'{dataset_name}_top_motifs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.svg'
        plt.savefig(filename, format='svg', bbox_inches='tight', dpi=300)
        plt.close()
        
        log_figure_creation(filename, f"Top {len(top_motifs)} high-attribution motifs", dataset_name)
        logger.info(f"  - Total unique motifs found: {len(kmer_stats)}")
        logger.info(f"  - Top 3 motifs: {', '.join([m['motif'] for m in top_motifs[:3]])}")
    
    return kmer_stats

def create_attribution_heatmap(attributions_data, dataset_name, num_sequences=50):
    """Create heatmap of attributions for top sequences."""
    # Sort by prediction score and take top sequences
    sorted_data = sorted(attributions_data, key=lambda x: x['prediction'], reverse=True)[:num_sequences]
    
    # Find max length for padding
    max_len = max(len(seq['aa_attributions']) for seq in sorted_data)
    
    # Create matrix
    matrix = np.zeros((num_sequences, max_len))
    aa_matrix = []
    
    for i, seq_data in enumerate(sorted_data):
        aa_seq = []
        for j, aa_attr in enumerate(seq_data['aa_attributions']):
            matrix[i, j] = aa_attr['attribution']
            aa_seq.append(aa_attr['amino_acid'])
        # Pad with zeros
        for j in range(len(seq_data['aa_attributions']), max_len):
            matrix[i, j] = 0
            aa_seq.append('')
        aa_matrix.append(aa_seq)
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # Use diverging colormap
    vmax = np.percentile(np.abs(matrix), 95)
    im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attribution Score', fontsize=12)
    
    # Labels
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Sequence (sorted by prediction)', fontsize=12)
    ax.set_title(f'Attribution Heatmap - Top {num_sequences} Sequences - {dataset_name}', fontsize=14)
    
    # Add prediction scores on y-axis
    y_labels = [f"Seq {sorted_data[i]['sequence_id']} (p={sorted_data[i]['prediction']:.3f})" 
                for i in range(num_sequences)]
    ax.set_yticks(range(num_sequences))
    ax.set_yticklabels(y_labels, fontsize=8)
    
    plt.tight_layout()
    filename = f'{dataset_name}_attribution_heatmap_{datetime.now().strftime("%Y%m%d_%H%M%S")}.svg'
    plt.savefig(filename, format='svg', bbox_inches='tight', dpi=300)
    plt.close()
    
    log_figure_creation(filename, f"Attribution heatmap for top {num_sequences} sequences", dataset_name)
    logger.info(f"  - Sequences sorted by prediction score")
    logger.info(f"  - Max sequence length: {max_len}")

def analyze_position_specific_aa_preferences(attributions_data, dataset_name):
    """Analyze amino acid preferences at high-attribution positions."""
    # Find positions with high attributions
    position_attrs = defaultdict(list)
    
    for seq_data in attributions_data:
        for aa_attr in seq_data['aa_attributions']:
            rel_pos_bin = int(aa_attr['relative_position'] * 10) / 10  # 10 bins
            position_attrs[rel_pos_bin].append((aa_attr['amino_acid'], abs(aa_attr['attribution'])))
    
    # For each position bin, find AA preferences
    high_attr_positions = {}
    for pos, aa_attrs in position_attrs.items():
        attrs = [attr for _, attr in aa_attrs]
        if np.mean(attrs) > np.percentile([np.mean([a for _, a in v]) for v in position_attrs.values()], 75):
            # This is a high attribution position
            aa_counts = defaultdict(lambda: {'count': 0, 'total_attr': 0})
            for aa, attr in aa_attrs:
                aa_counts[aa]['count'] += 1
                aa_counts[aa]['total_attr'] += attr
            
            # Calculate enrichment
            total_count = sum(d['count'] for d in aa_counts.values())
            aa_enrichment = {}
            for aa, data in aa_counts.items():
                aa_enrichment[aa] = {
                    'frequency': data['count'] / total_count,
                    'mean_attr': data['total_attr'] / data['count'] if data['count'] > 0 else 0
                }
            
            high_attr_positions[pos] = aa_enrichment
    
    # Visualize top positions
    if high_attr_positions:
        positions = sorted(high_attr_positions.keys())[:5]  # Top 5 positions
        
        fig, axes = plt.subplots(1, len(positions), figsize=(4*len(positions), 6))
        if len(positions) == 1:
            axes = [axes]
        
        for idx, (pos, ax) in enumerate(zip(positions, axes)):
            aa_data = high_attr_positions[pos]
            
            # Get top AAs by mean attribution
            sorted_aas = sorted(aa_data.items(), key=lambda x: x[1]['mean_attr'], reverse=True)[:10]
            
            aas = [aa for aa, _ in sorted_aas]
            mean_attrs = [data['mean_attr'] for _, data in sorted_aas]
            frequencies = [data['frequency'] for _, data in sorted_aas]
            
            # Create bar plot
            bars = ax.bar(range(len(aas)), mean_attrs, color='steelblue')
            
            # Color by frequency
            for bar, freq in zip(bars, frequencies):
                bar.set_alpha(0.3 + 0.7 * freq / max(frequencies))
            
            ax.set_xticks(range(len(aas)))
            ax.set_xticklabels(aas, fontsize=10)
            ax.set_ylabel('Mean Attribution', fontsize=10)
            ax.set_title(f'Position {pos:.1f}', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(f'AA Preferences at High-Attribution Positions - {dataset_name}', fontsize=14)
        plt.tight_layout()
        filename = f'{dataset_name}_position_aa_preferences_{datetime.now().strftime("%Y%m%d_%H%M%S")}.svg'
        plt.savefig(filename, format='svg', bbox_inches='tight', dpi=300)
        plt.close()
        
        log_figure_creation(filename, f"Amino acid preferences at {len(positions)} high-attribution positions", dataset_name)

def generate_summary_statistics(attributions_data, high_attr_regions, kmer_stats, dataset_name):
    """Generate summary statistics file."""
    filename = f'{dataset_name}_summary_stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    with open(filename, 'w') as f:
        f.write(f"DeepLift Attribution Analysis Summary - {dataset_name}\n")
        f.write("="*60 + "\n\n")
        
        # Basic statistics
        f.write("Dataset Statistics:\n")
        f.write(f"- Total sequences analyzed: {len(attributions_data)}\n")
        
        all_attrs = []
        for seq in attributions_data:
            all_attrs.extend([abs(aa['attribution']) for aa in seq['aa_attributions']])
        
        f.write(f"- Mean absolute attribution: {np.mean(all_attrs):.4f}\n")
        f.write(f"- Std absolute attribution: {np.std(all_attrs):.4f}\n")
        f.write(f"- Max absolute attribution: {np.max(all_attrs):.4f}\n")
        f.write(f"- 95th percentile attribution: {np.percentile(all_attrs, 95):.4f}\n")
        
        # High attribution regions
        f.write(f"\nHigh Attribution Regions:\n")
        f.write(f"- Total regions found: {len(high_attr_regions)}\n")
        if high_attr_regions:
            lengths = [r['end'] - r['start'] + 1 for r in high_attr_regions]
            f.write(f"- Average region length: {np.mean(lengths):.1f}\n")
            f.write(f"- Max region length: {max(lengths)}\n")
        
        # Top motifs
        f.write(f"\nTop 10 High-Attribution Motifs:\n")
        for i, motif in enumerate(kmer_stats[:10]):
            f.write(f"{i+1}. {motif['motif']} - count: {motif['count']}, "
                   f"mean attr: {motif['mean_attribution']:.4f}\n")
        
        # Amino acid statistics
        aa_attrs = defaultdict(list)
        for seq in attributions_data:
            for aa_attr in seq['aa_attributions']:
                aa_attrs[aa_attr['amino_acid']].append(abs(aa_attr['attribution']))
        
        f.write(f"\nTop 5 Amino Acids by Mean Attribution:\n")
        aa_means = [(aa, np.mean(attrs)) for aa, attrs in aa_attrs.items()]
        aa_means.sort(key=lambda x: x[1], reverse=True)
        
        for i, (aa, mean_attr) in enumerate(aa_means[:5]):
            f.write(f"{i+1}. {aa}: {mean_attr:.4f}\n")
    
    log_figure_creation(filename, "Summary statistics file", dataset_name)
    return filename

def main():
    """Main analysis pipeline."""
    # Set up logging
    start_time = time.time()
    run_timestamp = setup_logging()
    
    logger.info("Loading DeepLift results...")
    
    # Analyze both datasets
    datasets = [
        ('real_ppis', './full_deeplift_results/real_ppis_full_deeplift_detailed.pkl'),
        ('random_ppis', './full_deeplift_results/random_ppis_full_deeplift_detailed.pkl')
    ]
    
    total_sequences_analyzed = 0
    all_figures_created = []
    
    for dataset_name, filepath in datasets:
        logger.info(f"\nAnalyzing {dataset_name}...")
        dataset_start_time = time.time()
        
        # Load data
        results = load_deeplift_results(filepath)
        logger.info(f"Loaded {len(results)} sequences from {filepath}")
        
        # Extract amino acid attributions
        attributions_data = extract_amino_acid_attributions(results)
        logger.info(f"Successfully processed {len(attributions_data)} sequences")
        total_sequences_analyzed += len(attributions_data)
        
        # Position importance analysis
        logger.info("Analyzing position importance...")
        positions, means = analyze_position_importance(attributions_data, dataset_name)
        
        # Amino acid preferences
        logger.info("Analyzing amino acid preferences...")
        aa_df = analyze_amino_acid_preferences(attributions_data, dataset_name)
        logger.info(f"  - Top 3 AAs by mean attribution: {', '.join(aa_df.index[:3].tolist())}")
        
        # Find high attribution regions
        logger.info("Finding high attribution regions...")
        high_attr_regions = find_high_attribution_regions(attributions_data, threshold_percentile=90)
        logger.info(f"Found {len(high_attr_regions)} high attribution regions")
        
        # Create sequence logos
        if high_attr_regions:
            logger.info("Creating sequence logos...")
            # All high attribution sequences
            all_seqs = [r['sequence'] for r in high_attr_regions]
            create_sequence_logo(all_seqs, dataset_name, " - High Attribution Regions")
            
            # Separate by protein
            protein1_seqs = [r['sequence'] for r in high_attr_regions if r['protein'] == 'protein1']
            protein2_seqs = [r['sequence'] for r in high_attr_regions if r['protein'] == 'protein2']
            
            if protein1_seqs:
                create_sequence_logo(protein1_seqs, dataset_name, " - Protein 1")
            if protein2_seqs:
                create_sequence_logo(protein2_seqs, dataset_name, " - Protein 2")
        
        # Analyze motif patterns
        logger.info("Analyzing motif patterns...")
        kmer_stats = analyze_motif_patterns(high_attr_regions, dataset_name)
        if kmer_stats:
            logger.info(f"  - Found {len(kmer_stats)} significant motifs")
            top_motifs = [m['motif'] for m in kmer_stats[:5]]
            logger.info(f"  - Top 5 motifs: {', '.join(top_motifs)}")
        
        # Create attribution heatmap
        logger.info("Creating attribution heatmap...")
        create_attribution_heatmap(attributions_data, dataset_name)
        
        # Analyze position-specific AA preferences
        logger.info("Analyzing position-specific amino acid preferences...")
        analyze_position_specific_aa_preferences(attributions_data, dataset_name)
        
        # Generate summary statistics
        logger.info("Generating summary statistics...")
        stats_file = generate_summary_statistics(attributions_data, high_attr_regions, kmer_stats, dataset_name)
        
        dataset_time = time.time() - dataset_start_time
        logger.info(f"Completed analysis for {dataset_name} in {dataset_time:.1f} seconds")
    
    # Final summary
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS SUMMARY:")
    logger.info(f"- Total sequences analyzed: {total_sequences_analyzed}")
    
    # Count sequences for each dataset
    real_count = len(extract_amino_acid_attributions(load_deeplift_results(datasets[0][1])))
    random_count = len(extract_amino_acid_attributions(load_deeplift_results(datasets[1][1])))
    logger.info(f"- Real PPIs: {real_count} sequences")
    logger.info(f"- Random PPIs: {random_count} sequences")
    
    logger.info("")
    logger.info(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total Runtime: {minutes} minutes {seconds} seconds")
    logger.info("="*60)
    
    # Update master log
    master_logger.info(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    master_logger.info(f"Total Runtime: {minutes} minutes {seconds} seconds")
    master_logger.info(f"Total Sequences Analyzed: {total_sequences_analyzed}")
    master_logger.info("="*60)
    
    print(f"\nAnalysis complete! Check the generated plots and summary files.")
    print(f"Log file created: logs/deeplift_motif_analysis_log_{run_timestamp}.txt")

if __name__ == "__main__":
    main()