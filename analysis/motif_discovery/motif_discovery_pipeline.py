#!/usr/bin/env python3
"""
Motif Discovery Pipeline for MED4 PPI Analysis
Extracts common interaction patterns from high-confidence PPIs
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy import stats
from collections import defaultdict, Counter
import networkx as nx
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MotifDiscoveryPipeline:
    def __init__(self, analysis_dir: str, min_confidence: float = 0.8):
        self.analysis_dir = Path(analysis_dir)
        self.min_confidence = min_confidence
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.analysis_dir / f"motif_discovery_{self.timestamp}"
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("viridis")
        
        print(f"Motif Discovery Pipeline initialized")
        print(f"Output directory: {self.output_dir}")
        
    def load_high_confidence_pairs(self):
        """Load high confidence PPI pairs from analysis"""
        print("Loading high confidence pairs...")
        
        # Find all analysis results
        pkl_files = list(self.analysis_dir.glob("all_results.pkl"))
        if not pkl_files:
            raise FileNotFoundError("No analysis results found")
            
        with open(pkl_files[0], 'rb') as f:
            all_results = pickle.load(f)
            
        # Filter high confidence pairs
        self.high_conf_pairs = []
        for result in all_results:
            if result['baseline_prob'] >= self.min_confidence:
                self.high_conf_pairs.append({
                    'pair_id': result['pair_id'],
                    'protein1_seq': result['protein1_seq'],
                    'protein2_seq': result['protein2_seq'],
                    'probability': result['baseline_prob'],
                    'gradient_analysis': result.get('gradient_analysis', {}),
                    'perturbation_results': result.get('perturbation_results', {}),
                    'analysis': result.get('analysis', {})
                })
                
        print(f"Found {len(self.high_conf_pairs)} high confidence pairs (>= {self.min_confidence})")
        return self.high_conf_pairs
        
    def extract_critical_regions(self):
        """Extract critical regions from gradient and perturbation analyses"""
        print("Extracting critical regions...")
        
        self.critical_regions = []
        
        for pair in self.high_conf_pairs:
            # Extract from gradient analysis
            if 'gradient_analysis' in pair and 'top_residues' in pair['gradient_analysis']:
                grad_data = pair['gradient_analysis']['top_residues']
                
                # Get top residues for each protein
                p1_critical = []
                p2_critical = []
                
                if 'protein1_top' in grad_data:
                    for res in grad_data['protein1_top'][:5]:  # Top 5
                        p1_critical.append({
                            'position': res['position'],
                            'residue': res['residue'],
                            'importance': float(res['gradient_magnitude'])
                        })
                        
                if 'protein2_top' in grad_data:
                    for res in grad_data['protein2_top'][:5]:
                        p2_critical.append({
                            'position': res['position'],
                            'residue': res['residue'],
                            'importance': float(res['gradient_magnitude'])
                        })
                
                # Extract from perturbation analysis
                if 'analysis' in pair:
                    analysis = pair['analysis']
                    
                    # Get critical positions from perturbation
                    if 'protein1_analysis' in analysis:
                        p1_analysis = analysis['protein1_analysis']
                        if 'critical_residues' in p1_analysis:
                            for pos, data in p1_analysis['critical_residues'].items():
                                if isinstance(pos, str) and pos.isdigit():
                                    pos = int(pos)
                                p1_critical.append({
                                    'position': pos,
                                    'residue': data.get('original_aa', '?'),
                                    'importance': data.get('abs_max_effect', 0)
                                })
                    
                    if 'protein2_analysis' in analysis:
                        p2_analysis = analysis['protein2_analysis']
                        if 'critical_residues' in p2_analysis:
                            for pos, data in p2_analysis['critical_residues'].items():
                                if isinstance(pos, str) and pos.isdigit():
                                    pos = int(pos)
                                p2_critical.append({
                                    'position': pos,
                                    'residue': data.get('original_aa', '?'),
                                    'importance': data.get('abs_max_effect', 0)
                                })
                
                self.critical_regions.append({
                    'pair_id': pair['pair_id'],
                    'protein1_seq': pair['protein1_seq'],
                    'protein2_seq': pair['protein2_seq'],
                    'probability': pair['probability'],
                    'protein1_critical': p1_critical,
                    'protein2_critical': p2_critical
                })
                
        print(f"Extracted critical regions from {len(self.critical_regions)} pairs")
        return self.critical_regions
        
    def discover_sequence_motifs(self, window_size: int = 5):
        """Discover common sequence motifs around critical positions"""
        print(f"Discovering sequence motifs (window size: {window_size})...")
        
        # Extract sequences around critical positions
        motif_sequences = defaultdict(list)
        
        for region in self.critical_regions:
            # Process protein 1
            seq1 = region['protein1_seq']
            for crit in region['protein1_critical']:
                pos = crit['position']
                
                # Extract window around critical position
                start = max(0, pos - window_size)
                end = min(len(seq1), pos + window_size + 1)
                
                if end > start:
                    motif = seq1[start:end]
                    center_offset = pos - start
                    
                    motif_sequences['protein1'].append({
                        'motif': motif,
                        'critical_pos': center_offset,
                        'critical_residue': crit['residue'],
                        'importance': crit['importance'],
                        'pair_id': region['pair_id'],
                        'full_position': pos
                    })
            
            # Process protein 2
            seq2 = region['protein2_seq']
            for crit in region['protein2_critical']:
                pos = crit['position']
                
                start = max(0, pos - window_size)
                end = min(len(seq2), pos + window_size + 1)
                
                if end > start:
                    motif = seq2[start:end]
                    center_offset = pos - start
                    
                    motif_sequences['protein2'].append({
                        'motif': motif,
                        'critical_pos': center_offset,
                        'critical_residue': crit['residue'],
                        'importance': crit['importance'],
                        'pair_id': region['pair_id'],
                        'full_position': pos
                    })
        
        # Find common motifs
        self.motif_patterns = {}
        for protein in ['protein1', 'protein2']:
            if protein in motif_sequences:
                # Count motif occurrences
                motif_counter = Counter([m['motif'] for m in motif_sequences[protein]])
                
                # Get motifs appearing multiple times
                common_motifs = [(motif, count) for motif, count in motif_counter.items() 
                                if count > 1]
                common_motifs.sort(key=lambda x: x[1], reverse=True)
                
                self.motif_patterns[protein] = {
                    'all_motifs': motif_sequences[protein],
                    'common_motifs': common_motifs[:20],  # Top 20
                    'motif_counter': motif_counter
                }
                
                print(f"{protein}: Found {len(common_motifs)} recurring motifs")
                
        return self.motif_patterns
        
    def analyze_position_patterns(self):
        """Analyze positional patterns of critical residues"""
        print("Analyzing positional patterns...")
        
        position_stats = {'protein1': defaultdict(list), 'protein2': defaultdict(list)}
        
        for region in self.critical_regions:
            # Normalize positions by sequence length
            seq1_len = len(region['protein1_seq'])
            seq2_len = len(region['protein2_seq'])
            
            for crit in region['protein1_critical']:
                rel_pos = crit['position'] / seq1_len if seq1_len > 0 else 0
                position_stats['protein1']['relative_positions'].append(rel_pos)
                position_stats['protein1']['absolute_positions'].append(crit['position'])
                position_stats['protein1']['residues'].append(crit['residue'])
                
            for crit in region['protein2_critical']:
                rel_pos = crit['position'] / seq2_len if seq2_len > 0 else 0
                position_stats['protein2']['relative_positions'].append(rel_pos)
                position_stats['protein2']['absolute_positions'].append(crit['position'])
                position_stats['protein2']['residues'].append(crit['residue'])
                
        self.position_patterns = position_stats
        return position_stats
        
    def cluster_interaction_patterns(self):
        """Cluster PPIs based on their interaction patterns"""
        print("Clustering interaction patterns...")
        
        # Create feature vectors for each PPI
        features = []
        pair_ids = []
        
        for region in self.critical_regions:
            # Feature vector: critical residue types and positions
            feature_vec = np.zeros(40)  # 20 amino acids x 2 proteins
            
            # Count critical residue types
            for crit in region['protein1_critical']:
                aa_idx = ord(crit['residue']) - ord('A')
                if 0 <= aa_idx < 20:
                    feature_vec[aa_idx] += crit['importance']
                    
            for crit in region['protein2_critical']:
                aa_idx = ord(crit['residue']) - ord('A')
                if 0 <= aa_idx < 20:
                    feature_vec[20 + aa_idx] += crit['importance']
                    
            features.append(feature_vec)
            pair_ids.append(region['pair_id'])
            
        features = np.array(features)
        
        # Perform clustering
        if len(features) > 10:
            # Try DBSCAN clustering
            clustering = DBSCAN(eps=0.5, min_samples=3).fit(features)
            labels = clustering.labels_
            
            # Also try K-means
            n_clusters = min(10, len(features) // 10)
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(features)
                kmeans_labels = kmeans.labels_
            else:
                kmeans_labels = np.zeros(len(features))
                
            # PCA for visualization
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(features)
            
            self.clustering_results = {
                'features': features,
                'features_2d': features_2d,
                'dbscan_labels': labels,
                'kmeans_labels': kmeans_labels,
                'pair_ids': pair_ids,
                'pca': pca
            }
            
            print(f"DBSCAN found {len(set(labels)) - (1 if -1 in labels else 0)} clusters")
            print(f"K-means used {n_clusters} clusters")
            
        return self.clustering_results
        
    def extract_interaction_rules(self):
        """Extract rules for protein-protein interactions"""
        print("Extracting interaction rules...")
        
        rules = []
        
        # Rule 1: Critical residue co-occurrence
        residue_pairs = defaultdict(int)
        for region in self.critical_regions:
            if region['probability'] > 0.9:  # Very high confidence
                for crit1 in region['protein1_critical']:
                    for crit2 in region['protein2_critical']:
                        pair = (crit1['residue'], crit2['residue'])
                        residue_pairs[pair] += 1
                        
        # Find significant co-occurrences
        for pair, count in residue_pairs.items():
            if count >= 3:  # Appears in at least 3 high-conf PPIs
                rules.append({
                    'type': 'residue_co-occurrence',
                    'description': f"{pair[0]} in protein1 with {pair[1]} in protein2",
                    'count': count,
                    'confidence': count / len(self.critical_regions)
                })
                
        # Rule 2: Position-based patterns
        for protein in ['protein1', 'protein2']:
            if protein in self.position_patterns:
                rel_positions = self.position_patterns[protein]['relative_positions']
                if len(rel_positions) > 10:
                    # Find position hotspots
                    hist, bins = np.histogram(rel_positions, bins=10)
                    for i, count in enumerate(hist):
                        if count > len(rel_positions) * 0.1:  # >10% in this bin
                            rules.append({
                                'type': 'position_hotspot',
                                'description': f"{protein} critical residues at {bins[i]:.1f}-{bins[i+1]:.1f} relative position",
                                'count': count,
                                'confidence': count / len(rel_positions)
                            })
                            
        # Rule 3: Motif-based rules
        for protein in ['protein1', 'protein2']:
            if protein in self.motif_patterns:
                for motif, count in self.motif_patterns[protein]['common_motifs'][:5]:
                    if count >= 3:
                        rules.append({
                            'type': 'sequence_motif',
                            'description': f"{protein} contains motif '{motif}'",
                            'count': count,
                            'confidence': count / len(self.motif_patterns[protein]['all_motifs'])
                        })
                        
        self.interaction_rules = sorted(rules, key=lambda x: x['confidence'], reverse=True)
        print(f"Extracted {len(self.interaction_rules)} interaction rules")
        
        return self.interaction_rules
        
    def visualize_motifs(self):
        """Create comprehensive motif visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('MED4 PPI Motif Discovery Results', fontsize=20, fontweight='bold')
        
        # 1. Top motifs bar chart
        for idx, protein in enumerate(['protein1', 'protein2']):
            ax = axes[idx, 0]
            if protein in self.motif_patterns and self.motif_patterns[protein]['common_motifs']:
                motifs = self.motif_patterns[protein]['common_motifs'][:10]
                motif_names = [m[0] for m in motifs]
                motif_counts = [m[1] for m in motifs]
                
                bars = ax.barh(range(len(motifs)), motif_counts, color=plt.cm.viridis(np.linspace(0, 1, len(motifs))))
                ax.set_yticks(range(len(motifs)))
                ax.set_yticklabels(motif_names, fontfamily='monospace')
                ax.set_xlabel('Occurrences')
                ax.set_title(f'{protein.capitalize()}: Top Sequence Motifs')
                ax.grid(True, alpha=0.3)
                
                # Add count labels
                for i, (bar, count) in enumerate(zip(bars, motif_counts)):
                    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                           str(count), va='center')
        
        # 2. Position distribution
        for idx, protein in enumerate(['protein1', 'protein2']):
            ax = axes[idx, 1]
            if protein in self.position_patterns:
                rel_pos = self.position_patterns[protein]['relative_positions']
                if rel_pos:
                    ax.hist(rel_pos, bins=20, alpha=0.7, color='#440154', edgecolor='black')
                    ax.set_xlabel('Relative Position in Sequence')
                    ax.set_ylabel('Count')
                    ax.set_title(f'{protein.capitalize()}: Critical Position Distribution')
                    ax.grid(True, alpha=0.3)
                    
                    # Add statistics
                    mean_pos = np.mean(rel_pos)
                    ax.axvline(mean_pos, color='red', linestyle='--', linewidth=2, 
                              label=f'Mean: {mean_pos:.2f}')
                    ax.legend()
        
        # 3. Clustering visualization
        if hasattr(self, 'clustering_results'):
            ax = axes[0, 2]
            features_2d = self.clustering_results['features_2d']
            labels = self.clustering_results['kmeans_labels']
            
            scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], 
                               c=labels, cmap='viridis', alpha=0.6, s=50)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_title('PPI Clustering by Interaction Patterns')
            ax.grid(True, alpha=0.3)
            
            # Add cluster centers
            for i in range(len(set(labels))):
                cluster_points = features_2d[labels == i]
                if len(cluster_points) > 0:
                    center = cluster_points.mean(axis=0)
                    ax.scatter(center[0], center[1], marker='*', s=300, 
                             c='red', edgecolor='black', linewidth=2)
                    
        # 4. Critical residue heatmap
        ax = axes[1, 2]
        
        # Count critical residues by type
        residue_counts = defaultdict(lambda: {'protein1': 0, 'protein2': 0})
        for region in self.critical_regions:
            for crit in region['protein1_critical']:
                residue_counts[crit['residue']]['protein1'] += 1
            for crit in region['protein2_critical']:
                residue_counts[crit['residue']]['protein2'] += 1
                
        # Create matrix
        residues = sorted(residue_counts.keys())
        matrix = np.array([[residue_counts[r]['protein1'], residue_counts[r]['protein2']] 
                          for r in residues])
        
        im = ax.imshow(matrix.T, cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(residues)))
        ax.set_xticklabels(residues)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Protein 1', 'Protein 2'])
        ax.set_xlabel('Critical Residue')
        ax.set_title('Critical Residue Distribution')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Count')
        
        # Add values
        for i in range(len(residues)):
            for j in range(2):
                text = ax.text(i, j, str(int(matrix[i, j])), 
                             ha='center', va='center', color='white')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'motif_discovery_dashboard.svg', format='svg', bbox_inches='tight')
        plt.savefig(self.output_dir / 'motif_discovery_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def visualize_interaction_rules(self):
        """Visualize discovered interaction rules"""
        if not hasattr(self, 'interaction_rules'):
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Discovered MED4 PPI Interaction Rules', fontsize=18, fontweight='bold')
        
        # 1. Rule types distribution
        rule_types = defaultdict(int)
        for rule in self.interaction_rules:
            rule_types[rule['type']] += 1
            
        ax1.pie(rule_types.values(), labels=rule_types.keys(), autopct='%1.1f%%',
                colors=plt.cm.viridis(np.linspace(0, 1, len(rule_types))))
        ax1.set_title('Rule Type Distribution')
        
        # 2. Top rules by confidence
        ax2.axis('off')
        
        # Create table of top rules
        top_rules = self.interaction_rules[:15]
        table_data = []
        for i, rule in enumerate(top_rules):
            table_data.append([
                f"{i+1}",
                rule['type'].replace('_', ' ').title(),
                rule['description'][:50] + '...' if len(rule['description']) > 50 else rule['description'],
                f"{rule['confidence']:.3f}",
                str(rule['count'])
            ])
            
        table = ax2.table(cellText=table_data,
                         colLabels=['#', 'Type', 'Description', 'Confidence', 'Count'],
                         cellLoc='left',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(table_data) + 1):
            for j in range(5):
                cell = table[(i, j)]
                if i == 0:
                    cell.set_facecolor('#440154')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
                    
        ax2.set_title('Top Interaction Rules by Confidence', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'interaction_rules.svg', format='svg', bbox_inches='tight')
        plt.savefig(self.output_dir / 'interaction_rules.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def save_results(self):
        """Save all results to files"""
        print("Saving results...")
        
        # Save motif patterns
        if hasattr(self, 'motif_patterns'):
            motif_df_data = []
            for protein in ['protein1', 'protein2']:
                if protein in self.motif_patterns:
                    for motif, count in self.motif_patterns[protein]['common_motifs']:
                        motif_df_data.append({
                            'protein': protein,
                            'motif': motif,
                            'count': count,
                            'frequency': count / len(self.motif_patterns[protein]['all_motifs'])
                        })
            
            motif_df = pd.DataFrame(motif_df_data)
            motif_df.to_csv(self.output_dir / 'discovered_motifs.csv', index=False)
        
        # Save interaction rules
        if hasattr(self, 'interaction_rules'):
            rules_df = pd.DataFrame(self.interaction_rules)
            rules_df.to_csv(self.output_dir / 'interaction_rules.csv', index=False)
            
        # Save critical regions summary
        critical_summary = []
        for region in self.critical_regions:
            summary = {
                'pair_id': region['pair_id'],
                'probability': region['probability'],
                'n_critical_p1': len(region['protein1_critical']),
                'n_critical_p2': len(region['protein2_critical']),
                'top_critical_p1': region['protein1_critical'][0]['residue'] if region['protein1_critical'] else 'None',
                'top_critical_p2': region['protein2_critical'][0]['residue'] if region['protein2_critical'] else 'None'
            }
            critical_summary.append(summary)
            
        critical_df = pd.DataFrame(critical_summary)
        critical_df.to_csv(self.output_dir / 'critical_regions_summary.csv', index=False)
        
        # Save detailed results as pickle
        results = {
            'motif_patterns': self.motif_patterns if hasattr(self, 'motif_patterns') else None,
            'position_patterns': self.position_patterns if hasattr(self, 'position_patterns') else None,
            'clustering_results': self.clustering_results if hasattr(self, 'clustering_results') else None,
            'interaction_rules': self.interaction_rules if hasattr(self, 'interaction_rules') else None,
            'critical_regions': self.critical_regions
        }
        
        with open(self.output_dir / 'motif_discovery_results.pkl', 'wb') as f:
            pickle.dump(results, f)
            
    def run_pipeline(self):
        """Run the complete motif discovery pipeline"""
        print("\n" + "="*60)
        print("MOTIF DISCOVERY PIPELINE")
        print("="*60)
        
        # Load data
        self.load_high_confidence_pairs()
        
        # Extract critical regions
        self.extract_critical_regions()
        
        # Discover motifs
        self.discover_sequence_motifs()
        
        # Analyze patterns
        self.analyze_position_patterns()
        
        # Cluster interactions
        if len(self.critical_regions) > 10:
            self.cluster_interaction_patterns()
        
        # Extract rules
        self.extract_interaction_rules()
        
        # Create visualizations
        print("\nCreating visualizations...")
        self.visualize_motifs()
        self.visualize_interaction_rules()
        
        # Save results
        self.save_results()
        
        # Print summary
        print("\n" + "="*60)
        print("MOTIF DISCOVERY SUMMARY")
        print("="*60)
        print(f"High confidence pairs analyzed: {len(self.high_conf_pairs)}")
        print(f"Critical regions identified: {len(self.critical_regions)}")
        
        if hasattr(self, 'motif_patterns'):
            for protein in ['protein1', 'protein2']:
                if protein in self.motif_patterns and self.motif_patterns[protein]['common_motifs']:
                    top_motifs = self.motif_patterns[protein]['common_motifs'][:3]
                    print(f"\nTop {protein} motifs:")
                    for motif, count in top_motifs:
                        print(f"  '{motif}': {count} occurrences")
                        
        if hasattr(self, 'interaction_rules'):
            print(f"\nDiscovered {len(self.interaction_rules)} interaction rules")
            print("\nTop 5 rules by confidence:")
            for i, rule in enumerate(self.interaction_rules[:5]):
                print(f"  {i+1}. {rule['type']}: {rule['description']}")
                print(f"     Confidence: {rule['confidence']:.3f}, Count: {rule['count']}")
                
        print(f"\nResults saved to: {self.output_dir}")
        print("="*60)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        analysis_dir = sys.argv[1]
    else:
        # Default to real PPIs analysis
        analysis_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
        
    pipeline = MotifDiscoveryPipeline(analysis_dir, min_confidence=0.8)
    pipeline.run_pipeline()