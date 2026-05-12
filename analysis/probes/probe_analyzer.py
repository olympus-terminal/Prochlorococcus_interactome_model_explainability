"""
Probing Analysis Framework for PPI Model.
Probes internal representations to understand what biological properties the model has learned.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'ppiGPT_MED4_solo'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Optional, Any
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from model import GPTConfig, GPT


class ProbeAnalyzer:
    """Probing analysis to understand what biological properties the model learns."""
    
    def __init__(self, model, tokenizer_meta_path: str):
        """
        Initialize probe analyzer.
        
        Args:
            model: Trained GPT model
            tokenizer_meta_path: Path to tokenizer metadata
        """
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device
        
        # Load tokenizer
        with open(tokenizer_meta_path, 'rb') as f:
            meta = pickle.load(f)
        self.stoi, self.itos = meta['stoi'], meta['itos']
        self.vocab_size = meta['vocab_size']
        
        # Helper functions
        self.encode = lambda s: [self.stoi.get(c, self.stoi.get('A', 7)) for c in s]
        self.decode = lambda l: ''.join([self.itos.get(i, 'X') for i in l])
        
        # Define amino acid properties for probing
        self.aa_properties = {
            'hydrophobic': ['A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W'],
            'hydrophilic': ['S', 'T', 'N', 'Q', 'C'],
            'charged_positive': ['R', 'H', 'K'],
            'charged_negative': ['D', 'E'],
            'aromatic': ['F', 'Y', 'W'],
            'small': ['G', 'A', 'S'],
            'large': ['F', 'Y', 'W', 'R', 'K'],
            'polar': ['S', 'T', 'N', 'Q', 'Y', 'C'],
            'nonpolar': ['A', 'V', 'I', 'L', 'M', 'F', 'W', 'P', 'G']
        }
        
        print(f"Initialized ProbeAnalyzer with vocab size: {self.vocab_size}")
    
    def extract_embeddings(self, protein1_seq: str, protein2_seq: str, 
                          layer_indices: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
        """
        Extract embeddings from different layers of the model.
        
        Args:
            protein1_seq: First protein sequence
            protein2_seq: Second protein sequence
            layer_indices: Which layers to extract (None = all layers)
        
        Returns:
            Dictionary with embeddings from each layer
        """
        # Prepare input
        sep_token = '[SEP]'
        combined_seq = protein1_seq + sep_token + protein2_seq
        tokens = self.encode(combined_seq)
        idx = torch.tensor([tokens], dtype=torch.long).to(self.device)
        
        # Get embeddings from different layers
        embeddings = {}
        
        with torch.no_grad():
            # Get token embeddings
            b, t = idx.size()
            pos = torch.arange(0, t, dtype=torch.long, device=self.device)
            
            # Initial embeddings
            tok_emb = self.model.transformer.wte(idx)
            pos_emb = self.model.transformer.wpe(pos)
            x = self.model.transformer.drop(tok_emb + pos_emb)
            embeddings['layer_0_input'] = x.clone()
            
            # Extract from each transformer block
            if layer_indices is None:
                layer_indices = range(len(self.model.transformer.h))
            
            for i, block in enumerate(self.model.transformer.h):
                x = block(x)
                if i in layer_indices:
                    embeddings[f'layer_{i+1}'] = x.clone()
            
            # Final layer norm
            x = self.model.transformer.ln_f(x)
            embeddings['final'] = x.clone()
        
        # Also get position info for analysis
        sep_pos = combined_seq.find(sep_token)
        embeddings['metadata'] = {
            'protein1_range': (0, sep_pos),
            'protein2_range': (sep_pos + len(sep_token), len(combined_seq)),
            'sep_position': sep_pos,
            'sequence': combined_seq,
            'tokens': tokens
        }
        
        return embeddings
    
    
    def probe_all_properties(self, protein_pairs: List[Tuple[str, str]], 
                            layer_name: str = 'final') -> pd.DataFrame:
        """
        Probe all amino acid properties across multiple protein pairs.
        
        Args:
            protein_pairs: List of (protein1, protein2) tuples
            layer_name: Which layer to probe
        
        Returns:
            DataFrame with probe results for each property
        """
        results_list = []
        
        # Collect embeddings
        all_embeddings = []
        all_sequences = []
        
        print(f"Extracting embeddings from {len(protein_pairs)} protein pairs...")
        for p1, p2 in protein_pairs:
            emb_dict = self.extract_embeddings(p1, p2)
            if layer_name in emb_dict:
                all_embeddings.append(emb_dict[layer_name].squeeze(0))
                all_sequences.append(emb_dict['metadata']['sequence'])
        
        if not all_embeddings:
            print("No embeddings extracted!")
            return pd.DataFrame()
        
        # Probe each property
        print(f"\nProbing amino acid properties in {layer_name} layer:")
        for prop_name, aa_set in self.aa_properties.items():
            print(f"  - Probing {prop_name}...")
            
            # Collect all embeddings and labels for this property
            X_all = []
            y_all = []
            
            for emb, seq in zip(all_embeddings, all_sequences):
                # emb shape: [seq_len, hidden_dim]
                for j, aa in enumerate(seq):
                    if aa not in ['[', ']', 'S', 'E', 'P']:  # Skip special tokens
                        X_all.append(emb[j].cpu().numpy())
                        y_all.append(1 if aa in aa_set else 0)
            
            if X_all:
                X = np.array(X_all)
                y = np.array(y_all)
                
                # Probe this property
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                probe = LogisticRegression(max_iter=1000, random_state=42)
                
                # Skip cross-validation for speed, just do train-test split
                from sklearn.model_selection import train_test_split
                if len(np.unique(y)) > 1 and len(y) > 10:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y, test_size=0.3, random_state=42, stratify=y
                    )
                    probe.fit(X_train, y_train)
                    y_pred = probe.predict(X_test)
                    y_pred_proba = probe.predict_proba(X_test)[:, 1]
                    
                    results = {
                        'property': prop_name,
                        'accuracy': accuracy_score(y_test, y_pred),
                        'auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5,
                        'layer': layer_name,
                        'n_samples': len(y),
                        'n_positive': sum(y),
                        'n_negative': len(y) - sum(y),
                        'train_size': len(y_train),
                        'test_size': len(y_test)
                    }
                else:
                    # Not enough data for proper evaluation
                    results = {
                        'property': prop_name,
                        'accuracy': 0.5,
                        'auc': 0.5,
                        'layer': layer_name,
                        'n_samples': len(y),
                        'n_positive': sum(y),
                        'n_negative': len(y) - sum(y),
                        'train_size': 0,
                        'test_size': 0
                    }
                results_list.append(results)
        
        # Create results DataFrame
        df = pd.DataFrame(results_list)
        return df
    
    def probe_interaction_sites(self, protein_pairs: List[Tuple[str, str]], 
                               labels: List[int]) -> Dict[str, Any]:
        """
        Probe if the model learns to identify potential interaction sites.
        
        Args:
            protein_pairs: List of (protein1, protein2) tuples
            labels: Binary labels (1 for interacting, 0 for non-interacting)
        
        Returns:
            Analysis of position-specific patterns in interacting vs non-interacting pairs
        """
        results = {
            'position_importance': {},
            'cross_attention_patterns': {},
            'interaction_motifs': []
        }
        
        # Separate interacting and non-interacting pairs
        pos_pairs = [(p1, p2) for (p1, p2), label in zip(protein_pairs, labels) if label == 1]
        neg_pairs = [(p1, p2) for (p1, p2), label in zip(protein_pairs, labels) if label == 0]
        
        print(f"Analyzing {len(pos_pairs)} interacting and {len(neg_pairs)} non-interacting pairs")
        
        # Extract embeddings for both groups
        for pairs, pair_type in [(pos_pairs, 'interacting'), (neg_pairs, 'non_interacting')]:
            position_scores = []
            
            for p1, p2 in pairs[:20]:  # Limit for computational efficiency
                emb_dict = self.extract_embeddings(p1, p2)
                final_emb = emb_dict['final'].squeeze(0)
                
                # Get importance scores (norm of embeddings as proxy)
                importance = torch.norm(final_emb, dim=1).cpu().numpy()
                position_scores.append(importance)
            
            if position_scores:
                results['position_importance'][pair_type] = np.mean(position_scores, axis=0)
        
        return results
    
    def probe_layer_specialization(self, protein_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Analyze what different layers specialize in learning.
        
        Args:
            protein_pairs: List of protein pairs to analyze
        
        Returns:
            DataFrame showing property detection performance by layer
        """
        layer_results = []
        
        # Probe properties at different layers
        n_layers = len(self.model.transformer.h)
        layers_to_probe = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
        
        for layer_idx in layers_to_probe:
            print(f"\nProbing layer {layer_idx+1}/{n_layers}...")
            df = self.probe_all_properties(protein_pairs[:10], f'layer_{layer_idx+1}')
            df['layer_index'] = layer_idx + 1
            df['layer_position'] = f"Layer {layer_idx+1}"
            layer_results.append(df)
        
        # Also probe final layer
        print(f"\nProbing final layer...")
        df = self.probe_all_properties(protein_pairs[:10], 'final')
        df['layer_index'] = n_layers + 1
        df['layer_position'] = "Final"
        layer_results.append(df)
        
        return pd.concat(layer_results, ignore_index=True)
    
    def visualize_probe_results(self, probe_df: pd.DataFrame, output_dir: str):
        """Create visualizations of probe analysis results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Property detection accuracy heatmap
        plt.figure(figsize=(10, 8))
        if 'layer_position' in probe_df.columns:
            pivot_df = probe_df.pivot(index='property', columns='layer_position', values='auc')
            sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='YlOrRd', 
                       cbar_kws={'label': 'AUC Score'})
            plt.title('Amino Acid Property Detection by Layer')
            plt.xlabel('Layer')
            plt.ylabel('Property')
        else:
            plt.bar(probe_df['property'], probe_df['auc'])
            plt.xticks(rotation=45)
            plt.ylabel('AUC Score')
            plt.title('Amino Acid Property Detection Performance')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'property_detection_performance.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'property_detection_performance.svg', bbox_inches='tight')
        plt.close()
        
        # 2. Layer specialization plot
        if 'layer_index' in probe_df.columns:
            plt.figure(figsize=(12, 6))
            for prop in probe_df['property'].unique():
                prop_data = probe_df[probe_df['property'] == prop]
                plt.plot(prop_data['layer_index'], prop_data['auc'], 
                        marker='o', label=prop, linewidth=2)
            
            plt.xlabel('Layer Index')
            plt.ylabel('AUC Score')
            plt.title('Property Detection Performance Across Layers')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'layer_specialization.png', dpi=300, bbox_inches='tight')
            plt.savefig(output_dir / 'layer_specialization.svg', bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to {output_dir}")
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save probe analysis results."""
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved to {output_path}")


def main():
    """Run probe analysis on example protein pairs."""
    import sys
    import csv
    
    # Model paths
    out_dir = 'ppiGPT_MED4_solo/out_3e'  # Using the available checkpoint
    model_path = os.path.join(out_dir, 'ckpt.pt')
    meta_path = 'ppiGPT_MED4_solo/data/shakespeare_char/meta.pkl'
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Initialize config
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    
    # Load state dict
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Initialize analyzer
    analyzer = ProbeAnalyzer(model, meta_path)
    
    # Load real protein pairs from MED4 dataset
    csv_path = 'ppiGPT_MED4_solo/MED4-PPIs-low-confidence_ppiGPLM_cleaned2_prompts.csv'
    protein_pairs = []
    
    print("Loading protein pairs from dataset...")
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i >= 3:  # Limit to first 3 pairs for faster testing
                break
            # Extract protein sequences from the CSV format
            # Format: <ps1>,SEQUENCE1,<ps2>,SEQUENCE2,<
            if len(row) >= 4 and row[0] == '<ps1>' and row[2] == '<ps2>':
                protein1 = row[1]
                protein2 = row[3]
                protein_pairs.append((protein1, protein2))
    
    print(f"Loaded {len(protein_pairs)} protein pairs")
    
    # Run probing analysis
    print("\n1. Probing amino acid properties...")
    probe_results = analyzer.probe_all_properties(protein_pairs)
    print("\nProbe Results:")
    print(probe_results)
    
    # Skip layer specialization for now - focus on final layer
    # print("\n2. Analyzing layer specialization...")
    # layer_results = analyzer.probe_layer_specialization(protein_pairs)
    
    # Save results
    output_dir = "probe_analysis_results"
    Path(output_dir).mkdir(exist_ok=True)
    
    probe_results.to_csv(f"{output_dir}/property_probe_results.csv", index=False)
    # layer_results.to_csv(f"{output_dir}/layer_specialization_results.csv", index=False)
    
    # Create visualizations
    analyzer.visualize_probe_results(probe_results, output_dir)
    
    print(f"\nAnalysis complete! Results saved to {output_dir}/")


if __name__ == "__main__":
    main()