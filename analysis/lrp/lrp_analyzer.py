"""
Layer-wise Relevance Propagation (LRP) for PPI Model.
Traces relevance scores through the model to understand information flow.
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
import warnings
warnings.filterwarnings('ignore')

from model import GPTConfig, GPT, Block, CausalSelfAttention, MLP, LayerNorm


class LRPAnalyzer:
    """Layer-wise Relevance Propagation for understanding information flow in PPI model."""
    
    def __init__(self, model, tokenizer_meta_path: str):
        """
        Initialize LRP analyzer.
        
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
        
        # Store intermediate activations during forward pass
        self.activations = {}
        self.attention_weights = {}
        
        # LRP parameters
        self.epsilon = 1e-10
        self.gamma = 0.1  # For LRP-γ variant
        
        print(f"Initialized LRPAnalyzer with vocab size: {self.vocab_size}")
    
    def register_hooks(self):
        """Register forward hooks to capture activations."""
        self.handles = []
        
        # Hook for embeddings
        def embedding_hook(module, input, output):
            self.activations['embedding'] = output.detach()
        
        self.handles.append(
            self.model.transformer.wte.register_forward_hook(embedding_hook)
        )
        
        # Hooks for each transformer block
        for i, block in enumerate(self.model.transformer.h):
            # Attention output
            def make_attn_hook(layer_idx):
                def hook(module, input, output):
                    self.activations[f'block_{layer_idx}_attn'] = output.detach()
                return hook
            
            self.handles.append(
                block.attn.register_forward_hook(make_attn_hook(i))
            )
            
            # MLP output
            def make_mlp_hook(layer_idx):
                def hook(module, input, output):
                    self.activations[f'block_{layer_idx}_mlp'] = output.detach()
                return hook
            
            self.handles.append(
                block.mlp.register_forward_hook(make_mlp_hook(i))
            )
            
            # Block output
            def make_block_hook(layer_idx):
                def hook(module, input, output):
                    self.activations[f'block_{layer_idx}_output'] = output.detach()
                return hook
            
            self.handles.append(
                block.register_forward_hook(make_block_hook(i))
            )
        
        # Final layer norm
        def final_ln_hook(module, input, output):
            self.activations['final_ln'] = output.detach()
        
        self.handles.append(
            self.model.transformer.ln_f.register_forward_hook(final_ln_hook)
        )
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
    
    def forward_pass_with_activations(self, protein1: str, protein2: str) -> Tuple[torch.Tensor, Dict]:
        """
        Perform forward pass and collect all activations.
        
        Args:
            protein1: First protein sequence
            protein2: Second protein sequence
            
        Returns:
            logits, activations dictionary
        """
        # Clear previous activations
        self.activations = {}
        self.attention_weights = {}
        
        # Register hooks
        self.register_hooks()
        
        try:
            # Format input
            prompt = f"<ps1>,{protein1},<ps2>,{protein2},<"
            tokens = self.encode(prompt)
            idx = torch.tensor([tokens], dtype=torch.long).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                logits, _ = self.model(idx)
            
            # Get prediction
            probs = F.softmax(logits[0, -1, :], dim=-1)
            interaction_prob = probs[3] / (probs[2] + probs[3])
            
            # Store some metadata
            self.activations['metadata'] = {
                'sequence_length': len(tokens),
                'protein1_end': prompt.find(',<ps2>'),
                'protein2_start': prompt.find(',<ps2>,') + 7,
                'prediction': interaction_prob.item()
            }
            
        finally:
            # Always remove hooks
            self.remove_hooks()
        
        return logits, self.activations
    
    def compute_relevance_scores(self, target_class: int = 3) -> Dict[str, torch.Tensor]:
        """
        Compute LRP relevance scores for each layer.
        
        Args:
            target_class: Target class for relevance (3 for interaction, 2 for non-interaction)
            
        Returns:
            Dictionary of relevance scores for each layer
        """
        relevance_scores = {}
        
        # Start from output layer
        # R_output = indicator for target class
        output_shape = self.activations['final_ln'].shape
        R = torch.zeros(output_shape[0], output_shape[1], self.vocab_size).to(self.device)
        R[0, -1, target_class] = 1.0  # Relevance for target class at last position
        
        # Propagate through LM head (simple linear layer)
        # For linear layers: R_i = sum_j (a_i * w_ij / sum_k a_k * w_kj) * R_j
        final_activations = self.activations['final_ln'][0, -1, :].unsqueeze(0)  # [1, hidden_dim]
        
        # Get LM head weights
        lm_head_weight = self.model.lm_head.weight.T  # [hidden_dim, vocab_size]
        
        # Compute relevance for final layer norm output
        z = final_activations @ lm_head_weight + self.epsilon
        s = R[0, -1, :].unsqueeze(0) / z  # [1, vocab_size]
        c = s @ lm_head_weight.T  # [1, hidden_dim]
        R_final_ln = (final_activations * c).squeeze(0)  # [hidden_dim]
        
        relevance_scores['final_ln'] = R_final_ln
        
        # Propagate through transformer blocks (simplified)
        current_relevance = R_final_ln.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]
        
        for i in range(len(self.model.transformer.h) - 1, -1, -1):
            block_name = f'block_{i}_output'
            
            if i > 0:
                prev_block_name = f'block_{i-1}_output'
                prev_activation = self.activations[prev_block_name]
            else:
                prev_activation = self.activations['embedding']
            
            # Simplified relevance propagation through residual connection
            # R_prev = R_current * (activation_prev / activation_current)
            current_activation = self.activations[block_name]
            
            # Handle last position
            if current_activation.shape[1] > 1:
                # Take last position
                current_act_last = current_activation[0, -1:, :]
                prev_act_last = prev_activation[0, -1:, :]
                
                # Propagate relevance (simplified LRP-ε rule)
                denominator = current_act_last + self.epsilon
                relevance_ratio = prev_act_last / denominator
                block_relevance = current_relevance * relevance_ratio
                
                relevance_scores[f'block_{i}_relevance'] = block_relevance.squeeze()
                current_relevance = block_relevance
        
        # Propagate to embeddings
        embedding_relevance = current_relevance
        relevance_scores['embedding'] = embedding_relevance.squeeze()
        
        return relevance_scores
    
    def propagate_relevance_attention(self, attention_weights: torch.Tensor, 
                                    value_relevance: torch.Tensor) -> torch.Tensor:
        """
        Propagate relevance through attention mechanism.
        
        Args:
            attention_weights: Attention weights [batch, heads, seq, seq]
            value_relevance: Relevance of value vectors [batch, seq, hidden]
            
        Returns:
            Input relevance [batch, seq, hidden]
        """
        # Simplified: distribute relevance according to attention weights
        # In practice, this would require more careful handling of multi-head attention
        
        # Average over heads
        avg_attention = attention_weights.mean(dim=1)  # [batch, seq, seq]
        
        # Propagate relevance
        input_relevance = avg_attention.transpose(-1, -2) @ value_relevance
        
        return input_relevance
    
    def analyze_information_flow(self, protein1: str, protein2: str) -> Dict[str, Any]:
        """
        Analyze how information flows through the model for a protein pair.
        
        Args:
            protein1: First protein sequence
            protein2: Second protein sequence
            
        Returns:
            Dictionary with flow analysis results
        """
        # Forward pass with activation collection
        logits, activations = self.forward_pass_with_activations(protein1, protein2)
        
        # Compute relevance scores
        relevance_scores = self.compute_relevance_scores(target_class=3)  # For interaction
        
        # Analyze flow patterns
        flow_analysis = {
            'prediction': activations['metadata']['prediction'],
            'sequence_length': activations['metadata']['sequence_length'],
            'layer_relevance': {},
            'position_importance': {},
            'cross_protein_flow': {}
        }
        
        # Compute layer-wise relevance statistics
        for key, relevance in relevance_scores.items():
            if 'block' in key and len(relevance.shape) > 0:
                flow_analysis['layer_relevance'][key] = {
                    'mean': relevance.abs().mean().item(),
                    'max': relevance.abs().max().item(),
                    'std': relevance.abs().std().item()
                }
        
        # Analyze position importance (simplified)
        if 'embedding' in relevance_scores:
            embedding_rel = relevance_scores['embedding']
            if len(embedding_rel.shape) > 0:
                # Get position importance
                position_scores = embedding_rel.abs()
                
                # Split into protein regions
                p1_end = activations['metadata']['protein1_end']
                p2_start = activations['metadata']['protein2_start']
                
                flow_analysis['position_importance'] = {
                    'protein1_total': position_scores[:p1_end].sum().item() if p1_end > 0 else 0,
                    'protein2_total': position_scores[p2_start:].sum().item() if p2_start < len(position_scores) else 0,
                    'separator_total': position_scores[p1_end:p2_start].sum().item() if p2_start > p1_end else 0
                }
        
        return flow_analysis
    
    def visualize_relevance_flow(self, protein1: str, protein2: str, output_dir: str):
        """
        Create visualizations of relevance propagation.
        
        Args:
            protein1: First protein sequence
            protein2: Second protein sequence
            output_dir: Directory to save visualizations
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Analyze information flow
        flow_analysis = self.analyze_information_flow(protein1, protein2)
        
        # 1. Layer-wise relevance progression
        if flow_analysis['layer_relevance']:
            layers = sorted([k for k in flow_analysis['layer_relevance'].keys() 
                           if 'block' in k], key=lambda x: int(x.split('_')[1]))
            
            means = [flow_analysis['layer_relevance'][l]['mean'] for l in layers]
            maxs = [flow_analysis['layer_relevance'][l]['max'] for l in layers]
            
            plt.figure(figsize=(10, 6))
            x = range(len(layers))
            plt.plot(x, means, 'b-o', label='Mean relevance', linewidth=2)
            plt.plot(x, maxs, 'r-s', label='Max relevance', linewidth=2)
            plt.xlabel('Layer')
            plt.ylabel('Relevance Score')
            plt.title(f'Information Flow Through Layers\nPrediction: {flow_analysis["prediction"]:.3f}')
            plt.xticks(x, [f'Layer {i}' for i in range(len(layers))], rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(output_dir / 'layer_relevance_flow.png', dpi=300, bbox_inches='tight')
            plt.savefig(output_dir / 'layer_relevance_flow.svg', bbox_inches='tight')
            plt.close()
        
        # 2. Position importance distribution
        if 'position_importance' in flow_analysis and flow_analysis['position_importance']:
            pos_imp = flow_analysis['position_importance']
            
            plt.figure(figsize=(8, 6))
            categories = ['Protein 1', 'Separator', 'Protein 2']
            values = [pos_imp.get('protein1_total', 0), 
                     pos_imp.get('separator_total', 0),
                     pos_imp.get('protein2_total', 0)]
            
            bars = plt.bar(categories, values, alpha=0.7, color=['blue', 'gray', 'green'])
            plt.ylabel('Total Relevance')
            plt.title('Relevance Distribution Across Sequence Regions')
            
            # Add value labels
            for bar, val in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'position_relevance_distribution.png', dpi=300, bbox_inches='tight')
            plt.savefig(output_dir / 'position_relevance_distribution.svg', bbox_inches='tight')
            plt.close()
        
        # 3. Conceptual information flow diagram
        self._create_flow_diagram(flow_analysis, output_dir / 'information_flow_diagram.png')
        
        print(f"Visualizations saved to {output_dir}")
    
    def _create_flow_diagram(self, flow_analysis: Dict, output_path: Path):
        """Create a conceptual diagram of information flow."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Draw conceptual flow
        layers = 6  # Simplified representation
        y_positions = np.linspace(0.9, 0.1, layers)
        
        # Draw boxes for each layer
        box_width = 0.15
        box_height = 0.08
        
        components = ['Protein 1', 'Separator', 'Protein 2']
        x_positions = [0.2, 0.5, 0.8]
        
        for i, y in enumerate(y_positions):
            for j, (comp, x) in enumerate(zip(components, x_positions)):
                # Draw box
                rect = plt.Rectangle((x - box_width/2, y - box_height/2), 
                                   box_width, box_height,
                                   facecolor=['lightblue', 'lightgray', 'lightgreen'][j],
                                   edgecolor='black', linewidth=1)
                ax.add_patch(rect)
                
                if i == 0:
                    ax.text(x, y, comp, ha='center', va='center', fontsize=10)
                
                # Draw arrows to next layer
                if i < len(y_positions) - 1:
                    ax.arrow(x, y - box_height/2, 0, -(y_positions[i] - y_positions[i+1] - box_height),
                            head_width=0.02, head_length=0.02, fc='gray', ec='gray', alpha=0.5)
        
        # Add layer labels
        for i, y in enumerate(y_positions):
            ax.text(0.05, y, f'Layer {i}', ha='right', va='center', fontsize=10)
        
        # Add title and labels
        ax.text(0.5, 0.95, 'Information Flow Through Transformer Layers', 
               ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(0.5, 0.02, f'Final Prediction: {flow_analysis["prediction"]:.3f}', 
               ha='center', va='center', fontsize=12)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(str(output_path).replace('.png', '.svg'), bbox_inches='tight')
        plt.close()
    
    def compare_relevance_patterns(self, protein_pairs: List[Tuple[str, str]], 
                                  labels: List[int]) -> pd.DataFrame:
        """
        Compare relevance patterns between interacting and non-interacting pairs.
        
        Args:
            protein_pairs: List of protein pairs
            labels: Binary labels (1 for interacting, 0 for non-interacting)
            
        Returns:
            DataFrame with comparative analysis
        """
        results = []
        
        for (p1, p2), label in zip(protein_pairs, labels):
            flow_analysis = self.analyze_information_flow(p1, p2)
            
            result = {
                'true_label': label,
                'prediction': flow_analysis['prediction'],
                'sequence_length': flow_analysis['sequence_length']
            }
            
            # Add position importance if available
            if 'position_importance' in flow_analysis:
                pos_imp = flow_analysis['position_importance']
                total_relevance = sum(pos_imp.values())
                
                if total_relevance > 0:
                    result['protein1_relevance_ratio'] = pos_imp.get('protein1_total', 0) / total_relevance
                    result['protein2_relevance_ratio'] = pos_imp.get('protein2_total', 0) / total_relevance
                    result['separator_relevance_ratio'] = pos_imp.get('separator_total', 0) / total_relevance
                else:
                    result['protein1_relevance_ratio'] = 0
                    result['protein2_relevance_ratio'] = 0
                    result['separator_relevance_ratio'] = 0
            
            # Add layer statistics
            if flow_analysis['layer_relevance']:
                # Get final layer relevance
                final_layers = [k for k in flow_analysis['layer_relevance'].keys() if 'block' in k]
                if final_layers:
                    last_layer = sorted(final_layers, key=lambda x: int(x.split('_')[1]))[-1]
                    result['final_layer_mean_relevance'] = flow_analysis['layer_relevance'][last_layer]['mean']
                    result['final_layer_max_relevance'] = flow_analysis['layer_relevance'][last_layer]['max']
            
            results.append(result)
        
        return pd.DataFrame(results)


def main():
    """Run LRP analysis on example protein pairs."""
    import csv
    
    # Model paths
    out_dir = 'ppiGPT_MED4_solo/out_3e'
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
    analyzer = LRPAnalyzer(model, meta_path)
    
    # Load protein pairs
    protein_pairs = []
    labels = []
    
    # Load interacting pairs
    int_file = 'ppiGPT_MED4_solo/MED4_Int_100pairs_prompts.txt'
    print(f"\nLoading interacting pairs from {int_file}...")
    with open(int_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines[:5]):  # First 5 interacting
            parts = line.strip().split(',')
            if len(parts) >= 5 and parts[0] == '<ps1>' and parts[2] == '<ps2>':
                protein1 = parts[1]
                protein2 = parts[3]
                protein_pairs.append((protein1, protein2))
                labels.append(1)
    
    # Load non-interacting pairs
    rnd_file = 'ppiGPT_MED4_solo/MED4_100_RND_prompts.txt'
    print(f"Loading non-interacting pairs from {rnd_file}...")
    with open(rnd_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines[:5]):  # First 5 non-interacting
            parts = line.strip().split(',')
            if len(parts) >= 5 and parts[0] == '<ps1>' and parts[2] == '<ps2>':
                protein1 = parts[1]
                protein2 = parts[3]
                protein_pairs.append((protein1, protein2))
                labels.append(0)
    
    print(f"Loaded {len(protein_pairs)} protein pairs ({sum(labels)} interacting, {len(labels)-sum(labels)} non-interacting)")
    
    # Analyze first pair in detail
    if protein_pairs:
        print("\n1. Detailed analysis of first protein pair...")
        p1, p2 = protein_pairs[0]
        label = labels[0]
        
        output_dir = Path("lrp_analysis_results")
        output_dir.mkdir(exist_ok=True)
        
        # Visualize relevance flow
        analyzer.visualize_relevance_flow(p1, p2, output_dir)
        
        # Get flow analysis
        flow_analysis = analyzer.analyze_information_flow(p1, p2)
        print(f"\nFlow Analysis Results:")
        print(f"  Prediction: {flow_analysis['prediction']:.3f}")
        print(f"  True label: {'Interacting' if label == 1 else 'Non-interacting'}")
        
        if 'position_importance' in flow_analysis:
            print(f"\nPosition Importance:")
            for key, value in flow_analysis['position_importance'].items():
                print(f"  {key}: {value:.3f}")
    
    # Compare patterns across multiple pairs
    print("\n2. Comparing relevance patterns...")
    comparison_df = analyzer.compare_relevance_patterns(protein_pairs, labels)
    
    if not comparison_df.empty:
        print("\nRelevance Pattern Comparison:")
        print(comparison_df)
        
        # Save results
        comparison_df.to_csv(output_dir / "relevance_pattern_comparison.csv", index=False)
        
        # Analyze differences between interacting and non-interacting
        interacting = comparison_df[comparison_df['true_label'] == 1]
        non_interacting = comparison_df[comparison_df['true_label'] == 0]
        
        print("\nAverage relevance patterns:")
        print("Interacting pairs:")
        print(f"  Protein1 relevance: {interacting['protein1_relevance_ratio'].mean():.3f}")
        print(f"  Protein2 relevance: {interacting['protein2_relevance_ratio'].mean():.3f}")
        print("\nNon-interacting pairs:")
        print(f"  Protein1 relevance: {non_interacting['protein1_relevance_ratio'].mean():.3f}")
        print(f"  Protein2 relevance: {non_interacting['protein2_relevance_ratio'].mean():.3f}")
        
        print(f"\nAnalysis complete! Results saved to {output_dir}/")
    else:
        print("No comparison results generated")


if __name__ == "__main__":
    main()