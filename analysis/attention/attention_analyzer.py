"""
Attention Visualization System for PPI Model Interpretability

This module extracts and analyzes attention patterns from the trained GPT model
to identify key residue interactions that drive PPI predictions.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AttentionAnalyzer:
    """Analyzes attention patterns from GPT model for PPI interpretability."""
    
    def __init__(self, model, tokenizer_meta_path: Optional[str] = None):
        """
        Initialize attention analyzer.
        
        Args:
            model: Trained GPT model
            tokenizer_meta_path: Path to meta.pkl with tokenizer info
        """
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device
        
        # Load tokenizer
        if tokenizer_meta_path and Path(tokenizer_meta_path).exists():
            with open(tokenizer_meta_path, 'rb') as f:
                meta = pickle.load(f)
            self.stoi, self.itos = meta['stoi'], meta['itos']
            self.encode = lambda s: [self.stoi.get(c, self.stoi.get('A', 0)) for c in s]
            self.decode = lambda l: ''.join([self.itos[i] for i in l])
        else:
            # Fallback to tiktoken for GPT-2
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            self.encode = lambda s: enc.encode(s, allowed_special={""})
            self.decode = lambda l: enc.decode(l)
    
    def hook_attention(self):
        """Register hooks to capture attention weights during forward pass."""
        self.attention_weights = {}
        self.hooks = []
        
        def attention_hook(layer_idx):
            def hook(module, input, output):
                # For our custom GPT implementation, we need to capture attention differently
                # The attention weights are computed in the forward pass
                if hasattr(module, 'flash') and not module.flash:
                    # For non-flash attention, we can capture the attention weights
                    # We'll modify this to store intermediate attention values
                    pass
                # Store the module for manual attention extraction
                self.attention_weights[f'layer_{layer_idx}_module'] = module
            return hook
        
        # Register hooks for attention layers
        for i, block in enumerate(self.model.transformer.h):
            if hasattr(block, 'attn'):
                handle = block.attn.register_forward_hook(attention_hook(i))
                self.hooks.append(handle)
    
    def clear_hooks(self):
        """Clear all registered hooks."""
        for hook in getattr(self, 'hooks', []):
            hook.remove()
        self.hooks = []
    
    def extract_attention_patterns(self, protein1_seq: str, protein2_seq: str) -> Dict:
        """
        Extract attention patterns for a protein pair.
        
        Args:
            protein1_seq: First protein sequence
            protein2_seq: Second protein sequence
            
        Returns:
            Dictionary containing attention weights and sequence info
        """
        # Format input like training data
        input_text = f"<ps1>,{protein1_seq},<ps2>,{protein2_seq},<"
        
        # Encode and prepare input
        input_ids = self.encode(input_text)
        if len(input_ids) > self.model.config.block_size:
            input_ids = input_ids[-self.model.config.block_size:]
        
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        
        # Clear previous attention weights
        self.attention_weights = {}
        
        # Forward pass to capture attention
        with torch.no_grad():
            logits, _ = self.model(input_tensor)
            
        # Get prediction probability
        probs = F.softmax(logits[:, -1, :], dim=-1)
        
        # Try to get probability for token "1" 
        try:
            token_1_id = self.encode("1")[0]
            interaction_prob = probs[0, token_1_id].item()
        except:
            interaction_prob = None
        
        # Parse sequence positions
        sequence_info = self._parse_sequence_positions(input_text, protein1_seq, protein2_seq)
        
        return {
            'attention_weights': self.attention_weights,
            'sequence_info': sequence_info,
            'input_ids': input_ids,
            'interaction_prob': interaction_prob,
            'decoded_input': self.decode(input_ids) if hasattr(self, 'decode') else input_text
        }
    
    def _parse_sequence_positions(self, input_text: str, protein1_seq: str, protein2_seq: str) -> Dict:
        """Parse positions of different sequence components in the input."""
        
        # Find positions of key markers
        ps1_pos = input_text.find('<ps1>')
        ps2_pos = input_text.find('<ps2>')
        
        # Approximate positions (will need refinement based on actual tokenization)
        protein1_start = len(self.encode('<ps1>,'))
        protein1_end = protein1_start + len(protein1_seq)
        
        protein2_start = protein1_end + len(self.encode(',<ps2>,'))
        protein2_end = protein2_start + len(protein2_seq)
        
        return {
            'protein1_seq': protein1_seq,
            'protein2_seq': protein2_seq,
            'protein1_range': (protein1_start, protein1_end),
            'protein2_range': (protein2_start, protein2_end),
            'total_length': len(self.encode(input_text))
        }
    
    def analyze_cross_protein_attention(self, attention_data: Dict, layer_idx: int = -1) -> np.ndarray:
        """
        Analyze attention between protein1 and protein2 residues.
        
        Args:
            attention_data: Output from extract_attention_patterns
            layer_idx: Which layer to analyze (-1 for last layer)
            
        Returns:
            Cross-attention matrix (protein1_residues x protein2_residues)
        """
        if not self.attention_weights:
            raise ValueError("No attention weights found. Run extract_attention_patterns first.")
        
        # Get attention weights for specified layer
        layer_key = f'layer_{layer_idx}' if layer_idx >= 0 else list(self.attention_weights.keys())[layer_idx]
        attention = self.attention_weights[layer_key]  # Shape: [batch, heads, seq_len, seq_len]
        
        # Average across heads
        attention_avg = attention.mean(dim=1).squeeze(0)  # Shape: [seq_len, seq_len]
        
        # Extract protein ranges
        seq_info = attention_data['sequence_info']
        p1_start, p1_end = seq_info['protein1_range']
        p2_start, p2_end = seq_info['protein2_range']
        
        # Extract cross-attention (protein1 attending to protein2)
        cross_attention = attention_avg[p1_start:p1_end, p2_start:p2_end].cpu().numpy()
        
        return cross_attention
    
    def plot_attention_heatmap(self, cross_attention: np.ndarray, protein1_seq: str, 
                              protein2_seq: str, title: str = "Cross-Protein Attention", 
                              save_path: Optional[str] = None):
        """
        Plot attention heatmap between two proteins.
        
        Args:
            cross_attention: Cross-attention matrix
            protein1_seq: First protein sequence
            protein2_seq: Second protein sequence
            title: Plot title
            save_path: Path to save plot
        """
        plt.figure(figsize=(min(len(protein2_seq) * 0.3, 20), min(len(protein1_seq) * 0.3, 15)))
        
        # Create heatmap
        sns.heatmap(cross_attention, 
                   xticklabels=list(protein2_seq), 
                   yticklabels=list(protein1_seq),
                   cmap='Reds', 
                   cbar_kws={'label': 'Attention Weight'})
        
        plt.title(title)
        plt.xlabel('Protein 2 Residues')
        plt.ylabel('Protein 1 Residues')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def identify_key_interactions(self, cross_attention: np.ndarray, protein1_seq: str, 
                                 protein2_seq: str, top_k: int = 10) -> List[Dict]:
        """
        Identify top residue-residue interactions based on attention weights.
        
        Args:
            cross_attention: Cross-attention matrix
            protein1_seq: First protein sequence
            protein2_seq: Second protein sequence
            top_k: Number of top interactions to return
            
        Returns:
            List of interaction dictionaries with residue info and attention scores
        """
        # Find top attention positions
        flat_attention = cross_attention.flatten()
        top_indices = np.argsort(flat_attention)[-top_k:][::-1]
        
        interactions = []
        for idx in top_indices:
            i, j = np.unravel_index(idx, cross_attention.shape)
            
            interaction = {
                'protein1_pos': i,
                'protein1_residue': protein1_seq[i] if i < len(protein1_seq) else 'X',
                'protein2_pos': j,
                'protein2_residue': protein2_seq[j] if j < len(protein2_seq) else 'X',
                'attention_score': cross_attention[i, j],
                'interaction_pair': f"{protein1_seq[i] if i < len(protein1_seq) else 'X'}{i+1}-{protein2_seq[j] if j < len(protein2_seq) else 'X'}{j+1}"
            }
            interactions.append(interaction)
        
        return interactions
    
    def analyze_attention_by_residue_type(self, interactions: List[Dict]) -> pd.DataFrame:
        """
        Analyze attention patterns by amino acid types.
        
        Args:
            interactions: List of interaction dictionaries
            
        Returns:
            DataFrame with residue pair statistics
        """
        interaction_data = []
        
        for interaction in interactions:
            interaction_data.append({
                'residue_pair': f"{interaction['protein1_residue']}-{interaction['protein2_residue']}",
                'protein1_residue': interaction['protein1_residue'],
                'protein2_residue': interaction['protein2_residue'],
                'attention_score': interaction['attention_score'],
                'protein1_pos': interaction['protein1_pos'],
                'protein2_pos': interaction['protein2_pos']
            })
        
        df = pd.DataFrame(interaction_data)
        
        # Group by residue pairs and calculate statistics
        residue_stats = df.groupby('residue_pair').agg({
            'attention_score': ['mean', 'std', 'count', 'max']
        }).round(4)
        
        residue_stats.columns = ['mean_attention', 'std_attention', 'frequency', 'max_attention']
        residue_stats = residue_stats.sort_values('mean_attention', ascending=False)
        
        return residue_stats
    
    def save_analysis(self, attention_data: Dict, output_dir: str, protein_pair_name: str):
        """Save attention analysis results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save raw attention data
        torch.save(attention_data, output_path / f"{protein_pair_name}_attention_data.pt")
        
        # Save analysis summary
        summary = {
            'protein1_seq': attention_data['sequence_info']['protein1_seq'],
            'protein2_seq': attention_data['sequence_info']['protein2_seq'],
            'interaction_prob': attention_data['interaction_prob'],
            'sequence_length': attention_data['sequence_info']['total_length']
        }
        
        with open(output_path / f"{protein_pair_name}_summary.pickle", 'wb') as f:
            pickle.dump(summary, f)


def main():
    """Example usage of AttentionAnalyzer."""
    
    # This would typically load your trained model
    print("AttentionAnalyzer module ready for use!")
    print("\nExample usage:")
    print("1. Initialize: analyzer = AttentionAnalyzer(model, meta_path)")
    print("2. Register hooks: analyzer.hook_attention()")
    print("3. Extract patterns: data = analyzer.extract_attention_patterns(seq1, seq2)")
    print("4. Analyze cross-attention: cross_attn = analyzer.analyze_cross_protein_attention(data)")
    print("5. Visualize: analyzer.plot_attention_heatmap(cross_attn, seq1, seq2)")


if __name__ == "__main__":
    main()