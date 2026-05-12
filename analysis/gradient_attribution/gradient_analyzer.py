"""
Gradient-based interpretability analysis for PPI model.
Uses gradient attribution methods to identify important residues.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model'))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Optional

from model import GPTConfig, GPT

class GradientAnalyzer:
    """Gradient-based attribution analysis for PPI predictions."""
    
    def __init__(self, model, tokenizer_meta_path: str):
        """
        Initialize gradient analyzer.
        
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
        
        print(f"Initialized analyzer with vocab size: {self.vocab_size}")
        print(f"Available tokens: {list(self.itos.values())}")
    
    def compute_input_gradients(self, protein1_seq: str, protein2_seq: str) -> Dict:
        """
        Compute gradients of interaction prediction with respect to input tokens.
        
        Args:
            protein1_seq: First protein sequence
            protein2_seq: Second protein sequence
            
        Returns:
            Dictionary with gradients and sequence information
        """
        # Format input
        input_text = f"<ps1>,{protein1_seq},<ps2>,{protein2_seq},<"
        input_ids = self.encode(input_text)
        
        # Truncate if necessary
        max_length = min(len(input_ids), self.model.config.block_size)
        if len(input_ids) > max_length:
            input_ids = input_ids[-max_length:]
        
        # Create input tensor 
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        
        # Convert to one-hot for gradient computation
        one_hot = F.one_hot(input_tensor, num_classes=self.vocab_size).float()  # [1, seq_len, vocab_size]
        one_hot.requires_grad_(True)
        
        # Get embeddings through one-hot multiplication
        embeddings = torch.matmul(one_hot, self.model.transformer.wte.weight)  # [1, seq_len, emb_dim]
        
        # Forward pass through the rest of the model
        pos = torch.arange(0, embeddings.size(1), dtype=torch.long, device=self.device)
        pos_emb = self.model.transformer.wpe(pos)
        x = self.model.transformer.drop(embeddings + pos_emb)
        
        for block in self.model.transformer.h:
            x = block(x)
        
        x = self.model.transformer.ln_f(x)
        logits = self.model.lm_head(x[:, [-1], :])  # Only last token
        
        # Get prediction for interaction (token "1")
        token_1_id = self.stoi.get('1', 3)
        interaction_logit = logits[0, 0, token_1_id]
        interaction_prob = F.softmax(logits[0, 0, :], dim=0)[token_1_id]
        
        # Compute gradients
        interaction_logit.backward()
        
        # Get gradients with respect to one-hot input
        input_gradients = one_hot.grad  # [1, seq_len, vocab_size]
        
        # Compute gradient magnitudes (L2 norm across vocab dimension)
        gradient_magnitudes = torch.norm(input_gradients, dim=2, p=2).squeeze(0)  # [seq_len]
        
        # Parse sequence positions
        decoded = self.decode(input_ids)
        sequence_info = self._parse_sequence_positions(decoded, protein1_seq, protein2_seq)
        
        return {
            'input_ids': input_ids,
            'gradient_magnitudes': gradient_magnitudes.detach().cpu().numpy(),
            'interaction_prob': interaction_prob.item(),
            'interaction_logit': interaction_logit.item(),
            'sequence_info': sequence_info,
            'decoded_input': decoded
        }
    
    def _parse_sequence_positions(self, decoded_input: str, protein1_seq: str, protein2_seq: str) -> Dict:
        """Parse positions of proteins in the decoded input."""
        
        # Find marker positions
        ps1_marker = "<ps1>,"
        ps2_marker = ",<ps2>,"
        
        ps1_pos = decoded_input.find(ps1_marker)
        ps2_pos = decoded_input.find(ps2_marker)
        
        if ps1_pos >= 0 and ps2_pos >= 0:
            protein1_start = ps1_pos + len(ps1_marker)
            protein1_end = ps2_pos
            protein2_start = ps2_pos + len(ps2_marker)
            protein2_end = len(decoded_input) - 2  # Remove trailing ",<"
            
            # Extract actual sequences from decoded input
            extracted_seq1 = decoded_input[protein1_start:protein1_end]
            extracted_seq2 = decoded_input[protein2_start:protein2_end]
        else:
            # Fallback
            protein1_start = 6  # After "<ps1>,"
            protein1_end = protein1_start + len(protein1_seq)
            protein2_start = protein1_end + 6  # After ",<ps2>,"
            protein2_end = protein2_start + len(protein2_seq)
            extracted_seq1 = protein1_seq
            extracted_seq2 = protein2_seq
        
        return {
            'protein1_seq': extracted_seq1,
            'protein2_seq': extracted_seq2,
            'protein1_range': (protein1_start, protein1_end),
            'protein2_range': (protein2_start, protein2_end),
            'total_length': len(decoded_input)
        }
    
    def analyze_residue_importance(self, gradient_data: Dict) -> Dict:
        """
        Analyze importance of individual residues based on gradients.
        
        Args:
            gradient_data: Output from compute_input_gradients
            
        Returns:
            Analysis of residue importance
        """
        gradients = gradient_data['gradient_magnitudes']
        seq_info = gradient_data['sequence_info']
        
        # Extract gradients for each protein
        p1_start, p1_end = seq_info['protein1_range']
        p2_start, p2_end = seq_info['protein2_range']
        
        # Ensure ranges are within bounds
        p1_end = min(p1_end, len(gradients))
        p2_end = min(p2_end, len(gradients))
        
        if p1_start < p1_end:
            protein1_gradients = gradients[p1_start:p1_end]
        else:
            protein1_gradients = np.array([])
        
        if p2_start < p2_end:
            protein2_gradients = gradients[p2_start:p2_end]
        else:
            protein2_gradients = np.array([])
        
        # Create residue importance lists
        protein1_importance = []
        protein2_importance = []
        
        # Protein 1 analysis
        protein1_seq = seq_info['protein1_seq']
        for i, residue in enumerate(protein1_seq):
            if i < len(protein1_gradients):
                importance = {
                    'residue': residue,
                    'position': i,
                    'gradient_magnitude': protein1_gradients[i],
                    'protein': 1
                }
                protein1_importance.append(importance)
        
        # Protein 2 analysis
        protein2_seq = seq_info['protein2_seq']
        for i, residue in enumerate(protein2_seq):
            if i < len(protein2_gradients):
                importance = {
                    'residue': residue,
                    'position': i,
                    'gradient_magnitude': protein2_gradients[i],
                    'protein': 2
                }
                protein2_importance.append(importance)
        
        return {
            'protein1_importance': protein1_importance,
            'protein2_importance': protein2_importance,
            'protein1_gradients': protein1_gradients,
            'protein2_gradients': protein2_gradients
        }
    
    def integrated_gradients(self, protein1_seq: str, protein2_seq: str, steps: int = 20) -> Dict:
        """
        Compute integrated gradients for more robust attribution.
        
        Args:
            protein1_seq: First protein sequence
            protein2_seq: Second protein sequence
            steps: Number of integration steps
            
        Returns:
            Integrated gradients analysis
        """
        # Format input
        input_text = f"<ps1>,{protein1_seq},<ps2>,{protein2_seq},<"
        input_ids = self.encode(input_text)
        
        # Truncate if necessary
        max_length = min(len(input_ids), self.model.config.block_size)
        if len(input_ids) > max_length:
            input_ids = input_ids[-max_length:]
        
        # Create baseline (all tokens set to padding/unknown)
        baseline_ids = [self.stoi.get('A', 7)] * len(input_ids)  # Use 'A' as baseline
        
        # Convert to tensors
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        baseline_tensor = torch.tensor(baseline_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        
        # Get token "1" id for target
        token_1_id = self.stoi.get('1', 3)
        
        # Accumulate gradients across interpolation steps
        accumulated_gradients = None
        
        for step in range(steps):
            # Interpolate between baseline and input
            alpha = step / (steps - 1) if steps > 1 else 1.0
            
            # Linear interpolation in embedding space
            baseline_emb = self.model.transformer.wte(baseline_tensor)
            input_emb = self.model.transformer.wte(input_tensor)
            interpolated_emb = baseline_emb + alpha * (input_emb - baseline_emb)
            interpolated_emb.requires_grad_(True)
            
            # Forward pass
            pos = torch.arange(0, interpolated_emb.size(1), dtype=torch.long, device=self.device)
            pos_emb = self.model.transformer.wpe(pos)
            x = self.model.transformer.drop(interpolated_emb + pos_emb)
            
            for block in self.model.transformer.h:
                x = block(x)
            
            x = self.model.transformer.ln_f(x)
            logits = self.model.lm_head(x[:, [-1], :])
            
            # Get prediction for interaction
            interaction_logit = logits[0, 0, token_1_id]
            
            # Compute gradients
            if interpolated_emb.grad is not None:
                interpolated_emb.grad.zero_()
            
            interaction_logit.backward(retain_graph=True)
            step_gradients = interpolated_emb.grad
            
            # Accumulate gradients
            if accumulated_gradients is None:
                accumulated_gradients = step_gradients.clone()
            else:
                accumulated_gradients += step_gradients
        
        # Average gradients and compute integrated gradients
        avg_gradients = accumulated_gradients / steps
        
        # Compute difference in embeddings
        input_emb = self.model.transformer.wte(input_tensor)
        baseline_emb = self.model.transformer.wte(baseline_tensor)
        emb_diff = input_emb - baseline_emb
        
        # Integrated gradients = average_gradients * (input - baseline)
        integrated_grads = avg_gradients * emb_diff
        
        # Compute magnitude for each token
        integrated_magnitudes = torch.norm(integrated_grads, dim=2, p=2).squeeze(0)
        
        # Get final prediction
        with torch.no_grad():
            logits, _ = self.model(input_tensor)
            interaction_prob = F.softmax(logits[0, -1, :], dim=0)[token_1_id].item()
        
        # Parse sequence info
        decoded = self.decode(input_ids)
        sequence_info = self._parse_sequence_positions(decoded, protein1_seq, protein2_seq)
        
        return {
            'input_ids': input_ids,
            'integrated_gradients': integrated_magnitudes.detach().cpu().numpy(),
            'interaction_prob': interaction_prob,
            'sequence_info': sequence_info,
            'decoded_input': decoded
        }
    
    def plot_gradient_importance(self, importance_data: Dict, title: str = "Residue Importance", 
                                save_path: Optional[str] = None):
        """Plot gradient-based importance scores."""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot protein 1
        if importance_data['protein1_importance']:
            p1_data = importance_data['protein1_importance']
            positions = [d['position'] for d in p1_data]
            residues = [d['residue'] for d in p1_data]
            gradients = [d['gradient_magnitude'] for d in p1_data]
            
            bars1 = ax1.bar(positions, gradients, color='blue', alpha=0.7)
            ax1.set_title('Protein 1 Residue Importance')
            ax1.set_xlabel('Position')
            ax1.set_ylabel('Gradient Magnitude')
            ax1.set_xticks(positions)
            ax1.set_xticklabels(residues, rotation=45)
        
        # Plot protein 2
        if importance_data['protein2_importance']:
            p2_data = importance_data['protein2_importance']
            positions = [d['position'] for d in p2_data]
            residues = [d['residue'] for d in p2_data]
            gradients = [d['gradient_magnitude'] for d in p2_data]
            
            bars2 = ax2.bar(positions, gradients, color='red', alpha=0.7)
            ax2.set_title('Protein 2 Residue Importance')
            ax2.set_xlabel('Position')
            ax2.set_ylabel('Gradient Magnitude')
            ax2.set_xticks(positions)
            ax2.set_xticklabels(residues, rotation=45)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            # Save as SVG and PNG
            svg_path = str(save_path).replace('.png', '.svg')
            plt.savefig(svg_path, format='svg', bbox_inches='tight')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def get_top_important_residues(self, importance_data: Dict, top_k: int = 5) -> Dict:
        """Get top important residues for each protein."""
        
        results = {
            'protein1_top': [],
            'protein2_top': []
        }
        
        # Sort protein 1 residues by importance
        if importance_data['protein1_importance']:
            p1_sorted = sorted(importance_data['protein1_importance'], 
                             key=lambda x: x['gradient_magnitude'], reverse=True)
            results['protein1_top'] = p1_sorted[:top_k]
        
        # Sort protein 2 residues by importance
        if importance_data['protein2_importance']:
            p2_sorted = sorted(importance_data['protein2_importance'], 
                             key=lambda x: x['gradient_magnitude'], reverse=True)
            results['protein2_top'] = p2_sorted[:top_k]
        
        return results


def main():
    """Example usage of GradientAnalyzer."""
    print("GradientAnalyzer module ready!")
    print("Usage:")
    print("1. analyzer = GradientAnalyzer(model, meta_path)")
    print("2. grad_data = analyzer.compute_input_gradients(seq1, seq2)")
    print("3. importance = analyzer.analyze_residue_importance(grad_data)")
    print("4. analyzer.plot_gradient_importance(importance)")

if __name__ == "__main__":
    main()