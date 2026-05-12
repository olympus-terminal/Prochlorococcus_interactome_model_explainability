"""
Systematic sequence perturbation analysis for identifying critical residues.
This tool performs comprehensive mutations to identify residue importance.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model'))

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Optional, Union
from itertools import combinations

from model import GPTConfig, GPT

class PerturbationAnalyzer:
    """Systematic perturbation analysis for protein sequence importance."""
    
    def __init__(self, model, tokenizer_meta_path: str):
        """
        Initialize perturbation analyzer.
        
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
        
        # Get amino acid tokens
        self.amino_acids = [aa for aa in 'ACDEFGHIKLMNPQRSTVWY' if aa in self.stoi]
        self.amino_acid_ids = [self.stoi[aa] for aa in self.amino_acids]
        
        print(f"Initialized perturbation analyzer")
        print(f"Available amino acids: {self.amino_acids}")
    
    def get_baseline_prediction(self, protein1_seq: str, protein2_seq: str) -> Dict:
        """Get baseline prediction for a protein pair."""
        
        input_text = f"<ps1>,{protein1_seq},<ps2>,{protein2_seq},<"
        input_ids = self.encode(input_text)
        
        # Truncate if needed
        max_len = min(len(input_ids), self.model.config.block_size)
        input_ids = input_ids[:max_len]
        
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            logits, _ = self.model(input_tensor)
            probs = F.softmax(logits[0, -1, :], dim=0)
            
            # Get probabilities for interaction tokens
            token_1_id = self.stoi.get('1', 3)
            token_0_id = self.stoi.get('0', 2)
            
            prob_1 = probs[token_1_id].item()
            prob_0 = probs[token_0_id].item()
        
        return {
            'input_text': input_text,
            'input_ids': input_ids,
            'prob_interaction': prob_1,
            'prob_no_interaction': prob_0,
            'logits': logits[0, -1, :].cpu().numpy()
        }
    
    def single_residue_perturbation(self, protein1_seq: str, protein2_seq: str, 
                                   perturbation_type: str = 'alanine') -> Dict:
        """
        Perform single residue perturbations.
        
        Args:
            protein1_seq: First protein sequence
            protein2_seq: Second protein sequence
            perturbation_type: Type of perturbation ('alanine', 'all_amino_acids', 'deletion')
            
        Returns:
            Dictionary with perturbation results
        """
        baseline = self.get_baseline_prediction(protein1_seq, protein2_seq)
        baseline_prob = baseline['prob_interaction']
        
        results = {
            'baseline': baseline,
            'protein1_perturbations': [],
            'protein2_perturbations': []
        }
        
        # Perturbation targets
        if perturbation_type == 'alanine':
            targets = ['A']
        elif perturbation_type == 'all_amino_acids':
            targets = self.amino_acids
        elif perturbation_type == 'deletion':
            targets = ['DEL']  # Special marker for deletion
        else:
            targets = ['A']  # Default to alanine
        
        # Perturb protein 1
        print(f"Perturbing protein 1: {protein1_seq}")
        for pos in range(len(protein1_seq)):
            original_aa = protein1_seq[pos]
            
            for target_aa in targets:
                if target_aa == original_aa:
                    continue  # Skip if same as original
                
                if target_aa == 'DEL':
                    # Deletion
                    perturbed_seq1 = protein1_seq[:pos] + protein1_seq[pos+1:]
                else:
                    # Substitution
                    perturbed_seq1 = protein1_seq[:pos] + target_aa + protein1_seq[pos+1:]
                
                # Get prediction for perturbed sequence
                perturbed_pred = self.get_baseline_prediction(perturbed_seq1, protein2_seq)
                
                effect = baseline_prob - perturbed_pred['prob_interaction']
                
                results['protein1_perturbations'].append({
                    'position': pos,
                    'original_aa': original_aa,
                    'target_aa': target_aa,
                    'effect': effect,
                    'baseline_prob': baseline_prob,
                    'perturbed_prob': perturbed_pred['prob_interaction'],
                    'protein': 1
                })
        
        # Perturb protein 2
        print(f"Perturbing protein 2: {protein2_seq}")
        for pos in range(len(protein2_seq)):
            original_aa = protein2_seq[pos]
            
            for target_aa in targets:
                if target_aa == original_aa:
                    continue  # Skip if same as original
                
                if target_aa == 'DEL':
                    # Deletion
                    perturbed_seq2 = protein2_seq[:pos] + protein2_seq[pos+1:]
                else:
                    # Substitution
                    perturbed_seq2 = protein2_seq[:pos] + target_aa + protein2_seq[pos+1:]
                
                # Get prediction for perturbed sequence
                perturbed_pred = self.get_baseline_prediction(protein1_seq, perturbed_seq2)
                
                effect = baseline_prob - perturbed_pred['prob_interaction']
                
                results['protein2_perturbations'].append({
                    'position': pos,
                    'original_aa': original_aa,
                    'target_aa': target_aa,
                    'effect': effect,
                    'baseline_prob': baseline_prob,
                    'perturbed_prob': perturbed_pred['prob_interaction'],
                    'protein': 2
                })
        
        return results
    
    def double_perturbation(self, protein1_seq: str, protein2_seq: str, 
                           top_positions: Dict, max_combinations: int = 10) -> Dict:
        """
        Perform double perturbations on top important positions.
        
        Args:
            protein1_seq: First protein sequence
            protein2_seq: Second protein sequence
            top_positions: Top positions from single perturbation analysis
            max_combinations: Maximum number of combinations to test
            
        Returns:
            Dictionary with double perturbation results
        """
        baseline = self.get_baseline_prediction(protein1_seq, protein2_seq)
        baseline_prob = baseline['prob_interaction']
        
        results = {
            'baseline': baseline,
            'double_perturbations': []
        }
        
        # Get top positions for each protein
        p1_top = top_positions.get('protein1_top', [])[:5]  # Top 5
        p2_top = top_positions.get('protein2_top', [])[:5]  # Top 5
        
        combinations_tested = 0
        
        # Test combinations within protein 1
        for i, pos1_info in enumerate(p1_top):
            for j, pos2_info in enumerate(p1_top[i+1:], i+1):
                if combinations_tested >= max_combinations:
                    break
                
                pos1, pos2 = pos1_info['position'], pos2_info['position']
                
                # Double alanine substitution
                perturbed_seq1 = list(protein1_seq)
                perturbed_seq1[pos1] = 'A'
                perturbed_seq1[pos2] = 'A'
                perturbed_seq1 = ''.join(perturbed_seq1)
                
                perturbed_pred = self.get_baseline_prediction(perturbed_seq1, protein2_seq)
                effect = baseline_prob - perturbed_pred['prob_interaction']
                
                results['double_perturbations'].append({
                    'positions': (pos1, pos2),
                    'proteins': (1, 1),
                    'original_aas': (protein1_seq[pos1], protein1_seq[pos2]),
                    'effect': effect,
                    'baseline_prob': baseline_prob,
                    'perturbed_prob': perturbed_pred['prob_interaction']
                })
                
                combinations_tested += 1
        
        # Test combinations within protein 2
        for i, pos1_info in enumerate(p2_top):
            for j, pos2_info in enumerate(p2_top[i+1:], i+1):
                if combinations_tested >= max_combinations:
                    break
                
                pos1, pos2 = pos1_info['position'], pos2_info['position']
                
                # Double alanine substitution
                perturbed_seq2 = list(protein2_seq)
                perturbed_seq2[pos1] = 'A'
                perturbed_seq2[pos2] = 'A'
                perturbed_seq2 = ''.join(perturbed_seq2)
                
                perturbed_pred = self.get_baseline_prediction(protein1_seq, perturbed_seq2)
                effect = baseline_prob - perturbed_pred['prob_interaction']
                
                results['double_perturbations'].append({
                    'positions': (pos1, pos2),
                    'proteins': (2, 2),
                    'original_aas': (protein2_seq[pos1], protein2_seq[pos2]),
                    'effect': effect,
                    'baseline_prob': baseline_prob,
                    'perturbed_prob': perturbed_pred['prob_interaction']
                })
                
                combinations_tested += 1
        
        # Test cross-protein combinations
        for pos1_info in p1_top[:3]:  # Top 3 from protein 1
            for pos2_info in p2_top[:3]:  # Top 3 from protein 2
                if combinations_tested >= max_combinations:
                    break
                
                pos1, pos2 = pos1_info['position'], pos2_info['position']
                
                # Double alanine substitution across proteins
                perturbed_seq1 = protein1_seq[:pos1] + 'A' + protein1_seq[pos1+1:]
                perturbed_seq2 = protein2_seq[:pos2] + 'A' + protein2_seq[pos2+1:]
                
                perturbed_pred = self.get_baseline_prediction(perturbed_seq1, perturbed_seq2)
                effect = baseline_prob - perturbed_pred['prob_interaction']
                
                results['double_perturbations'].append({
                    'positions': (pos1, pos2),
                    'proteins': (1, 2),
                    'original_aas': (protein1_seq[pos1], protein2_seq[pos2]),
                    'effect': effect,
                    'baseline_prob': baseline_prob,
                    'perturbed_prob': perturbed_pred['prob_interaction']
                })
                
                combinations_tested += 1
        
        return results
    
    def analyze_perturbation_results(self, perturbation_results: Dict) -> Dict:
        """Analyze perturbation results to identify critical residues."""
        
        analysis = {
            'protein1_analysis': {},
            'protein2_analysis': {},
            'summary': {}
        }
        
        # Analyze protein 1
        if perturbation_results['protein1_perturbations']:
            p1_data = pd.DataFrame(perturbation_results['protein1_perturbations'])
            
            # Group by position to get max effect per position
            p1_by_pos = p1_data.groupby('position').agg({
                'effect': ['max', 'min', 'mean'],
                'original_aa': 'first'
            }).round(4)
            
            p1_by_pos.columns = ['max_effect', 'min_effect', 'mean_effect', 'original_aa']
            p1_by_pos['abs_max_effect'] = p1_by_pos[['max_effect', 'min_effect']].abs().max(axis=1)
            p1_by_pos = p1_by_pos.sort_values('abs_max_effect', ascending=False)
            
            analysis['protein1_analysis'] = {
                'by_position': p1_by_pos,
                'top_positions': p1_by_pos.head(5).to_dict('index'),
                'critical_residues': p1_by_pos[p1_by_pos['abs_max_effect'] > 0.05].to_dict('index')
            }
        
        # Analyze protein 2
        if perturbation_results['protein2_perturbations']:
            p2_data = pd.DataFrame(perturbation_results['protein2_perturbations'])
            
            # Group by position to get max effect per position
            p2_by_pos = p2_data.groupby('position').agg({
                'effect': ['max', 'min', 'mean'],
                'original_aa': 'first'
            }).round(4)
            
            p2_by_pos.columns = ['max_effect', 'min_effect', 'mean_effect', 'original_aa']
            p2_by_pos['abs_max_effect'] = p2_by_pos[['max_effect', 'min_effect']].abs().max(axis=1)
            p2_by_pos = p2_by_pos.sort_values('abs_max_effect', ascending=False)
            
            analysis['protein2_analysis'] = {
                'by_position': p2_by_pos,
                'top_positions': p2_by_pos.head(5).to_dict('index'),
                'critical_residues': p2_by_pos[p2_by_pos['abs_max_effect'] > 0.05].to_dict('index')
            }
        
        # Overall summary
        baseline_prob = perturbation_results['baseline']['prob_interaction']
        
        analysis['summary'] = {
            'baseline_interaction_prob': baseline_prob,
            'num_protein1_critical': len(analysis['protein1_analysis'].get('critical_residues', {})),
            'num_protein2_critical': len(analysis['protein2_analysis'].get('critical_residues', {})),
        }
        
        return analysis
    
    def plot_perturbation_effects(self, perturbation_results: Dict, analysis: Dict, 
                                 save_path: Optional[str] = None):
        """Plot perturbation effects."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Protein 1 position effects
        if 'protein1_analysis' in analysis and 'by_position' in analysis['protein1_analysis']:
            p1_data = analysis['protein1_analysis']['by_position']
            ax1 = axes[0, 0]
            positions = p1_data.index
            effects = p1_data['abs_max_effect']
            
            bars = ax1.bar(positions, effects, alpha=0.7, color='blue')
            ax1.set_title('Protein 1: Max Perturbation Effect by Position')
            ax1.set_xlabel('Position')
            ax1.set_ylabel('Max |Effect|')
            ax1.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='Threshold')
            
            # Label amino acids
            for i, (pos, row) in enumerate(p1_data.iterrows()):
                if row['abs_max_effect'] > 0.02:  # Label significant positions
                    ax1.text(pos, row['abs_max_effect'] + 0.01, row['original_aa'], 
                            ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Protein 2 position effects
        if 'protein2_analysis' in analysis and 'by_position' in analysis['protein2_analysis']:
            p2_data = analysis['protein2_analysis']['by_position']
            ax2 = axes[0, 1]
            positions = p2_data.index
            effects = p2_data['abs_max_effect']
            
            bars = ax2.bar(positions, effects, alpha=0.7, color='red')
            ax2.set_title('Protein 2: Max Perturbation Effect by Position')
            ax2.set_xlabel('Position')
            ax2.set_ylabel('Max |Effect|')
            ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='Threshold')
            
            # Label amino acids
            for i, (pos, row) in enumerate(p2_data.iterrows()):
                if row['abs_max_effect'] > 0.02:  # Label significant positions
                    ax2.text(pos, row['abs_max_effect'] + 0.01, row['original_aa'], 
                            ha='center', va='bottom', fontsize=8)
        
        # Plot 3: Effect distribution
        ax3 = axes[1, 0]
        all_effects = []
        if perturbation_results['protein1_perturbations']:
            all_effects.extend([p['effect'] for p in perturbation_results['protein1_perturbations']])
        if perturbation_results['protein2_perturbations']:
            all_effects.extend([p['effect'] for p in perturbation_results['protein2_perturbations']])
        
        if all_effects:
            ax3.hist(all_effects, bins=20, alpha=0.7, color='green')
            ax3.set_title('Distribution of Perturbation Effects')
            ax3.set_xlabel('Effect (Baseline - Perturbed)')
            ax3.set_ylabel('Frequency')
            ax3.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        # Plot 4: Top critical residues comparison
        ax4 = axes[1, 1]
        
        p1_top = analysis.get('protein1_analysis', {}).get('by_position', pd.DataFrame())
        p2_top = analysis.get('protein2_analysis', {}).get('by_position', pd.DataFrame())
        
        if not p1_top.empty and not p2_top.empty:
            # Get top 5 from each
            p1_top5 = p1_top.head(5)
            p2_top5 = p2_top.head(5)
            
            x_pos = np.arange(5)
            width = 0.35
            
            ax4.bar(x_pos - width/2, p1_top5['abs_max_effect'], width, 
                   label='Protein 1', alpha=0.7, color='blue')
            ax4.bar(x_pos + width/2, p2_top5['abs_max_effect'], width, 
                   label='Protein 2', alpha=0.7, color='red')
            
            ax4.set_title('Top 5 Critical Positions Comparison')
            ax4.set_xlabel('Rank')
            ax4.set_ylabel('Max |Effect|')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels([f'#{i+1}' for i in range(5)])
            ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            # Save as SVG and PNG
            svg_path = str(save_path).replace('.png', '.svg')
            plt.savefig(svg_path, format='svg', bbox_inches='tight')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def get_critical_residue_summary(self, analysis: Dict) -> Dict:
        """Get summary of critical residues."""
        
        summary = {
            'protein1_critical': [],
            'protein2_critical': [],
            'interaction_rules': []
        }
        
        # Extract critical residues for protein 1
        if 'protein1_analysis' in analysis:
            p1_critical = analysis['protein1_analysis'].get('critical_residues', {})
            for pos, data in p1_critical.items():
                summary['protein1_critical'].append({
                    'position': pos,
                    'residue': data['original_aa'],
                    'max_effect': data['abs_max_effect'],
                    'interpretation': 'Critical for interaction' if data['abs_max_effect'] > 0.1 else 'Moderate importance'
                })
        
        # Extract critical residues for protein 2
        if 'protein2_analysis' in analysis:
            p2_critical = analysis['protein2_analysis'].get('critical_residues', {})
            for pos, data in p2_critical.items():
                summary['protein2_critical'].append({
                    'position': pos,
                    'residue': data['original_aa'],
                    'max_effect': data['abs_max_effect'],
                    'interpretation': 'Critical for interaction' if data['abs_max_effect'] > 0.1 else 'Moderate importance'
                })
        
        # Generate interaction rules
        p1_top = summary['protein1_critical'][:3]  # Top 3
        p2_top = summary['protein2_critical'][:3]  # Top 3
        
        for p1_res in p1_top:
            summary['interaction_rules'].append(
                f"Protein1 {p1_res['residue']}{p1_res['position']+1} is critical (effect: {p1_res['max_effect']:.3f})"
            )
        
        for p2_res in p2_top:
            summary['interaction_rules'].append(
                f"Protein2 {p2_res['residue']}{p2_res['position']+1} is critical (effect: {p2_res['max_effect']:.3f})"
            )
        
        return summary


def main():
    """Example usage of PerturbationAnalyzer."""
    print("PerturbationAnalyzer module ready!")

if __name__ == "__main__":
    main()