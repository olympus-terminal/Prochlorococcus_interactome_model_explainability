"""
Counterfactual Explanation Generator for PPI Model.
Finds minimal sequence changes that flip interaction predictions.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'ppiGPT_MED4_solo'))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import heapq
from collections import defaultdict

from model import GPTConfig, GPT


@dataclass
class CounterfactualEdit:
    """Represents a single edit to generate a counterfactual."""
    position: int
    original_aa: str
    new_aa: str
    protein_idx: int  # 0 for protein1, 1 for protein2
    impact_score: float
    
    def __repr__(self):
        protein = f"Protein{self.protein_idx+1}"
        return f"{protein}[{self.position}]: {self.original_aa}→{self.new_aa} (impact={self.impact_score:.3f})"


@dataclass 
class Counterfactual:
    """Represents a complete counterfactual explanation."""
    edits: List[CounterfactualEdit]
    original_pred: float
    new_pred: float
    original_seq1: str
    original_seq2: str
    modified_seq1: str
    modified_seq2: str
    
    @property
    def num_edits(self) -> int:
        return len(self.edits)
    
    @property
    def prediction_change(self) -> float:
        return abs(self.new_pred - self.original_pred)


class CounterfactualGenerator:
    """Generate counterfactual explanations for PPI predictions."""
    
    def __init__(self, model, tokenizer_meta_path: str):
        """
        Initialize counterfactual generator.
        
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
        
        # Standard amino acids
        self.amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
                           'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        
        # Amino acid properties for chemically plausible substitutions
        self.aa_groups = {
            'hydrophobic': ['A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W'],
            'polar': ['S', 'T', 'N', 'Q', 'Y', 'C'],
            'positive': ['R', 'H', 'K'],
            'negative': ['D', 'E'],
            'aromatic': ['F', 'Y', 'W'],
            'small': ['G', 'A', 'S'],
            'proline': ['P'],  # Special case - often disrupts structure
        }
        
        # Create similarity matrix for plausible substitutions
        self.aa_similarity = self._compute_aa_similarity()
        
        print(f"Initialized CounterfactualGenerator with vocab size: {self.vocab_size}")
    
    def _compute_aa_similarity(self) -> Dict[str, Set[str]]:
        """Compute which amino acids can plausibly substitute for each other."""
        similarity = defaultdict(set)
        
        # Each AA can substitute with itself
        for aa in self.amino_acids:
            similarity[aa].add(aa)
        
        # Add substitutions within groups
        for group_name, group_aas in self.aa_groups.items():
            for aa1 in group_aas:
                for aa2 in group_aas:
                    similarity[aa1].add(aa2)
        
        # Add specific plausible substitutions based on biochemistry
        plausible_pairs = [
            ('D', 'E'), ('E', 'D'),  # Acidic
            ('K', 'R'), ('R', 'K'),  # Basic
            ('S', 'T'), ('T', 'S'),  # Hydroxyl
            ('I', 'V'), ('V', 'I'), ('I', 'L'), ('L', 'I'), ('V', 'L'), ('L', 'V'),  # Aliphatic
            ('F', 'Y'), ('Y', 'F'),  # Aromatic with OH
            ('N', 'Q'), ('Q', 'N'),  # Amide
        ]
        
        for aa1, aa2 in plausible_pairs:
            similarity[aa1].add(aa2)
            similarity[aa2].add(aa1)
        
        return dict(similarity)
    
    def get_prediction(self, protein1_seq: str, protein2_seq: str) -> float:
        """Get model prediction for a protein pair."""
        # Format as in training data
        prompt = f"<ps1>,{protein1_seq},<ps2>,{protein2_seq},<"
        
        tokens = self.encode(prompt)
        idx = torch.tensor([tokens], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            logits, _ = self.model(idx)
            
            # Get probability of interaction
            # Token '1' (index 3) represents interaction, '0' (index 2) represents non-interaction
            interaction_token = 3  # Token index for '1'
            non_interaction_token = 2  # Token index for '0'
            
            logits_last = logits[0, -1, :]
            probs = F.softmax(logits_last, dim=-1)
            
            # Get probabilities for both classes
            prob_interact = probs[interaction_token].item()
            prob_non_interact = probs[non_interaction_token].item()
            
            # Normalize to get interaction probability
            total_prob = prob_interact + prob_non_interact
            if total_prob > 0:
                interaction_prob = prob_interact / total_prob
            else:
                interaction_prob = 0.5
        
        return interaction_prob
    
    def compute_gradient_importance(self, protein1_seq: str, protein2_seq: str) -> Dict[str, np.ndarray]:
        """Compute gradient-based importance for each position."""
        # Format as in training data
        prompt = f"<ps1>,{protein1_seq},<ps2>,{protein2_seq},<"
        
        tokens = self.encode(prompt)
        idx = torch.tensor([tokens], dtype=torch.long, requires_grad=False).to(self.device)
        
        # Enable gradients temporarily
        self.model.zero_grad()
        
        # Create embedding that requires grad
        embeddings = self.model.transformer.wte(idx).detach().requires_grad_(True)
        
        # Forward pass
        pos = torch.arange(0, idx.size(1), dtype=torch.long, device=self.device)
        pos_emb = self.model.transformer.wpe(pos)
        x = self.model.transformer.drop(embeddings + pos_emb)
        
        for block in self.model.transformer.h:
            x = block(x)
        x = self.model.transformer.ln_f(x)
        logits = self.model.lm_head(x)
        
        # Get interaction probability
        interaction_token = self.stoi.get('1', 1)
        logits_last = logits[0, -1, :]
        probs = F.softmax(logits_last, dim=-1)
        interaction_prob = probs[interaction_token]
        
        # Compute gradients
        interaction_prob.backward()
        
        # Get gradient magnitudes
        grad_magnitudes = torch.norm(embeddings.grad[0], dim=1).cpu().numpy()
        
        # Find positions of protein sequences in the prompt
        # Format: <ps1>,PROTEIN1,<ps2>,PROTEIN2,<
        ps1_start = prompt.find(',') + 1
        ps1_end = prompt.find(',<ps2>')
        ps2_start = prompt.find(',<ps2>,') + 7
        ps2_end = prompt.rfind(',<')
        
        importance = {
            'protein1': grad_magnitudes[ps1_start:ps1_end],
            'protein2': grad_magnitudes[ps2_start:ps2_end],
            'combined': grad_magnitudes
        }
        
        return importance
    
    def generate_counterfactual_greedy(self, protein1_seq: str, protein2_seq: str,
                                     target_flip: bool = True,
                                     max_edits: int = 5,
                                     threshold: float = 0.5) -> Optional[Counterfactual]:
        """
        Generate counterfactual using greedy search.
        
        Args:
            protein1_seq: First protein sequence
            protein2_seq: Second protein sequence  
            target_flip: If True, flip from positive to negative; else negative to positive
            max_edits: Maximum number of edits allowed
            threshold: Decision threshold for interaction
            
        Returns:
            Counterfactual object or None if not found
        """
        original_pred = self.get_prediction(protein1_seq, protein2_seq)
        
        # Determine target direction
        if target_flip:
            if original_pred < threshold:
                print(f"Already non-interacting (prob={original_pred:.3f})")
                return None
            target_satisfied = lambda p: p < threshold
        else:
            if original_pred > threshold:
                print(f"Already interacting (prob={original_pred:.3f})")
                return None
            target_satisfied = lambda p: p > threshold
        
        # Get importance scores
        importance = self.compute_gradient_importance(protein1_seq, protein2_seq)
        
        # Create list of possible edits sorted by importance
        candidate_edits = []
        
        # Process protein 1
        for i, aa in enumerate(protein1_seq):
            if aa in self.amino_acids:
                for new_aa in self.aa_similarity.get(aa, []):
                    if new_aa != aa:
                        edit = CounterfactualEdit(
                            position=i,
                            original_aa=aa,
                            new_aa=new_aa,
                            protein_idx=0,
                            impact_score=importance['protein1'][i]
                        )
                        candidate_edits.append(edit)
        
        # Process protein 2  
        for i, aa in enumerate(protein2_seq):
            if aa in self.amino_acids:
                for new_aa in self.aa_similarity.get(aa, []):
                    if new_aa != aa:
                        edit = CounterfactualEdit(
                            position=i,
                            original_aa=aa,
                            new_aa=new_aa,
                            protein_idx=1,
                            impact_score=importance['protein2'][i]
                        )
                        candidate_edits.append(edit)
        
        # Sort by impact score
        candidate_edits.sort(key=lambda e: e.impact_score, reverse=True)
        
        # Greedy search
        current_seq1 = protein1_seq
        current_seq2 = protein2_seq
        applied_edits = []
        
        print(f"Searching through {len(candidate_edits)} candidate edits...")
        print(f"Target: flip prediction {'down' if target_flip else 'up'} from {original_pred:.3f}")
        
        for i, edit in enumerate(candidate_edits[:50]):  # Check first 50 candidates
            if len(applied_edits) >= max_edits:
                break
            
            # Apply edit
            if edit.protein_idx == 0:
                new_seq1 = list(current_seq1)
                new_seq1[edit.position] = edit.new_aa
                new_seq1 = ''.join(new_seq1)
                new_seq2 = current_seq2
            else:
                new_seq1 = current_seq1
                new_seq2 = list(current_seq2)
                new_seq2[edit.position] = edit.new_aa
                new_seq2 = ''.join(new_seq2)
            
            # Check prediction
            new_pred = self.get_prediction(new_seq1, new_seq2)
            
            # Keep edit if it moves in the right direction
            if target_flip and new_pred < original_pred - 0.01:  # Small tolerance
                print(f"  Edit {i}: {edit} -> pred={new_pred:.3f} ✓")
                current_seq1 = new_seq1
                current_seq2 = new_seq2
                applied_edits.append(edit)
                original_pred = new_pred  # Update for next iteration
                
                if target_satisfied(new_pred):
                    # Success!
                    return Counterfactual(
                        edits=applied_edits,
                        original_pred=self.get_prediction(protein1_seq, protein2_seq),
                        new_pred=new_pred,
                        original_seq1=protein1_seq,
                        original_seq2=protein2_seq,
                        modified_seq1=current_seq1,
                        modified_seq2=current_seq2
                    )
            elif not target_flip and new_pred > original_pred + 0.01:
                print(f"  Edit {i}: {edit} -> pred={new_pred:.3f} ✓")
                current_seq1 = new_seq1
                current_seq2 = new_seq2
                applied_edits.append(edit)
                original_pred = new_pred  # Update for next iteration
                
                if target_satisfied(new_pred):
                    # Success!
                    return Counterfactual(
                        edits=applied_edits,
                        original_pred=self.get_prediction(protein1_seq, protein2_seq),
                        new_pred=new_pred,
                        original_seq1=protein1_seq,
                        original_seq2=protein2_seq,
                        modified_seq1=current_seq1,
                        modified_seq2=current_seq2
                    )
        
        # Failed to find counterfactual
        if applied_edits:
            # Return best attempt
            final_pred = self.get_prediction(current_seq1, current_seq2)
            print(f"Best attempt: {len(applied_edits)} edits, final pred={final_pred:.3f}")
            return Counterfactual(
                edits=applied_edits,
                original_pred=self.get_prediction(protein1_seq, protein2_seq),
                new_pred=final_pred,
                original_seq1=protein1_seq,
                original_seq2=protein2_seq,
                modified_seq1=current_seq1,
                modified_seq2=current_seq2
            )
        
        return None
    
    def analyze_counterfactuals(self, protein_pairs: List[Tuple[str, str]], 
                               labels: List[int]) -> pd.DataFrame:
        """
        Analyze counterfactuals for multiple protein pairs.
        
        Args:
            protein_pairs: List of (protein1, protein2) tuples
            labels: Binary labels (1 for interacting, 0 for non-interacting)
            
        Returns:
            DataFrame with counterfactual analysis results
        """
        results = []
        
        for (p1, p2), label in zip(protein_pairs, labels):
            pred = self.get_prediction(p1, p2)
            
            # Try to flip the prediction
            cf = self.generate_counterfactual_greedy(p1, p2, target_flip=(pred > 0.5))
            
            if cf:
                result = {
                    'original_label': label,
                    'original_pred': pred,
                    'counterfactual_pred': cf.new_pred,
                    'num_edits': cf.num_edits,
                    'prediction_change': cf.prediction_change,
                    'edits': str(cf.edits),
                    'success': (cf.original_pred > 0.5) != (cf.new_pred > 0.5)
                }
                results.append(result)
            else:
                result = {
                    'original_label': label,
                    'original_pred': pred,
                    'counterfactual_pred': pred,
                    'num_edits': 0,
                    'prediction_change': 0,
                    'edits': 'None',
                    'success': False
                }
                results.append(result)
        
        return pd.DataFrame(results)
    
    def visualize_counterfactual(self, cf: Counterfactual, output_path: Optional[str] = None):
        """Visualize a counterfactual explanation."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Sequence alignment with edits highlighted
        seq1_original = list(cf.original_seq1)
        seq1_modified = list(cf.modified_seq1)
        seq2_original = list(cf.original_seq2)
        seq2_modified = list(cf.modified_seq2)
        
        # Create color arrays
        colors1 = ['red' if seq1_original[i] != seq1_modified[i] else 'black' 
                  for i in range(len(seq1_original))]
        colors2 = ['red' if seq2_original[i] != seq2_modified[i] else 'black'
                  for i in range(len(seq2_original))]
        
        # Plot sequences (simplified visualization)
        ax1.text(0.5, 0.8, 'Protein 1 Edits:', transform=ax1.transAxes, fontweight='bold')
        ax1.text(0.5, 0.6, f'Original: {"".join(seq1_original[:50])}...', 
                transform=ax1.transAxes, fontfamily='monospace', fontsize=8)
        ax1.text(0.5, 0.4, f'Modified: {"".join(seq1_modified[:50])}...', 
                transform=ax1.transAxes, fontfamily='monospace', fontsize=8)
        
        ax1.text(0.5, 0.2, 'Protein 2 Edits:', transform=ax1.transAxes, fontweight='bold')
        ax1.text(0.5, 0.0, f'First 50 residues shown. Total edits: {cf.num_edits}',
                transform=ax1.transAxes, fontsize=8, style='italic')
        
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        
        # Plot 2: Prediction change
        predictions = [cf.original_pred, cf.new_pred]
        labels = ['Original', 'Counterfactual']
        colors = ['blue', 'red']
        
        bars = ax2.bar(labels, predictions, color=colors, alpha=0.7)
        ax2.axhline(y=0.5, color='black', linestyle='--', label='Decision threshold')
        ax2.set_ylabel('Interaction Probability')
        ax2.set_ylim(0, 1)
        ax2.legend()
        
        # Add value labels on bars
        for bar, val in zip(bars, predictions):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom')
        
        # Add edit summary
        edit_summary = '\n'.join([str(e) for e in cf.edits[:5]])  # Show first 5 edits
        if cf.num_edits > 5:
            edit_summary += f'\n... and {cf.num_edits - 5} more edits'
        
        plt.figtext(0.5, 0.02, f'Edits applied:\n{edit_summary}', 
                   ha='center', fontsize=8, wrap=True)
        
        plt.suptitle(f'Counterfactual Explanation: {cf.num_edits} edits change prediction from {cf.original_pred:.3f} to {cf.new_pred:.3f}')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            svg_path = str(output_path).replace('.png', '.svg')
            plt.savefig(svg_path, bbox_inches='tight')
        
        plt.show()
    
    def find_minimal_counterfactuals(self, protein_pairs: List[Tuple[str, str]], 
                                   max_pairs: int = 10) -> List[Counterfactual]:
        """Find minimal counterfactuals for a set of protein pairs."""
        counterfactuals = []
        
        for i, (p1, p2) in enumerate(protein_pairs[:max_pairs]):
            print(f"\nAnalyzing pair {i+1}/{min(len(protein_pairs), max_pairs)}...")
            
            # Get original prediction
            orig_pred = self.get_prediction(p1, p2)
            print(f"Original prediction: {orig_pred:.3f}")
            
            # Try to find counterfactual with more edits allowed
            cf = self.generate_counterfactual_greedy(p1, p2, max_edits=10)
            
            if cf and cf.prediction_change > 0.05:  # Lower threshold
                print(f"Found counterfactual with {cf.num_edits} edits (change: {cf.prediction_change:.3f})")
                counterfactuals.append(cf)
            else:
                print("No effective counterfactual found")
        
        return counterfactuals


def main():
    """Run counterfactual analysis on example protein pairs."""
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
    
    # Initialize generator
    generator = CounterfactualGenerator(model, meta_path)
    
    # Load some interacting protein pairs from MED4_Int file
    int_file = 'ppiGPT_MED4_solo/MED4_Int_100pairs_prompts.txt'
    protein_pairs = []
    
    print("\nLoading interacting protein pairs...")
    with open(int_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i >= 5:  # Limit for testing
                break
            parts = line.strip().split(',')
            if len(parts) >= 5 and parts[0] == '<ps1>' and parts[2] == '<ps2>':
                protein1 = parts[1]
                protein2 = parts[3]
                protein_pairs.append((protein1, protein2))
    
    print(f"Loaded {len(protein_pairs)} protein pairs")
    
    # Generate counterfactuals
    print("\nGenerating counterfactuals...")
    counterfactuals = generator.find_minimal_counterfactuals(protein_pairs)
    
    # Save results
    output_dir = Path("counterfactual_results")
    output_dir.mkdir(exist_ok=True)
    
    # Visualize first counterfactual if found
    if counterfactuals:
        print(f"\nFound {len(counterfactuals)} counterfactuals")
        generator.visualize_counterfactual(
            counterfactuals[0], 
            output_dir / "example_counterfactual.png"
        )
        
        # Save all counterfactuals
        cf_data = []
        for cf in counterfactuals:
            cf_data.append({
                'num_edits': cf.num_edits,
                'original_pred': cf.original_pred,
                'new_pred': cf.new_pred,
                'prediction_change': cf.prediction_change,
                'edits': '; '.join([str(e) for e in cf.edits])
            })
        
        cf_df = pd.DataFrame(cf_data)
        cf_df.to_csv(output_dir / "counterfactuals.csv", index=False)
        print(f"\nResults saved to {output_dir}/")
    else:
        print("\nNo counterfactuals found")


if __name__ == "__main__":
    main()