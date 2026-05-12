"""
Uncertainty Quantification for PPI Model.
Estimates prediction uncertainty using multiple approaches.
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
from scipy import stats
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

from model import GPTConfig, GPT


class UncertaintyQuantifier:
    """Quantify uncertainty in PPI predictions using various methods."""
    
    def __init__(self, model, tokenizer_meta_path: str):
        """
        Initialize uncertainty quantifier.
        
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
        
        # Uncertainty parameters
        self.mc_samples = 30  # Monte Carlo samples for dropout
        self.temperature_values = [0.5, 1.0, 2.0]  # For temperature scaling
        
        print(f"Initialized UncertaintyQuantifier with vocab size: {self.vocab_size}")
    
    def enable_dropout(self):
        """Enable dropout for Monte Carlo dropout uncertainty estimation."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def disable_dropout(self):
        """Disable dropout for standard inference."""
        self.model.eval()
    
    def get_prediction(self, protein1: str, protein2: str, temperature: float = 1.0) -> Dict[str, float]:
        """
        Get model prediction with optional temperature scaling.
        
        Args:
            protein1: First protein sequence
            protein2: Second protein sequence
            temperature: Temperature for softmax scaling
            
        Returns:
            Dictionary with prediction details
        """
        prompt = f"<ps1>,{protein1},<ps2>,{protein2},<"
        tokens = self.encode(prompt)
        idx = torch.tensor([tokens], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            logits, _ = self.model(idx)
            
            # Apply temperature scaling
            logits_scaled = logits[0, -1, :] / temperature
            probs = F.softmax(logits_scaled, dim=-1)
            
            # Get probabilities for interaction classes
            prob_0 = probs[2].item()  # Non-interaction
            prob_1 = probs[3].item()  # Interaction
            
            # Normalize to binary classification
            total = prob_0 + prob_1
            if total > 0:
                interaction_prob = prob_1 / total
            else:
                interaction_prob = 0.5
        
        return {
            'interaction_prob': interaction_prob,
            'raw_prob_0': prob_0,
            'raw_prob_1': prob_1,
            'logit_0': logits[0, -1, 2].item(),
            'logit_1': logits[0, -1, 3].item(),
            'entropy': -torch.sum(probs * torch.log(probs + 1e-10)).item()
        }
    
    def monte_carlo_dropout_uncertainty(self, protein1: str, protein2: str) -> Dict[str, Any]:
        """
        Estimate uncertainty using Monte Carlo dropout.
        
        Args:
            protein1: First protein sequence
            protein2: Second protein sequence
            
        Returns:
            Dictionary with uncertainty metrics
        """
        # Enable dropout
        self.enable_dropout()
        
        predictions = []
        entropies = []
        
        # Collect multiple predictions
        for _ in range(self.mc_samples):
            pred = self.get_prediction(protein1, protein2)
            predictions.append(pred['interaction_prob'])
            entropies.append(pred['entropy'])
        
        # Disable dropout
        self.disable_dropout()
        
        predictions = np.array(predictions)
        entropies = np.array(entropies)
        
        # Calculate uncertainty metrics
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        # Predictive entropy (uncertainty)
        mean_entropy = np.mean(entropies)
        
        # Mutual information (epistemic uncertainty)
        # MI = H[E[p]] - E[H[p]]
        mean_probs = np.mean(predictions)
        entropy_of_mean = -mean_probs * np.log(mean_probs + 1e-10) - \
                          (1 - mean_probs) * np.log(1 - mean_probs + 1e-10)
        mutual_information = entropy_of_mean - mean_entropy
        
        # Coefficient of variation
        cv = std_pred / (mean_pred + 1e-10)
        
        return {
            'mean_prediction': mean_pred,
            'std_prediction': std_pred,
            'coefficient_variation': cv,
            'predictive_entropy': mean_entropy,
            'mutual_information': mutual_information,
            'epistemic_uncertainty': mutual_information,
            'aleatoric_uncertainty': mean_entropy - mutual_information,
            'predictions': predictions,
            'min_prediction': np.min(predictions),
            'max_prediction': np.max(predictions),
            'prediction_range': np.max(predictions) - np.min(predictions)
        }
    
    def temperature_scaling_analysis(self, protein1: str, protein2: str) -> Dict[str, Any]:
        """
        Analyze how predictions change with temperature scaling.
        
        Args:
            protein1: First protein sequence
            protein2: Second protein sequence
            
        Returns:
            Dictionary with temperature analysis
        """
        results = {}
        
        for temp in self.temperature_values:
            pred = self.get_prediction(protein1, protein2, temperature=temp)
            results[f'temp_{temp}'] = pred
        
        # Analyze sensitivity to temperature
        predictions = [results[f'temp_{t}']['interaction_prob'] for t in self.temperature_values]
        pred_range = max(predictions) - min(predictions)
        pred_std = np.std(predictions)
        
        return {
            'temperature_results': results,
            'prediction_range': pred_range,
            'prediction_std': pred_std,
            'temperature_sensitive': pred_range > 0.1  # Arbitrary threshold
        }
    
    def sequence_perturbation_uncertainty(self, protein1: str, protein2: str, 
                                        n_perturbations: int = 20) -> Dict[str, Any]:
        """
        Estimate uncertainty by perturbing sequences slightly.
        
        Args:
            protein1: First protein sequence
            protein2: Second protein sequence
            n_perturbations: Number of perturbations to test
            
        Returns:
            Dictionary with perturbation-based uncertainty
        """
        # Get baseline prediction
        baseline = self.get_prediction(protein1, protein2)
        baseline_pred = baseline['interaction_prob']
        
        # Conservative amino acid substitutions
        conservative_subs = {
            'A': ['V', 'G'], 'V': ['A', 'I', 'L'], 'I': ['V', 'L'], 'L': ['I', 'V', 'M'],
            'R': ['K'], 'K': ['R'], 'D': ['E'], 'E': ['D'],
            'S': ['T'], 'T': ['S'], 'F': ['Y'], 'Y': ['F'],
            'N': ['Q'], 'Q': ['N']
        }
        
        perturbation_results = []
        
        for _ in range(n_perturbations):
            # Randomly select which protein to perturb
            if np.random.random() < 0.5:
                # Perturb protein1
                seq = list(protein1)
                pos = np.random.randint(0, len(seq))
                original_aa = seq[pos]
                
                if original_aa in conservative_subs:
                    new_aa = np.random.choice(conservative_subs[original_aa])
                    seq[pos] = new_aa
                    perturbed_p1 = ''.join(seq)
                    perturbed_p2 = protein2
                else:
                    continue
            else:
                # Perturb protein2
                seq = list(protein2)
                pos = np.random.randint(0, len(seq))
                original_aa = seq[pos]
                
                if original_aa in conservative_subs:
                    new_aa = np.random.choice(conservative_subs[original_aa])
                    seq[pos] = new_aa
                    perturbed_p1 = protein1
                    perturbed_p2 = ''.join(seq)
                else:
                    continue
            
            # Get prediction for perturbed sequence
            perturbed_pred = self.get_prediction(perturbed_p1, perturbed_p2)
            perturbation_results.append(perturbed_pred['interaction_prob'])
        
        if perturbation_results:
            perturbation_results = np.array(perturbation_results)
            
            return {
                'baseline_prediction': baseline_pred,
                'mean_perturbed': np.mean(perturbation_results),
                'std_perturbed': np.std(perturbation_results),
                'max_deviation': np.max(np.abs(perturbation_results - baseline_pred)),
                'robust': np.max(np.abs(perturbation_results - baseline_pred)) < 0.1,
                'n_perturbations': len(perturbation_results)
            }
        else:
            return {
                'baseline_prediction': baseline_pred,
                'mean_perturbed': baseline_pred,
                'std_perturbed': 0.0,
                'max_deviation': 0.0,
                'robust': True,
                'n_perturbations': 0
            }
    
    def analyze_prediction_confidence(self, protein1: str, protein2: str) -> Dict[str, Any]:
        """
        Comprehensive confidence analysis combining multiple uncertainty methods.
        
        Args:
            protein1: First protein sequence
            protein2: Second protein sequence
            
        Returns:
            Dictionary with comprehensive confidence metrics
        """
        # Get base prediction
        base_pred = self.get_prediction(protein1, protein2)
        
        # Monte Carlo dropout
        mc_uncertainty = self.monte_carlo_dropout_uncertainty(protein1, protein2)
        
        # Temperature scaling
        temp_analysis = self.temperature_scaling_analysis(protein1, protein2)
        
        # Sequence perturbation
        perturb_analysis = self.sequence_perturbation_uncertainty(protein1, protein2)
        
        # Aggregate confidence score
        # High confidence if: low MC uncertainty, low temperature sensitivity, robust to perturbations
        confidence_components = {
            'prediction_consistency': 1.0 - mc_uncertainty['coefficient_variation'],
            'temperature_stability': 1.0 - min(temp_analysis['prediction_std'] * 2, 1.0),
            'perturbation_robustness': 1.0 if perturb_analysis['robust'] else 0.5,
            'entropy_confidence': 1.0 - min(base_pred['entropy'] / np.log(self.vocab_size), 1.0)
        }
        
        # Weighted confidence score
        weights = {'prediction_consistency': 0.3, 'temperature_stability': 0.2,
                  'perturbation_robustness': 0.3, 'entropy_confidence': 0.2}
        
        overall_confidence = sum(confidence_components[k] * weights[k] for k in confidence_components)
        
        # Determine confidence level
        if overall_confidence > 0.8:
            confidence_level = 'HIGH'
        elif overall_confidence > 0.6:
            confidence_level = 'MEDIUM'
        else:
            confidence_level = 'LOW'
        
        return {
            'prediction': base_pred['interaction_prob'],
            'confidence_score': overall_confidence,
            'confidence_level': confidence_level,
            'confidence_components': confidence_components,
            'mc_dropout': mc_uncertainty,
            'temperature_analysis': temp_analysis,
            'perturbation_analysis': perturb_analysis,
            'base_entropy': base_pred['entropy']
        }
    
    def calibration_analysis(self, protein_pairs: List[Tuple[str, str]], 
                           labels: List[int], n_bins: int = 10) -> Dict[str, Any]:
        """
        Analyze model calibration - do predicted probabilities match actual frequencies?
        
        Args:
            protein_pairs: List of protein pairs
            labels: True binary labels
            n_bins: Number of bins for calibration plot
            
        Returns:
            Dictionary with calibration metrics
        """
        predictions = []
        confidences = []
        uncertainties = []
        
        print("Computing predictions and uncertainties...")
        for i, ((p1, p2), label) in enumerate(zip(protein_pairs, labels)):
            if i % 10 == 0:
                print(f"  Processing pair {i+1}/{len(protein_pairs)}...")
            
            # Get prediction and uncertainty
            analysis = self.analyze_prediction_confidence(p1, p2)
            
            predictions.append(analysis['prediction'])
            confidences.append(analysis['confidence_score'])
            uncertainties.append(analysis['mc_dropout']['std_prediction'])
        
        predictions = np.array(predictions)
        confidences = np.array(confidences)
        uncertainties = np.array(uncertainties)
        labels = np.array(labels)
        
        # Compute calibration
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for i in range(n_bins):
            bin_mask = (predictions >= bin_boundaries[i]) & (predictions < bin_boundaries[i+1])
            if np.sum(bin_mask) > 0:
                bin_accuracy = np.mean(labels[bin_mask])
                bin_confidence = np.mean(predictions[bin_mask])
                bin_accuracies.append(bin_accuracy)
                bin_confidences.append(bin_confidence)
                bin_counts.append(np.sum(bin_mask))
            else:
                bin_accuracies.append(np.nan)
                bin_confidences.append(bin_centers[i])
                bin_counts.append(0)
        
        # Expected Calibration Error (ECE)
        ece = 0
        total_samples = len(predictions)
        for i in range(n_bins):
            if bin_counts[i] > 0:
                ece += (bin_counts[i] / total_samples) * abs(bin_accuracies[i] - bin_confidences[i])
        
        # Uncertainty quality metrics
        # Good uncertainty: high for incorrect predictions, low for correct
        correct_mask = (predictions > 0.5) == labels
        uncertainty_quality = {
            'mean_uncertainty_correct': np.mean(uncertainties[correct_mask]) if np.sum(correct_mask) > 0 else 0,
            'mean_uncertainty_incorrect': np.mean(uncertainties[~correct_mask]) if np.sum(~correct_mask) > 0 else 0,
            'uncertainty_discrimination': 0
        }
        
        if uncertainty_quality['mean_uncertainty_incorrect'] > 0:
            uncertainty_quality['uncertainty_discrimination'] = \
                uncertainty_quality['mean_uncertainty_incorrect'] / \
                (uncertainty_quality['mean_uncertainty_correct'] + 1e-10) - 1
        
        return {
            'expected_calibration_error': ece,
            'bin_accuracies': bin_accuracies,
            'bin_confidences': bin_confidences,
            'bin_counts': bin_counts,
            'uncertainty_quality': uncertainty_quality,
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences)
        }
    
    def visualize_uncertainty_analysis(self, protein_pairs: List[Tuple[str, str]], 
                                     labels: List[int], output_dir: str):
        """
        Create comprehensive uncertainty visualizations.
        
        Args:
            protein_pairs: List of protein pairs
            labels: True binary labels
            output_dir: Directory to save visualizations
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run calibration analysis
        calibration = self.calibration_analysis(protein_pairs, labels)
        
        # 1. Calibration plot
        plt.figure(figsize=(8, 8))
        
        # Perfect calibration line
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        
        # Actual calibration
        valid_bins = ~np.isnan(calibration['bin_accuracies'])
        plt.plot(np.array(calibration['bin_confidences'])[valid_bins],
                np.array(calibration['bin_accuracies'])[valid_bins],
                'bo-', linewidth=2, markersize=8, label='Model calibration')
        
        # Add bin counts as text
        for i, (conf, acc, count) in enumerate(zip(calibration['bin_confidences'],
                                                   calibration['bin_accuracies'],
                                                   calibration['bin_counts'])):
            if count > 0 and not np.isnan(acc):
                plt.text(conf, acc + 0.02, str(count), ha='center', fontsize=8)
        
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title(f'Calibration Plot\nECE = {calibration["expected_calibration_error"]:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'calibration_plot.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'calibration_plot.svg', bbox_inches='tight')
        plt.close()
        
        # 2. Uncertainty distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Collect detailed uncertainty data for visualization
        mc_uncertainties = []
        predictions = []
        confidence_scores = []
        
        print("\nCollecting uncertainty data for visualization...")
        for i, ((p1, p2), label) in enumerate(zip(protein_pairs[:20], labels[:20])):  # Limit for speed
            analysis = self.analyze_prediction_confidence(p1, p2)
            mc_uncertainties.append(analysis['mc_dropout']['std_prediction'])
            predictions.append(analysis['prediction'])
            confidence_scores.append(analysis['confidence_score'])
        
        # Uncertainty vs prediction
        ax1.scatter(predictions, mc_uncertainties, alpha=0.6)
        ax1.set_xlabel('Predicted Probability')
        ax1.set_ylabel('Prediction Uncertainty (MC Dropout Std)')
        ax1.set_title('Uncertainty vs Prediction')
        ax1.grid(True, alpha=0.3)
        
        # Confidence score distribution
        ax2.hist(confidence_scores, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of Confidence Scores')
        ax2.axvline(np.mean(confidence_scores), color='red', linestyle='--',
                   label=f'Mean: {np.mean(confidence_scores):.2f}')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'uncertainty_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'uncertainty_analysis.svg', bbox_inches='tight')
        plt.close()
        
        # 3. Example uncertainty decomposition
        if protein_pairs:
            p1, p2 = protein_pairs[0]
            analysis = self.analyze_prediction_confidence(p1, p2)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            components = list(analysis['confidence_components'].keys())
            values = list(analysis['confidence_components'].values())
            
            bars = ax.bar(components, values, alpha=0.7)
            ax.set_ylabel('Component Score')
            ax.set_title(f'Confidence Components Breakdown\nOverall Confidence: {analysis["confidence_score"]:.3f} ({analysis["confidence_level"]})')
            ax.set_ylim(0, 1.1)
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.2f}', ha='center', va='bottom')
            
            # Rotate x labels
            plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'confidence_breakdown.png', dpi=300, bbox_inches='tight')
            plt.savefig(output_dir / 'confidence_breakdown.svg', bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to {output_dir}")


def main():
    """Run uncertainty quantification analysis."""
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
    
    # Initialize quantifier
    quantifier = UncertaintyQuantifier(model, meta_path)
    
    # Load protein pairs
    protein_pairs = []
    labels = []
    
    # Load interacting pairs
    int_file = 'ppiGPT_MED4_solo/MED4_Int_100pairs_prompts.txt'
    with open(int_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines[:10]):
            parts = line.strip().split(',')
            if len(parts) >= 5 and parts[0] == '<ps1>' and parts[2] == '<ps2>':
                protein1 = parts[1]
                protein2 = parts[3]
                protein_pairs.append((protein1, protein2))
                labels.append(1)
    
    # Load non-interacting pairs
    rnd_file = 'ppiGPT_MED4_solo/MED4_100_RND_prompts.txt'
    with open(rnd_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines[:10]):
            parts = line.strip().split(',')
            if len(parts) >= 5 and parts[0] == '<ps1>' and parts[2] == '<ps2>':
                protein1 = parts[1]
                protein2 = parts[3]
                protein_pairs.append((protein1, protein2))
                labels.append(0)
    
    print(f"\nLoaded {len(protein_pairs)} protein pairs ({sum(labels)} interacting, {len(labels)-sum(labels)} non-interacting)")
    
    # 1. Analyze individual example
    if protein_pairs:
        print("\n1. Detailed uncertainty analysis for first protein pair...")
        p1, p2 = protein_pairs[0]
        label = labels[0]
        
        confidence_analysis = quantifier.analyze_prediction_confidence(p1, p2)
        
        print(f"\nPrediction: {confidence_analysis['prediction']:.3f}")
        print(f"True label: {'Interacting' if label == 1 else 'Non-interacting'}")
        print(f"Confidence level: {confidence_analysis['confidence_level']}")
        print(f"Confidence score: {confidence_analysis['confidence_score']:.3f}")
        
        print("\nConfidence components:")
        for component, score in confidence_analysis['confidence_components'].items():
            print(f"  {component}: {score:.3f}")
        
        print("\nMonte Carlo Dropout analysis:")
        mc = confidence_analysis['mc_dropout']
        print(f"  Mean prediction: {mc['mean_prediction']:.3f} ± {mc['std_prediction']:.3f}")
        print(f"  Epistemic uncertainty: {mc['epistemic_uncertainty']:.3f}")
        print(f"  Aleatoric uncertainty: {mc['aleatoric_uncertainty']:.3f}")
    
    # 2. Calibration and visualization
    print("\n2. Running calibration analysis...")
    output_dir = Path("uncertainty_analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    quantifier.visualize_uncertainty_analysis(protein_pairs, labels, output_dir)
    
    # 3. Save detailed results
    results_data = []
    print("\n3. Analyzing all protein pairs...")
    
    for i, ((p1, p2), label) in enumerate(zip(protein_pairs, labels)):
        if i % 5 == 0:
            print(f"  Processing pair {i+1}/{len(protein_pairs)}...")
        
        analysis = quantifier.analyze_prediction_confidence(p1, p2)
        
        results_data.append({
            'pair_index': i,
            'true_label': label,
            'prediction': analysis['prediction'],
            'confidence_score': analysis['confidence_score'],
            'confidence_level': analysis['confidence_level'],
            'mc_std': analysis['mc_dropout']['std_prediction'],
            'epistemic_uncertainty': analysis['mc_dropout']['epistemic_uncertainty'],
            'aleatoric_uncertainty': analysis['mc_dropout']['aleatoric_uncertainty'],
            'temperature_sensitive': analysis['temperature_analysis']['temperature_sensitive'],
            'perturbation_robust': analysis['perturbation_analysis']['robust']
        })
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(output_dir / 'uncertainty_analysis_results.csv', index=False)
    
    print("\nSummary statistics:")
    print(f"Mean confidence score: {results_df['confidence_score'].mean():.3f}")
    print(f"Predictions with HIGH confidence: {(results_df['confidence_level'] == 'HIGH').sum()}")
    print(f"Predictions with LOW confidence: {(results_df['confidence_level'] == 'LOW').sum()}")
    
    print(f"\nAnalysis complete! Results saved to {output_dir}/")


if __name__ == "__main__":
    main()