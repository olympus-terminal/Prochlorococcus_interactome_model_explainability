#!/usr/bin/env python3
"""
Comprehensive DeepLift Analysis for PPI Model using Captum
==========================================================

This script performs proper DeepLift attribution analysis on the GPT-based PPI model
to identify important residues and motifs for protein-protein interactions.

Key features:
- Uses Captum's DeepLift implementation correctly with proper nn.Module wrapper
- Processes all 1084 sequences from real and random PPI datasets
- Performs motif discovery on DeepLift attributions
- Creates comprehensive visualizations as SVG vector graphics
- Includes detailed logging and error handling
- All outputs include "deeplift_motif" in filenames for clear identification

Author: PPI Analysis Pipeline
Date: 2025-06-29
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
import re
from tqdm import tqdm

# Captum imports
from captum.attr import DeepLift, visualization as viz
from captum.attr._utils.visualization import visualize_image_attr

# Local imports
from model import GPTConfig, GPT

# Configure matplotlib for better SVG output
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150


class PPIModelWrapper(nn.Module):
    """
    Proper nn.Module wrapper for the GPT model that returns only the interaction logit.
    This wrapper works with embeddings instead of token IDs to support gradient computation.
    """
    
    def __init__(self, model: GPT, token_1_id: int):
        """
        Initialize the wrapper.
        
        Args:
            model: The GPT model
            token_1_id: Token ID for interaction class ("1")
        """
        super().__init__()
        self.model = model
        self.token_1_id = token_1_id
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass from embeddings that returns only the interaction logit.
        
        Args:
            embeddings: Input embeddings [batch_size, seq_len, embed_dim]
            
        Returns:
            Interaction logit at the last position
        """
        # Add positional embeddings
        seq_len = embeddings.size(1)
        pos = torch.arange(0, seq_len, dtype=torch.long, device=embeddings.device)
        pos_emb = self.model.transformer.wpe(pos)
        x = self.model.transformer.drop(embeddings + pos_emb)
        
        # Forward through transformer blocks
        for block in self.model.transformer.h:
            x = block(x)
        
        x = self.model.transformer.ln_f(x)
        logits = self.model.lm_head(x)
        
        # Return only the interaction logit at the last position
        return logits[:, -1, self.token_1_id]


class DeepLiftPPIAnalyzer:
    """
    DeepLift attribution analysis for PPI predictions using Captum.
    """
    
    def __init__(self, model_path: str, meta_path: str, device: str = 'cuda'):
        """
        Initialize the DeepLift analyzer.
        
        Args:
            model_path: Path to model checkpoint
            meta_path: Path to tokenizer metadata
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Setup logging
        self._setup_logging()
        
        # Load model
        self.base_model = self._load_model(model_path)
        self.base_model.eval()
        
        # Load tokenizer
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        self.stoi = meta['stoi']
        self.itos = meta['itos']
        self.vocab_size = meta['vocab_size']
        
        # Create encoding/decoding functions
        self.encode = lambda s: [self.stoi.get(c, self.stoi.get('A', 7)) for c in s]
        self.decode = lambda l: ''.join([self.itos.get(i, 'X') for i in l])
        
        # Get special token IDs
        self.token_1_id = self.stoi.get('1', 3)
        self.token_0_id = self.stoi.get('0', 4)
        
        # Create wrapped model for DeepLift
        self.wrapped_model = PPIModelWrapper(self.base_model, self.token_1_id)
        self.wrapped_model.eval()
        
        # Create DeepLift explainer
        self._setup_deeplift()
        
        self.logger.info(f"Initialized DeepLift analyzer with vocab size: {self.vocab_size}")
        self.logger.info(f"Model device: {self.device}")
        self.logger.info(f"Special tokens: '1'={self.token_1_id}, '0'={self.token_0_id}")
    
    def _setup_logging(self):
        """Setup comprehensive logging."""
        log_dir = Path(f'deeplift_motif_analysis_{self.timestamp}')
        log_dir.mkdir(exist_ok=True)
        self.log_dir = log_dir
        
        # Configure logging
        log_file = log_dir / 'deeplift_motif_analysis.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('DeepLiftAnalyzer')
        self.logger.info("="*80)
        self.logger.info("DeepLift Motif PPI Analysis Started")
        self.logger.info("="*80)
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load the GPT model from checkpoint."""
        self.logger.info(f"Loading model from {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        config = GPTConfig(**checkpoint['model_args'])
        model = GPT(config)
        
        # Load state dict
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        
        model.load_state_dict(state_dict)
        model.to(self.device)
        
        self.logger.info(f"Model loaded successfully. Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model
    
    def _setup_deeplift(self):
        """Setup DeepLift explainer with proper configuration."""
        # Create DeepLift explainer with the wrapped model
        self.deeplift = DeepLift(self.wrapped_model)
        self.logger.info("DeepLift explainer initialized with proper nn.Module wrapper")
    
    def prepare_input(self, protein1: str, protein2: str) -> Tuple[torch.Tensor, str]:
        """
        Prepare input for the model.
        
        Args:
            protein1: First protein sequence
            protein2: Second protein sequence
            
        Returns:
            Tuple of (input_ids tensor, formatted string)
        """
        # Format input
        input_text = f"<ps1>,{protein1},<ps2>,{protein2},<"
        input_ids = self.encode(input_text)
        
        # Truncate if necessary
        max_length = min(len(input_ids), self.base_model.config.block_size)
        if len(input_ids) > max_length:
            input_ids = input_ids[-max_length:]
            self.logger.warning(f"Input truncated from {len(input_ids)} to {max_length} tokens")
        
        # Convert to tensor
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        
        return input_tensor, input_text
    
    def compute_deeplift_attributions(self, protein1: str, protein2: str) -> Dict[str, Any]:
        """
        Compute DeepLift attributions for a protein pair.
        
        Args:
            protein1: First protein sequence
            protein2: Second protein sequence
            
        Returns:
            Dictionary containing attributions and analysis
        """
        # Prepare input
        input_ids, input_text = self.prepare_input(protein1, protein2)
        
        # Get embeddings
        with torch.no_grad():
            embeddings = self.base_model.transformer.wte(input_ids)
            baseline_embeddings = self.base_model.transformer.wte(torch.zeros_like(input_ids))
        
        # Compute attributions using DeepLift
        attributions = self.deeplift.attribute(
            inputs=embeddings,
            baselines=baseline_embeddings,
            return_convergence_delta=True
        )
        
        if isinstance(attributions, tuple):
            attributions, convergence_delta = attributions
            self.logger.debug(f"Convergence delta: {convergence_delta.item():.6f}")
        
        # Compute attribution magnitudes (L2 norm across embedding dimension)
        attribution_magnitudes = torch.norm(attributions, dim=2, p=2).squeeze(0).detach().cpu().numpy()
        
        # Get model prediction
        with torch.no_grad():
            logits, _ = self.base_model(input_ids)
            probs = F.softmax(logits[0, -1, :], dim=0)
            interaction_prob = probs[self.token_1_id].item()
            non_interaction_prob = probs[self.token_0_id].item()
        
        # Parse sequence positions
        decoded = self.decode(input_ids.squeeze(0).tolist())
        seq_info = self._parse_sequence_positions(decoded, protein1, protein2)
        
        return {
            'input_ids': input_ids.squeeze(0).tolist(),
            'attributions': attribution_magnitudes,
            'interaction_prob': interaction_prob,
            'non_interaction_prob': non_interaction_prob,
            'sequence_info': seq_info,
            'decoded_input': decoded,
            'protein1': protein1,
            'protein2': protein2
        }
    
    def _parse_sequence_positions(self, decoded_input: str, protein1: str, protein2: str) -> Dict:
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
        else:
            # Fallback
            protein1_start = 6
            protein1_end = protein1_start + len(protein1)
            protein2_start = protein1_end + 6
            protein2_end = protein2_start + len(protein2)
        
        return {
            'protein1_range': (protein1_start, protein1_end),
            'protein2_range': (protein2_start, protein2_end),
            'total_length': len(decoded_input)
        }
    
    def analyze_batch(self, protein_pairs: List[Tuple[str, str]], 
                     dataset_name: str) -> Dict[str, Any]:
        """
        Analyze a batch of protein pairs.
        
        Args:
            protein_pairs: List of (protein1, protein2) tuples
            dataset_name: Name of the dataset (e.g., 'real', 'random')
            
        Returns:
            Dictionary containing batch analysis results
        """
        self.logger.info(f"Analyzing {len(protein_pairs)} pairs from {dataset_name} dataset")
        
        results = []
        all_attributions = []
        protein1_attributions = []
        protein2_attributions = []
        
        for i, (p1, p2) in enumerate(tqdm(protein_pairs, desc=f"Processing {dataset_name}")):
            try:
                # Compute attributions
                result = self.compute_deeplift_attributions(p1, p2)
                results.append(result)
                
                # Extract protein-specific attributions
                seq_info = result['sequence_info']
                attributions = result['attributions']
                
                p1_start, p1_end = seq_info['protein1_range']
                p2_start, p2_end = seq_info['protein2_range']
                
                if p1_start < p1_end and p1_end <= len(attributions):
                    p1_attrs = attributions[p1_start:p1_end]
                    protein1_attributions.extend(zip(p1, p1_attrs))
                
                if p2_start < p2_end and p2_end <= len(attributions):
                    p2_attrs = attributions[p2_start:p2_end]
                    protein2_attributions.extend(zip(p2, p2_attrs))
                
                if len(attributions) > 0:
                    all_attributions.extend(attributions)
                
            except Exception as e:
                self.logger.error(f"Error processing pair {i}: {e}")
                continue
        
        # Aggregate statistics
        interaction_probs = [r['interaction_prob'] for r in results] if results else []
        
        return {
            'dataset_name': dataset_name,
            'num_pairs': len(protein_pairs),
            'results': results,
            'all_attributions': np.array(all_attributions) if all_attributions else np.array([]),
            'protein1_attributions': protein1_attributions,
            'protein2_attributions': protein2_attributions,
            'mean_interaction_prob': np.mean(interaction_probs) if interaction_probs else 0.0,
            'std_interaction_prob': np.std(interaction_probs) if interaction_probs else 0.0
        }
    
    def discover_motifs(self, batch_results: Dict[str, Any], 
                       window_size: int = 5, 
                       top_k: int = 20) -> Dict[str, Any]:
        """
        Discover important motifs based on DeepLift attributions.
        
        Args:
            batch_results: Results from analyze_batch
            window_size: Size of motif window
            top_k: Number of top motifs to return
            
        Returns:
            Dictionary containing discovered motifs
        """
        self.logger.info(f"Discovering motifs with window size {window_size}")
        
        motif_scores = defaultdict(list)
        motif_contexts = defaultdict(list)
        
        # Process each result
        for result in batch_results['results']:
            attributions = result['attributions']
            decoded = result['decoded_input']
            
            # Scan for high-attribution regions
            for i in range(len(decoded) - window_size + 1):
                motif = decoded[i:i + window_size]
                
                # Skip motifs with special tokens
                if any(token in motif for token in ['<', '>', ',']):
                    continue
                
                # Calculate average attribution for this window
                window_attrs = attributions[i:i + window_size]
                if len(window_attrs) == window_size:
                    avg_score = np.mean(window_attrs)
                    motif_scores[motif].append(avg_score)
                    
                    # Store context
                    context_start = max(0, i - 2)
                    context_end = min(len(decoded), i + window_size + 2)
                    context = decoded[context_start:context_end]
                    motif_contexts[motif].append(context)
        
        # Aggregate motif scores
        motif_stats = []
        for motif, scores in motif_scores.items():
            stats = {
                'motif': motif,
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'max_score': np.max(scores),
                'count': len(scores),
                'contexts': motif_contexts[motif][:5]  # Keep top 5 contexts
            }
            motif_stats.append(stats)
        
        # Sort by mean score
        motif_stats.sort(key=lambda x: x['mean_score'], reverse=True)
        
        return {
            'top_motifs': motif_stats[:top_k],
            'all_motifs': motif_stats,
            'window_size': window_size
        }
    
    def analyze_position_importance(self, batch_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze position-specific importance across all sequences.
        
        Args:
            batch_results: Results from analyze_batch
            
        Returns:
            Dictionary containing position-specific analysis
        """
        self.logger.info("Analyzing position-specific importance")
        
        # Collect position-specific data
        position_scores_p1 = defaultdict(list)
        position_scores_p2 = defaultdict(list)
        aa_position_scores = defaultdict(lambda: defaultdict(list))
        
        for result in batch_results['results']:
            seq_info = result['sequence_info']
            attributions = result['attributions']
            decoded = result['decoded_input']
            
            # Protein 1 analysis
            p1_start, p1_end = seq_info['protein1_range']
            if p1_start < p1_end and p1_end <= len(attributions):
                p1_seq = result['protein1']
                p1_attrs = attributions[p1_start:p1_end]
                
                for pos, (aa, attr) in enumerate(zip(p1_seq, p1_attrs)):
                    position_scores_p1[pos].append(attr)
                    aa_position_scores[pos][aa].append(attr)
            
            # Protein 2 analysis
            p2_start, p2_end = seq_info['protein2_range']
            if p2_start < p2_end and p2_end <= len(attributions):
                p2_seq = result['protein2']
                p2_attrs = attributions[p2_start:p2_end]
                
                for pos, (aa, attr) in enumerate(zip(p2_seq, p2_attrs)):
                    position_scores_p2[pos].append(attr)
        
        # Compute statistics
        position_stats = {
            'protein1': {},
            'protein2': {},
            'aa_preferences': {}
        }
        
        # Protein 1 position statistics
        for pos, scores in position_scores_p1.items():
            position_stats['protein1'][pos] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'max': np.max(scores),
                'count': len(scores)
            }
        
        # Protein 2 position statistics
        for pos, scores in position_scores_p2.items():
            position_stats['protein2'][pos] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'max': np.max(scores),
                'count': len(scores)
            }
        
        # Amino acid preferences by position
        for pos, aa_scores in aa_position_scores.items():
            position_stats['aa_preferences'][pos] = {}
            for aa, scores in aa_scores.items():
                position_stats['aa_preferences'][pos][aa] = {
                    'mean': np.mean(scores),
                    'count': len(scores)
                }
        
        return position_stats
    
    def create_visualizations(self, batch_results: Dict[str, Any], 
                            motif_results: Dict[str, Any],
                            position_results: Dict[str, Any]):
        """
        Create comprehensive visualizations of DeepLift results.
        
        Args:
            batch_results: Results from analyze_batch
            motif_results: Results from discover_motifs
            position_results: Results from analyze_position_importance
        """
        dataset_name = batch_results['dataset_name']
        self.logger.info(f"Creating visualizations for {dataset_name} dataset")
        
        # Create figure directory
        fig_dir = self.log_dir / f'{dataset_name}_figures'
        fig_dir.mkdir(exist_ok=True)
        
        # 1. Attribution distribution
        self._plot_attribution_distribution(batch_results, fig_dir)
        
        # 2. Top motifs visualization
        self._plot_top_motifs(motif_results, fig_dir, dataset_name)
        
        # 3. Position importance heatmap
        self._plot_position_importance(position_results, fig_dir, dataset_name)
        
        # 4. Amino acid preferences
        self._plot_aa_preferences(position_results, fig_dir, dataset_name)
        
        # 5. Attribution heatmap for top examples
        self._plot_attribution_heatmap(batch_results, fig_dir, dataset_name)
        
        # 6. Sequence logos for high-attribution regions
        self._create_sequence_logos(batch_results, fig_dir, dataset_name)
        
        self.logger.info(f"Visualizations saved to {fig_dir}")
    
    def _plot_attribution_distribution(self, batch_results: Dict, fig_dir: Path):
        """Plot distribution of attribution scores."""
        plt.figure(figsize=(10, 6))
        
        all_attrs = batch_results['all_attributions']
        
        # Plot histogram
        if len(all_attrs) > 0:
            plt.hist(all_attrs, bins=50, alpha=0.7, color='blue', edgecolor='black')
            plt.axvline(np.mean(all_attrs), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(all_attrs):.3f}')
            plt.axvline(np.percentile(all_attrs, 95), color='green', linestyle='--',
                       label=f'95th percentile: {np.percentile(all_attrs, 95):.3f}')
        else:
            self.logger.warning("No attribution data available for distribution plot")
        
        plt.xlabel('Attribution Score')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of DeepLift Attribution Scores - {batch_results["dataset_name"]}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save with deeplift_motif prefix
        save_path = fig_dir / f'deeplift_motif_{batch_results["dataset_name"]}_attribution_distribution_{self.timestamp}.svg'
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        plt.close()
    
    def _plot_top_motifs(self, motif_results: Dict, fig_dir: Path, dataset_name: str):
        """Plot top motifs by attribution score."""
        top_motifs = motif_results['top_motifs'][:15]  # Show top 15
        
        if not top_motifs:
            self.logger.warning("No motifs found to plot")
            return
        
        plt.figure(figsize=(12, 8))
        
        motifs = [m['motif'] for m in top_motifs]
        scores = [m['mean_score'] for m in top_motifs]
        counts = [m['count'] for m in top_motifs]
        
        # Create bar plot
        bars = plt.bar(range(len(motifs)), scores, color='darkblue', alpha=0.7)
        
        # Add count annotations
        for i, (bar, count) in enumerate(zip(bars, counts)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'n={count}', ha='center', va='bottom', fontsize=8)
        
        plt.xlabel('Motif')
        plt.ylabel('Mean Attribution Score')
        plt.title(f'Top {len(motifs)} Motifs by DeepLift Attribution - {dataset_name}')
        plt.xticks(range(len(motifs)), motifs, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Save with deeplift_motif prefix
        save_path = fig_dir / f'deeplift_motif_{dataset_name}_top_motifs_{self.timestamp}.svg'
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        plt.close()
    
    def _plot_position_importance(self, position_results: Dict, fig_dir: Path, dataset_name: str):
        """Plot position-specific importance."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Protein 1
        if position_results['protein1']:
            positions = sorted(position_results['protein1'].keys())
            means = [position_results['protein1'][p]['mean'] for p in positions]
            stds = [position_results['protein1'][p]['std'] for p in positions]
            
            ax1.bar(positions, means, yerr=stds, capsize=5, color='blue', alpha=0.7)
            ax1.set_xlabel('Position')
            ax1.set_ylabel('Mean Attribution Score')
            ax1.set_title(f'Position Importance - Protein 1 ({dataset_name})')
            ax1.grid(True, alpha=0.3, axis='y')
        
        # Protein 2
        if position_results['protein2']:
            positions = sorted(position_results['protein2'].keys())
            means = [position_results['protein2'][p]['mean'] for p in positions]
            stds = [position_results['protein2'][p]['std'] for p in positions]
            
            ax2.bar(positions, means, yerr=stds, capsize=5, color='red', alpha=0.7)
            ax2.set_xlabel('Position')
            ax2.set_ylabel('Mean Attribution Score')
            ax2.set_title(f'Position Importance - Protein 2 ({dataset_name})')
            ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save with deeplift_motif prefix
        save_path = fig_dir / f'deeplift_motif_{dataset_name}_position_importance_{self.timestamp}.svg'
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        plt.close()
    
    def _plot_aa_preferences(self, position_results: Dict, fig_dir: Path, dataset_name: str):
        """Plot amino acid preferences at high-attribution positions."""
        # Find top positions by mean attribution
        position_scores = {}
        for pos, stats in position_results['protein1'].items():
            position_scores[pos] = stats['mean']
        
        if not position_scores:
            self.logger.warning("No position scores found for AA preferences")
            return
        
        # Get top 10 positions
        top_positions = sorted(position_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Create heatmap data
        amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                      'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        
        heatmap_data = []
        position_labels = []
        
        for pos, score in top_positions:
            position_labels.append(f'Pos {pos}')
            row = []
            for aa in amino_acids:
                if pos in position_results['aa_preferences'] and aa in position_results['aa_preferences'][pos]:
                    row.append(position_results['aa_preferences'][pos][aa]['mean'])
                else:
                    row.append(0)
            heatmap_data.append(row)
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, xticklabels=amino_acids, yticklabels=position_labels,
                   cmap='RdBu_r', center=0, cbar_kws={'label': 'Mean Attribution Score'})
        
        plt.title(f'Amino Acid Preferences at High-Attribution Positions - {dataset_name}')
        plt.xlabel('Amino Acid')
        plt.ylabel('Position')
        
        # Save with deeplift_motif prefix
        save_path = fig_dir / f'deeplift_motif_{dataset_name}_aa_preferences_{self.timestamp}.svg'
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        plt.close()
    
    def _plot_attribution_heatmap(self, batch_results: Dict, fig_dir: Path, dataset_name: str):
        """Create attribution heatmap for top examples."""
        # Get top 10 examples by interaction probability
        results = batch_results['results']
        sorted_results = sorted(results, key=lambda x: x['interaction_prob'], reverse=True)[:10]
        
        # Create heatmap data
        heatmap_data = []
        labels = []
        
        for i, result in enumerate(sorted_results):
            attrs = result['attributions']
            # Normalize to fixed length for visualization
            if len(attrs) > 200:
                attrs = attrs[:200]
            elif len(attrs) < 200:
                attrs = np.pad(attrs, (0, 200 - len(attrs)), constant_values=0)
            
            heatmap_data.append(attrs)
            labels.append(f"Pair {i+1} (p={result['interaction_prob']:.3f})")
        
        # Create heatmap
        plt.figure(figsize=(20, 8))
        sns.heatmap(heatmap_data, cmap='hot', cbar_kws={'label': 'Attribution Score'},
                   yticklabels=labels)
        
        plt.title(f'Attribution Heatmap for Top 10 Interactions - {dataset_name}')
        plt.xlabel('Sequence Position')
        plt.ylabel('Protein Pair')
        
        # Save with deeplift_motif prefix
        save_path = fig_dir / f'deeplift_motif_{dataset_name}_attribution_heatmap_{self.timestamp}.svg'
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        plt.close()
    
    def _create_sequence_logos(self, batch_results: Dict, fig_dir: Path, dataset_name: str):
        """Create sequence logos for high-attribution regions."""
        from matplotlib import patches
        import matplotlib.patches as mpatches
        
        # Collect high-attribution sequences
        high_attr_sequences = defaultdict(list)
        threshold = np.percentile(batch_results['all_attributions'], 90)
        
        for result in batch_results['results']:
            attributions = result['attributions']
            decoded = result['decoded_input']
            seq_info = result['sequence_info']
            
            # Find high-attribution regions
            for protein_idx, (start, end) in enumerate([seq_info['protein1_range'], 
                                                       seq_info['protein2_range']]):
                if start < end and end <= len(attributions):
                    protein_attrs = attributions[start:end]
                    protein_seq = result[f'protein{protein_idx+1}']
                    
                    # Find positions above threshold
                    high_positions = np.where(protein_attrs > threshold)[0]
                    
                    # Extract windows around high positions
                    for pos in high_positions:
                        window_start = max(0, pos - 2)
                        window_end = min(len(protein_seq), pos + 3)
                        window_seq = protein_seq[window_start:window_end]
                        
                        if len(window_seq) == 5:  # Full window
                            high_attr_sequences[f'protein{protein_idx+1}'].append(window_seq)
        
        # Create sequence logos
        for protein_type, sequences in high_attr_sequences.items():
            if not sequences:
                continue
            
            # Count amino acids at each position
            position_counts = defaultdict(lambda: defaultdict(int))
            for seq in sequences:
                for pos, aa in enumerate(seq):
                    position_counts[pos][aa] += 1
            
            # Create logo plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Calculate frequencies and create logo
            num_sequences = len(sequences)
            x_positions = list(range(5))
            
            for pos in x_positions:
                counts = position_counts[pos]
                total = sum(counts.values())
                
                if total == 0:
                    continue
                
                # Sort amino acids by frequency
                sorted_aas = sorted(counts.items(), key=lambda x: x[1], reverse=True)
                
                y_offset = 0
                for aa, count in sorted_aas:
                    height = count / total
                    ax.text(pos, y_offset + height/2, aa, 
                           fontsize=40 * height, ha='center', va='center',
                           weight='bold')
                    y_offset += height
            
            ax.set_xlim(-0.5, 4.5)
            ax.set_ylim(0, 1)
            ax.set_xlabel('Position')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Sequence Logo - High Attribution Regions ({protein_type}, {dataset_name})')
            ax.set_xticks(x_positions)
            ax.set_xticklabels([f'{i-2:+d}' for i in range(5)])
            
            # Save with deeplift_motif prefix
            save_path = fig_dir / f'deeplift_motif_{dataset_name}_logo_{protein_type}_{self.timestamp}.svg'
            plt.savefig(save_path, format='svg', bbox_inches='tight')
            plt.close()
    
    def save_results(self, all_results: Dict[str, Any]):
        """Save all analysis results."""
        # Save as pickle with deeplift_motif prefix
        results_file = self.log_dir / 'deeplift_motif_analysis_results.pkl'
        with open(results_file, 'wb') as f:
            pickle.dump(all_results, f)
        
        # Save summary statistics as text
        summary_file = self.log_dir / 'deeplift_motif_analysis_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("DeepLift Motif PPI Analysis Summary\n")
            f.write("=" * 80 + "\n\n")
            
            for dataset_name, results in all_results.items():
                if dataset_name == 'timestamp':
                    continue
                
                f.write(f"\nDataset: {dataset_name}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Number of pairs: {results['batch_results']['num_pairs']}\n")
                f.write(f"Mean interaction probability: {results['batch_results']['mean_interaction_prob']:.4f}\n")
                f.write(f"Std interaction probability: {results['batch_results']['std_interaction_prob']:.4f}\n")
                
                # Top motifs
                f.write("\nTop 10 Motifs:\n")
                for i, motif in enumerate(results['motif_results']['top_motifs'][:10]):
                    f.write(f"{i+1}. {motif['motif']} - Score: {motif['mean_score']:.4f} (n={motif['count']})\n")
                
                # Important positions
                f.write("\nTop 10 Important Positions (Protein 1):\n")
                pos_scores = [(pos, stats['mean']) for pos, stats in 
                             results['position_results']['protein1'].items()]
                pos_scores.sort(key=lambda x: x[1], reverse=True)
                for i, (pos, score) in enumerate(pos_scores[:10]):
                    f.write(f"{i+1}. Position {pos}: {score:.4f}\n")
        
        self.logger.info(f"Results saved to {self.log_dir}")
    
    def run_full_analysis(self, real_ppis_file: str, random_ppis_file: str, 
                         max_pairs: Optional[int] = None):
        """
        Run complete DeepLift analysis on both real and random PPI datasets.
        
        Args:
            real_ppis_file: Path to real PPIs CSV file
            random_ppis_file: Path to random PPIs CSV file
            max_pairs: Maximum number of pairs to analyze (None for all)
        """
        self.logger.info("Starting full DeepLift analysis")
        all_results = {'timestamp': self.timestamp}
        
        # Load datasets
        real_df = pd.read_csv(real_ppis_file)
        random_df = pd.read_csv(random_ppis_file)
        
        if max_pairs:
            real_df = real_df.head(max_pairs)
            random_df = random_df.head(max_pairs)
        
        self.logger.info(f"Loaded {len(real_df)} real PPIs and {len(random_df)} random PPIs")
        
        # Process real PPIs
        real_pairs = [(row['protein1'], row['protein2']) for _, row in real_df.iterrows()]
        real_batch_results = self.analyze_batch(real_pairs, 'real_ppis')
        real_motif_results = self.discover_motifs(real_batch_results)
        real_position_results = self.analyze_position_importance(real_batch_results)
        
        # Create visualizations for real PPIs
        self.create_visualizations(real_batch_results, real_motif_results, real_position_results)
        
        all_results['real_ppis'] = {
            'batch_results': real_batch_results,
            'motif_results': real_motif_results,
            'position_results': real_position_results
        }
        
        # Process random PPIs
        random_pairs = [(row['protein1'], row['protein2']) for _, row in random_df.iterrows()]
        random_batch_results = self.analyze_batch(random_pairs, 'random_ppis')
        random_motif_results = self.discover_motifs(random_batch_results)
        random_position_results = self.analyze_position_importance(random_batch_results)
        
        # Create visualizations for random PPIs
        self.create_visualizations(random_batch_results, random_motif_results, 
                                 random_position_results)
        
        all_results['random_ppis'] = {
            'batch_results': random_batch_results,
            'motif_results': random_motif_results,
            'position_results': random_position_results
        }
        
        # Save all results
        self.save_results(all_results)
        
        # Create comparison visualizations
        self._create_comparison_plots(all_results)
        
        self.logger.info("Analysis complete!")
        return all_results
    
    def _create_comparison_plots(self, all_results: Dict):
        """Create plots comparing real vs random datasets."""
        fig_dir = self.log_dir / 'comparisons'
        fig_dir.mkdir(exist_ok=True)
        
        # 1. Compare attribution distributions
        plt.figure(figsize=(12, 6))
        
        real_attrs = all_results['real_ppis']['batch_results']['all_attributions']
        random_attrs = all_results['random_ppis']['batch_results']['all_attributions']
        
        plt.hist(real_attrs, bins=50, alpha=0.5, label='Real PPIs', color='blue', density=True)
        plt.hist(random_attrs, bins=50, alpha=0.5, label='Random PPIs', color='red', density=True)
        
        plt.xlabel('Attribution Score')
        plt.ylabel('Density')
        plt.title('DeepLift Attribution Distribution: Real vs Random PPIs')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = fig_dir / f'deeplift_motif_attribution_comparison_{self.timestamp}.svg'
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        plt.close()
        
        # 2. Compare top motifs
        real_motifs = all_results['real_ppis']['motif_results']['top_motifs'][:10]
        random_motifs = all_results['random_ppis']['motif_results']['top_motifs'][:10]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Real motifs
        if real_motifs:
            motifs = [m['motif'] for m in real_motifs]
            scores = [m['mean_score'] for m in real_motifs]
            ax1.barh(range(len(motifs)), scores, color='blue', alpha=0.7)
            ax1.set_yticks(range(len(motifs)))
            ax1.set_yticklabels(motifs)
            ax1.set_xlabel('Mean Attribution Score')
            ax1.set_title('Top Motifs - Real PPIs')
            ax1.grid(True, alpha=0.3, axis='x')
        
        # Random motifs
        if random_motifs:
            motifs = [m['motif'] for m in random_motifs]
            scores = [m['mean_score'] for m in random_motifs]
            ax2.barh(range(len(motifs)), scores, color='red', alpha=0.7)
            ax2.set_yticks(range(len(motifs)))
            ax2.set_yticklabels(motifs)
            ax2.set_xlabel('Mean Attribution Score')
            ax2.set_title('Top Motifs - Random PPIs')
            ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        save_path = fig_dir / f'deeplift_motif_comparison_{self.timestamp}.svg'
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Comparison plots saved to {fig_dir}")


def main():
    """Main execution function."""
    # Paths
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model', 'out_3e', 'ckpt.pt')
    meta_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model', 'data', 'meta.pkl')
    real_ppis_file = 'formatted_real_PPIs.csv'
    random_ppis_file = 'formatted_random_PPIs.csv'
    
    # Initialize analyzer
    analyzer = DeepLiftPPIAnalyzer(model_path, meta_path)
    
    # Run full analysis
    results = analyzer.run_full_analysis(real_ppis_file, random_ppis_file)
    
    print(f"\nAnalysis complete! Results saved to: {analyzer.log_dir}")
    print("\nCreated files:")
    for file in sorted(analyzer.log_dir.glob('**/*')):
        if file.is_file():
            print(f"  - {file.relative_to(analyzer.log_dir)}")


if __name__ == "__main__":
    main()