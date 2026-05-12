#!/usr/bin/env python3
"""
Analyze AlphaFold3 results for Prochlorococcus MED4 protein pairs.

This script processes AF3 predictions and generates explainability reports
using the Interactomics Explainability Framework.

Timestamp: 2025-10-11
Input: results/af3-out/ (relative to repo root)
Output: analysis_results/med4_results_TIMESTAMP/

Usage:
    python analyze_med4_results_20251011.py
"""

import json
import numpy as np
from pathlib import Path
import sys
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.confidence_metrics import ConfidenceScores, ConfidenceMetrics


def load_confidence_data(conf_file: Path) -> Dict:
    """Load confidence data from summary_confidences JSON file."""
    with open(conf_file, 'r') as f:
        return json.load(f)


def analyze_single_model(
    orf_pair: str,
    model_idx: int,
    conf_file: Path,
    cif_file: Path
) -> Dict:
    """
    Analyze a single AF3 model.

    Args:
        orf_pair: Name of the orf pair (e.g., 'orf_pair_017')
        model_idx: Model index (0-4)
        conf_file: Path to summary_confidences JSON
        cif_file: Path to model CIF file

    Returns:
        Dictionary with analysis results
    """
    try:
        # Load confidence data
        conf_data = load_confidence_data(conf_file)

        # Extract key metrics
        ptm = conf_data.get('ptm', 0.0)
        iptm = conf_data.get('iptm', 0.0)
        ranking_score = conf_data.get('ranking_score', 0.0)
        fraction_disordered = conf_data.get('fraction_disordered', 0.0)
        has_clash = conf_data.get('has_clash', 0.0)
        num_recycles = conf_data.get('num_recycles', 0.0)

        # Chain-specific metrics
        chain_ptm = conf_data.get('chain_ptm', [])
        chain_iptm = conf_data.get('chain_iptm', [])
        chain_pair_iptm = conf_data.get('chain_pair_iptm', [])
        chain_pair_pae_min = conf_data.get('chain_pair_pae_min', [])

        # Calculate derived metrics
        ranking_confidence = 0.8 * iptm + 0.2 * ptm

        # Assess quality based on iptm and ptm
        if iptm > 0.7 and ptm > 0.7:
            quality = "Excellent"
        elif iptm > 0.5 and ptm > 0.5:
            quality = "Good"
        elif iptm > 0.3 and ptm > 0.3:
            quality = "Moderate"
        else:
            quality = "Low"

        # Get interface PAE if available
        interface_pae = None
        if chain_pair_pae_min and len(chain_pair_pae_min) >= 2:
            # PAE between chain 0 and chain 1
            interface_pae = chain_pair_pae_min[0][1] if len(chain_pair_pae_min[0]) > 1 else None

        results = {
            "orf_pair": orf_pair,
            "model_idx": model_idx,
            "ptm": ptm,
            "iptm": iptm,
            "ranking_score": ranking_score,
            "ranking_confidence": ranking_confidence,
            "quality": quality,
            "fraction_disordered": fraction_disordered,
            "has_clash": bool(has_clash),
            "num_recycles": int(num_recycles),
            "chain_a_ptm": chain_ptm[0] if len(chain_ptm) > 0 else None,
            "chain_b_ptm": chain_ptm[1] if len(chain_ptm) > 1 else None,
            "chain_a_iptm": chain_iptm[0] if len(chain_iptm) > 0 else None,
            "chain_b_iptm": chain_iptm[1] if len(chain_iptm) > 1 else None,
            "interface_pae": interface_pae,
            "cif_file": str(cif_file),
            "conf_file": str(conf_file),
            "success": True
        }

        return results

    except Exception as e:
        return {
            "orf_pair": orf_pair,
            "model_idx": model_idx,
            "error": str(e),
            "success": False
        }


def analyze_orf_pair(orf_pair_dir: Path) -> List[Dict]:
    """
    Analyze all 5 models for a single orf pair.

    Args:
        orf_pair_dir: Directory containing orf pair results

    Returns:
        List of dictionaries with results for each model
    """
    orf_pair = orf_pair_dir.name
    print(f"\nAnalyzing: {orf_pair}")
    print("-" * 60)

    results = []

    # Process each of the 5 models
    for model_idx in range(5):
        # Find confidence and CIF files
        conf_pattern = f"*summary_confidences_{model_idx}.json"
        cif_pattern = f"*model_{model_idx}.cif"

        conf_files = list(orf_pair_dir.glob(conf_pattern))
        cif_files = list(orf_pair_dir.glob(cif_pattern))

        if not conf_files or not cif_files:
            print(f"  Model {model_idx}: Missing files - skipping")
            continue

        conf_file = conf_files[0]
        cif_file = cif_files[0]

        # Analyze this model
        result = analyze_single_model(orf_pair, model_idx, conf_file, cif_file)

        if result["success"]:
            print(f"  Model {model_idx}: iptm={result['iptm']:.3f}, ptm={result['ptm']:.3f}, quality={result['quality']}")
        else:
            print(f"  Model {model_idx}: ERROR - {result.get('error', 'Unknown')}")

        results.append(result)

    return results


def analyze_all_med4_pairs(base_dir: Path) -> pd.DataFrame:
    """
    Analyze all MED4 orf pairs.

    Args:
        base_dir: Base directory containing orf_pair_* folders

    Returns:
        DataFrame with all results
    """
    print("=" * 70)
    print("ANALYZING ALL PROCHLOROCOCCUS MED4 PROTEIN PAIRS")
    print("=" * 70)

    all_results = []

    # Find all orf_pair directories
    orf_pair_dirs = sorted(base_dir.glob("orf_pair_*"))

    print(f"\nFound {len(orf_pair_dirs)} orf pair directories")

    # Analyze each orf pair
    for orf_pair_dir in orf_pair_dirs:
        if orf_pair_dir.is_dir():
            pair_results = analyze_orf_pair(orf_pair_dir)
            all_results.extend(pair_results)

    # Convert to DataFrame
    if not all_results:
        print("\n✗ No results found to analyze!")
        return pd.DataFrame()

    # Filter successful results
    successful_results = [r for r in all_results if r.get("success", False)]

    print(f"\n{'=' * 70}")
    print(f"Successfully analyzed: {len(successful_results)} / {len(all_results)} models")
    print(f"{'=' * 70}")

    df = pd.DataFrame(successful_results)
    return df


def generate_summary_report(df: pd.DataFrame, output_file: Path):
    """Generate summary report for MED4 analysis."""

    if df.empty:
        print("No data to summarize")
        return

    report = []
    report.append("=" * 70)
    report.append("PROCHLOROCOCCUS MED4 PROTEIN INTERACTIONS")
    report.append("ALPHAFOLD3 EXPLAINABILITY ANALYSIS REPORT")
    report.append("=" * 70)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append(f"Total protein pairs analyzed: {df['orf_pair'].nunique()}")
    report.append(f"Total models analyzed: {len(df)}")
    report.append("")

    # Overall statistics
    report.append("=" * 70)
    report.append("OVERALL CONFIDENCE METRICS")
    report.append("=" * 70)
    report.append("")
    report.append(f"ipTM (interface confidence):")
    report.append(f"  Mean:   {df['iptm'].mean():.3f}")
    report.append(f"  Median: {df['iptm'].median():.3f}")
    report.append(f"  Std:    {df['iptm'].std():.3f}")
    report.append(f"  Min:    {df['iptm'].min():.3f}")
    report.append(f"  Max:    {df['iptm'].max():.3f}")
    report.append("")

    report.append(f"pTM (overall structure confidence):")
    report.append(f"  Mean:   {df['ptm'].mean():.3f}")
    report.append(f"  Median: {df['ptm'].median():.3f}")
    report.append(f"  Std:    {df['ptm'].std():.3f}")
    report.append(f"  Min:    {df['ptm'].min():.3f}")
    report.append(f"  Max:    {df['ptm'].max():.3f}")
    report.append("")

    report.append(f"Ranking Score:")
    report.append(f"  Mean:   {df['ranking_score'].mean():.3f}")
    report.append(f"  Median: {df['ranking_score'].median():.3f}")
    report.append("")

    # Quality distribution
    report.append("=" * 70)
    report.append("QUALITY DISTRIBUTION")
    report.append("=" * 70)
    report.append("")
    quality_counts = df['quality'].value_counts()
    for quality in ["Excellent", "Good", "Moderate", "Low"]:
        count = quality_counts.get(quality, 0)
        pct = 100 * count / len(df)
        report.append(f"  {quality:12s}: {count:3d} models ({pct:5.1f}%)")
    report.append("")

    # Best predictions per pair (based on ranking_confidence)
    report.append("=" * 70)
    report.append("BEST MODEL PER PROTEIN PAIR")
    report.append("=" * 70)
    report.append("")

    # Get best model for each pair
    best_models = df.loc[df.groupby('orf_pair')['ranking_confidence'].idxmax()]
    best_models = best_models.sort_values('ranking_confidence', ascending=False)

    report.append(f"{'Pair':<15} {'Model':<7} {'ipTM':<8} {'pTM':<8} {'Rank':<8} {'Quality':<12}")
    report.append("-" * 70)

    for _, row in best_models.iterrows():
        report.append(f"{row['orf_pair']:<15} {row['model_idx']:<7} "
                     f"{row['iptm']:<8.3f} {row['ptm']:<8.3f} "
                     f"{row['ranking_confidence']:<8.3f} {row['quality']:<12}")

    report.append("")

    # High confidence interactions (iptm > 0.5)
    report.append("=" * 70)
    report.append("HIGH CONFIDENCE INTERACTIONS (ipTM > 0.5)")
    report.append("=" * 70)
    report.append("")

    high_conf = best_models[best_models['iptm'] > 0.5].sort_values('iptm', ascending=False)

    if len(high_conf) > 0:
        report.append(f"Found {len(high_conf)} protein pairs with high interface confidence")
        report.append("")
        report.append(f"{'Pair':<15} {'ipTM':<8} {'pTM':<8} {'Quality':<12} {'Has Clash':<10}")
        report.append("-" * 70)

        for _, row in high_conf.iterrows():
            clash_str = "Yes" if row['has_clash'] else "No"
            report.append(f"{row['orf_pair']:<15} {row['iptm']:<8.3f} "
                         f"{row['ptm']:<8.3f} {row['quality']:<12} {clash_str:<10}")
    else:
        report.append("No high confidence interactions found (all ipTM <= 0.5)")
        report.append("This may indicate:")
        report.append("  - Weak or transient interactions")
        report.append("  - Proteins that don't strongly interact")
        report.append("  - Need for additional validation")

    report.append("")

    # Structural issues
    report.append("=" * 70)
    report.append("STRUCTURAL QUALITY INDICATORS")
    report.append("=" * 70)
    report.append("")

    report.append(f"Models with clashes: {df['has_clash'].sum()} / {len(df)}")
    report.append(f"Mean fraction disordered: {df['fraction_disordered'].mean():.3f}")
    report.append(f"Mean recycling iterations: {df['num_recycles'].mean():.1f}")
    report.append("")

    # Interface PAE statistics
    if 'interface_pae' in df.columns and df['interface_pae'].notna().any():
        report.append("=" * 70)
        report.append("INTERFACE PAE (Predicted Aligned Error)")
        report.append("=" * 70)
        report.append("")
        report.append("Lower PAE indicates higher confidence in relative positioning")
        report.append("")
        pae_data = df[df['interface_pae'].notna()]['interface_pae']
        report.append(f"  Mean interface PAE:   {pae_data.mean():.2f} Å")
        report.append(f"  Median interface PAE: {pae_data.median():.2f} Å")
        report.append(f"  Min interface PAE:    {pae_data.min():.2f} Å")
        report.append(f"  Max interface PAE:    {pae_data.max():.2f} Å")
        report.append("")

    # Key findings
    report.append("=" * 70)
    report.append("KEY FINDINGS & INTERPRETATION")
    report.append("=" * 70)
    report.append("")

    mean_iptm = df['iptm'].mean()
    high_conf_count = len(best_models[best_models['iptm'] > 0.5])

    if mean_iptm > 0.5:
        report.append("✓ POSITIVE: High average interface confidence suggests these")
        report.append("  protein pairs have genuine interaction potential.")
    elif mean_iptm > 0.3:
        report.append("~ MODERATE: Average interface confidence suggests some pairs")
        report.append("  may interact, but confidence is not uniformly high.")
    else:
        report.append("! LOW: Low average interface confidence suggests many pairs")
        report.append("  may not form stable complexes under these conditions.")

    report.append("")
    report.append(f"Protein pairs with likely strong interactions: {high_conf_count}")
    report.append("")

    report.append("RECOMMENDATIONS:")
    report.append("")
    report.append("1. Focus on protein pairs with ipTM > 0.5 for experimental validation")
    report.append("2. Examine interface residues in high-confidence models for:")
    report.append("   - Key binding residues")
    report.append("   - Evolutionary conservation")
    report.append("   - Potential functional sites")
    report.append("3. Low ipTM pairs may represent:")
    report.append("   - Weak/transient interactions")
    report.append("   - Condition-dependent interactions")
    report.append("   - False positives requiring validation")
    report.append("")

    report.append("=" * 70)
    report.append("END OF REPORT")
    report.append("=" * 70)

    # Print and save
    report_text = "\n".join(report)
    print("\n" + report_text)

    with open(output_file, 'w') as f:
        f.write(report_text)

    print(f"\n✓ Report saved to: {output_file}")


def main():
    """Main analysis pipeline."""

    print("\n" + "=" * 70)
    print("AlphaFold3 Results Analysis Pipeline")
    print("Prochlorococcus MED4 Interactome")
    print("Interactomics Explainability Framework")
    print("=" * 70)

    # Set paths
    base_dir = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "af3-out"))

    if not base_dir.exists():
        print(f"\n✗ MED4 directory not found: {base_dir}")
        return

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("analysis_results") / f"med4_results_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nResults will be saved to: {output_dir}")

    # Analyze all pairs
    df = analyze_all_med4_pairs(base_dir)

    if df.empty:
        print("\n✗ No successful analyses. Check your data files.")
        return

    # Save detailed results
    detailed_csv = output_dir / "detailed_results.csv"
    df.to_csv(detailed_csv, index=False)
    print(f"\n✓ Detailed results saved to: {detailed_csv}")

    # Save best models per pair
    best_models = df.loc[df.groupby('orf_pair')['ranking_confidence'].idxmax()]
    best_models = best_models.sort_values('ranking_confidence', ascending=False)
    best_csv = output_dir / "best_models_per_pair.csv"
    best_models.to_csv(best_csv, index=False)
    print(f"✓ Best models saved to: {best_csv}")

    # Generate summary report
    report_file = output_dir / "summary_report.txt"
    generate_summary_report(df, report_file)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print(f"  - {detailed_csv}")
    print(f"  - {best_csv}")
    print(f"  - {report_file}")
    print("\nNext steps:")
    print("  1. Review summary_report.txt for key findings")
    print("  2. Examine best_models_per_pair.csv for high-confidence pairs")
    print("  3. Open detailed_results.csv for comprehensive analysis")
    print("  4. Focus on pairs with ipTM > 0.5 for further study")
    print("")


if __name__ == "__main__":
    main()
