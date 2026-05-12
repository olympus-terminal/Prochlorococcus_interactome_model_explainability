#!/usr/bin/env python3
"""
Analyze AlphaFold3 results for all protein-protein interaction pairs.

This script processes AF3 predictions and generates explainability reports
using the Interactomics Explainability Framework.

Usage:
    python analyze_af3_results.py
"""

import json
import numpy as np
from pathlib import Path
import sys
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.analyzer import AF3Analyzer
from core.confidence_metrics import ConfidenceScores, ConfidenceMetrics
from utils.io import parse_cif_file, save_analysis_report


def analyze_single_prediction(result_dir: Path, job_name: str) -> dict:
    """
    Analyze a single AF3 prediction.

    Args:
        result_dir: Directory containing AF3 results
        job_name: Name of the job (e.g., 'real_PPI_01')

    Returns:
        Dictionary with analysis results
    """
    print(f"\n{'='*70}")
    print(f"Analyzing: {job_name}")
    print(f"{'='*70}")

    # Find structure file
    cif_files = list(result_dir.glob("*.cif"))
    if not cif_files:
        print(f"  ✗ No CIF file found in {result_dir}")
        return {"error": "No structure file found"}

    cif_file = cif_files[0]
    print(f"  Structure: {cif_file.name}")

    # Find confidence file
    conf_files = list(result_dir.glob("*confidence*.json"))
    if not conf_files:
        print(f"  ✗ No confidence JSON found in {result_dir}")
        return {"error": "No confidence scores found"}

    conf_file = conf_files[0]
    print(f"  Confidence: {conf_file.name}")

    try:
        # Load confidence data
        with open(conf_file, 'r') as f:
            conf_data = json.load(f)

        # Extract scores (AF3 format may vary - adjust as needed)
        plddt = np.array(conf_data.get('plddt', []))
        pae = np.array(conf_data.get('pae', []))
        ptm = conf_data.get('ptm', 0.0)
        iptm = conf_data.get('iptm', 0.0)

        print(f"  pLDDT: mean={np.mean(plddt):.2f}, shape={plddt.shape}")
        print(f"  PAE: shape={pae.shape}")
        print(f"  pTM: {ptm:.3f}")
        print(f"  ipTM: {iptm:.3f}")

        # Create confidence scores object
        scores = ConfidenceScores(
            plddt=plddt,
            pae=pae,
            ptm=ptm,
            iptm=iptm
        )

        # Analyze confidence
        analyzer = ConfidenceMetrics(scores)

        # Get summary
        summary = analyzer.get_confidence_summary()
        distribution = analyzer.get_confidence_distribution()
        quality, interpretation = analyzer.assess_overall_quality()

        # Compile results
        results = {
            "job_name": job_name,
            "structure_file": str(cif_file),
            "confidence_file": str(conf_file),
            "summary": summary,
            "distribution": distribution,
            "quality": quality,
            "interpretation": interpretation,
            "success": True
        }

        print(f"  ✓ Analysis complete")
        print(f"    Quality: {quality}")
        print(f"    Mean pLDDT: {summary['mean_plddt']:.2f}")
        print(f"    ipTM: {summary['iptm']:.3f}")

        return results

    except Exception as e:
        print(f"  ✗ Error during analysis: {e}")
        return {
            "job_name": job_name,
            "error": str(e),
            "success": False
        }


def analyze_all_results(base_dir: Path = Path("af3_results")) -> pd.DataFrame:
    """
    Analyze all AF3 results and generate comparison report.

    Args:
        base_dir: Base directory containing real_PPIs and random_PPIs folders

    Returns:
        DataFrame with all results
    """
    print("="*70)
    print("ANALYZING ALL AF3 RESULTS")
    print("="*70)

    all_results = []

    # Analyze real PPIs
    real_dir = base_dir / "real_PPIs"
    if real_dir.exists():
        print("\n" + "="*70)
        print("REAL PPIs")
        print("="*70)

        for job_dir in sorted(real_dir.iterdir()):
            if job_dir.is_dir():
                result = analyze_single_prediction(job_dir, job_dir.name)
                result["type"] = "real"
                all_results.append(result)

    # Analyze random PPIs
    random_dir = base_dir / "random_PPIs"
    if random_dir.exists():
        print("\n" + "="*70)
        print("RANDOM PPIs")
        print("="*70)

        for job_dir in sorted(random_dir.iterdir()):
            if job_dir.is_dir():
                result = analyze_single_prediction(job_dir, job_dir.name)
                result["type"] = "random"
                all_results.append(result)

    # Create DataFrame
    if not all_results:
        print("\n✗ No results found to analyze!")
        return pd.DataFrame()

    # Extract key metrics into flat structure
    flat_results = []
    for r in all_results:
        if r.get("success", False):
            flat_results.append({
                "job_name": r["job_name"],
                "type": r["type"],
                "mean_plddt": r["summary"]["mean_plddt"],
                "median_plddt": r["summary"]["median_plddt"],
                "ptm": r["summary"]["ptm"],
                "iptm": r["summary"]["iptm"],
                "ranking_confidence": r["summary"].get("ranking_confidence", 0),
                "quality": r["quality"],
                "very_high_pct": r["distribution"]["very_high_confidence_pct"],
                "high_pct": r["distribution"]["high_confidence_pct"],
                "low_pct": r["distribution"]["low_confidence_pct"],
            })

    df = pd.DataFrame(flat_results)
    return df


def generate_comparison_report(df: pd.DataFrame, output_file: str = "comparison_report.txt"):
    """Generate comparison report between real and random PPIs."""

    if df.empty:
        print("No data to compare")
        return

    print("\n" + "="*70)
    print("COMPARISON REPORT: REAL vs RANDOM PPIs")
    print("="*70)

    real_df = df[df["type"] == "real"]
    random_df = df[df["type"] == "random"]

    report = []
    report.append("="*70)
    report.append("COMPARISON REPORT: REAL vs RANDOM PPIs")
    report.append("="*70)
    report.append("")

    # Compare means
    metrics = ["mean_plddt", "median_plddt", "ptm", "iptm", "ranking_confidence"]

    report.append("METRIC COMPARISON:")
    report.append("-" * 70)
    report.append(f"{'Metric':<25} {'Real (mean)':<15} {'Random (mean)':<15} {'Difference':<15}")
    report.append("-" * 70)

    for metric in metrics:
        if metric in real_df.columns and metric in random_df.columns:
            real_mean = real_df[metric].mean()
            random_mean = random_df[metric].mean()
            diff = real_mean - random_mean
            report.append(f"{metric:<25} {real_mean:<15.3f} {random_mean:<15.3f} {diff:<15.3f}")

    report.append("")
    report.append("INTERPRETATION:")
    report.append("-" * 70)

    # Key metric: ipTM (interface confidence)
    real_iptm = real_df["iptm"].mean()
    random_iptm = random_df["iptm"].mean()

    if real_iptm > random_iptm + 0.1:
        report.append("✓ GOOD: Real PPIs have significantly higher ipTM (interface confidence)")
        report.append(f"  This suggests real interactions are predicted with higher confidence.")
    else:
        report.append("⚠ WARNING: Real and random PPIs have similar ipTM scores")
        report.append(f"  This may indicate:")
        report.append(f"    - Some 'random' pairs may actually interact")
        report.append(f"    - Dataset needs validation")
        report.append(f"    - AF3 cannot distinguish these pairs")

    report.append("")
    report.append("STATISTICAL SUMMARY:")
    report.append("-" * 70)
    report.append(f"Real PPIs analyzed: {len(real_df)}")
    report.append(f"Random PPIs analyzed: {len(random_df)}")
    report.append("")
    report.append(f"Real PPIs - Mean ipTM: {real_iptm:.3f} (±{real_df['iptm'].std():.3f})")
    report.append(f"Random PPIs - Mean ipTM: {random_iptm:.3f} (±{random_df['iptm'].std():.3f})")

    # Statistical test
    from scipy.stats import mannwhitneyu
    statistic, pvalue = mannwhitneyu(real_df["iptm"], random_df["iptm"])
    report.append("")
    report.append(f"Mann-Whitney U test p-value: {pvalue:.4e}")
    if pvalue < 0.05:
        report.append("✓ Statistically significant difference (p < 0.05)")
    else:
        report.append("✗ NOT statistically significant (p >= 0.05)")

    report.append("")
    report.append("="*70)

    # Print and save
    report_text = "\n".join(report)
    print(report_text)

    with open(output_file, 'w') as f:
        f.write(report_text)

    print(f"\n✓ Report saved to: {output_file}")


def main():
    """Main analysis pipeline."""

    print("\n" + "="*70)
    print("AlphaFold3 Results Analysis Pipeline")
    print("Interactomics Explainability Framework")
    print("="*70)

    # Check if results directory exists
    results_dir = Path("af3_results")
    if not results_dir.exists():
        print(f"\n✗ Results directory not found: {results_dir}")
        print("  Please download your AF3 results first!")
        print("  Expected structure:")
        print("    af3_results/")
        print("      real_PPIs/")
        print("        real_PPI_01/")
        print("        real_PPI_02/")
        print("        ...")
        print("      random_PPIs/")
        print("        random_PPI_01/")
        print("        ...")
        return

    # Analyze all results
    df = analyze_all_results(results_dir)

    if df.empty:
        print("\n✗ No successful analyses. Check that your results are downloaded.")
        return

    # Save detailed results
    output_csv = "analysis_results/detailed_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Detailed results saved to: {output_csv}")

    # Generate comparison report
    generate_comparison_report(df, "analysis_results/comparison_report.txt")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print(f"  - {output_csv}")
    print(f"  - analysis_results/comparison_report.txt")
    print("\nNext steps:")
    print("  - Review comparison_report.txt for key findings")
    print("  - Open detailed_results.csv in Excel/Python for further analysis")
    print("  - Use visualization tools to plot ipTM distributions")


if __name__ == "__main__":
    main()
