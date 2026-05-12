#!/usr/bin/env python3
"""
Batch submission script for AlphaFold3 Server.

This script submits multiple protein pairs to AlphaFold3 Server API.
Requires API key from Google AlphaFold Server.

Usage:
    python submit_to_af3_batch.py --api-key YOUR_API_KEY --type real
    python submit_to_af3_batch.py --api-key YOUR_API_KEY --type random
    python submit_to_af3_batch.py --api-key YOUR_API_KEY --type all
"""

import argparse
import json
import time
import requests
from pathlib import Path
from datetime import datetime
import sys

# AlphaFold3 Server API endpoint
AF3_API_URL = "https://alphafoldserver.com/api/v1/fold"

class AF3BatchSubmitter:
    def __init__(self, api_key: str, delay_seconds: int = 10):
        """
        Initialize batch submitter.

        Args:
            api_key: AlphaFold Server API key
            delay_seconds: Delay between submissions to avoid rate limiting
        """
        self.api_key = api_key
        self.delay_seconds = delay_seconds
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.submission_log = []

    def submit_job(self, json_file: Path) -> dict:
        """
        Submit a single job to AF3 Server.

        Args:
            json_file: Path to JSON input file

        Returns:
            Response dictionary with job_id and status
        """
        print(f"\n{'='*70}")
        print(f"Submitting: {json_file.name}")
        print(f"{'='*70}")

        # Read JSON input
        with open(json_file, 'r') as f:
            job_data = json.load(f)

        # Display job info
        print(f"Job name: {job_data['name']}")
        n_proteins = len(job_data['sequences'])
        total_length = sum(len(seq['proteinChain']['sequence'])
                          for seq in job_data['sequences'])
        print(f"Proteins: {n_proteins}")
        print(f"Total length: {total_length} amino acids")

        try:
            # Submit to AF3 Server
            print("Submitting to AlphaFold Server...")
            response = requests.post(
                AF3_API_URL,
                headers=self.headers,
                json=job_data,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                job_id = result.get('job_id', 'unknown')
                print(f"✓ SUCCESS! Job ID: {job_id}")

                log_entry = {
                    'file': str(json_file),
                    'job_name': job_data['name'],
                    'job_id': job_id,
                    'status': 'submitted',
                    'timestamp': datetime.now().isoformat(),
                    'total_length': total_length
                }
                self.submission_log.append(log_entry)
                return log_entry

            else:
                print(f"✗ FAILED! Status code: {response.status_code}")
                print(f"Response: {response.text}")

                log_entry = {
                    'file': str(json_file),
                    'job_name': job_data['name'],
                    'job_id': None,
                    'status': 'failed',
                    'error': response.text,
                    'timestamp': datetime.now().isoformat()
                }
                self.submission_log.append(log_entry)
                return log_entry

        except requests.exceptions.RequestException as e:
            print(f"✗ ERROR: {e}")
            log_entry = {
                'file': str(json_file),
                'job_name': job_data['name'],
                'job_id': None,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            self.submission_log.append(log_entry)
            return log_entry

    def submit_directory(self, directory: Path, dry_run: bool = False):
        """
        Submit all JSON files in a directory.

        Args:
            directory: Directory containing JSON files
            dry_run: If True, just list files without submitting
        """
        json_files = sorted(directory.glob("*.json"))

        if not json_files:
            print(f"No JSON files found in {directory}")
            return

        print(f"\nFound {len(json_files)} JSON files in {directory}")

        if dry_run:
            print("\n🔍 DRY RUN - Would submit:")
            for f in json_files:
                print(f"  - {f.name}")
            return

        print(f"\n{'='*70}")
        print(f"Starting batch submission: {len(json_files)} jobs")
        print(f"Delay between submissions: {self.delay_seconds} seconds")
        print(f"{'='*70}")

        for i, json_file in enumerate(json_files, 1):
            print(f"\n[{i}/{len(json_files)}]")
            self.submit_job(json_file)

            # Delay before next submission (except for last one)
            if i < len(json_files):
                print(f"\nWaiting {self.delay_seconds} seconds before next submission...")
                time.sleep(self.delay_seconds)

        self.print_summary()

    def print_summary(self):
        """Print submission summary."""
        print(f"\n{'='*70}")
        print("SUBMISSION SUMMARY")
        print(f"{'='*70}")

        total = len(self.submission_log)
        successful = sum(1 for log in self.submission_log if log['status'] == 'submitted')
        failed = total - successful

        print(f"\nTotal jobs: {total}")
        print(f"✓ Successful: {successful}")
        print(f"✗ Failed: {failed}")

        if successful > 0:
            print(f"\n{'='*70}")
            print("SUBMITTED JOBS:")
            print(f"{'='*70}")
            for log in self.submission_log:
                if log['status'] == 'submitted':
                    print(f"  {log['job_name']}: {log['job_id']}")

        if failed > 0:
            print(f"\n{'='*70}")
            print("FAILED JOBS:")
            print(f"{'='*70}")
            for log in self.submission_log:
                if log['status'] != 'submitted':
                    print(f"  {log['job_name']}: {log.get('error', 'Unknown error')}")

    def save_log(self, output_file: str = "submission_log.json"):
        """Save submission log to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"submission_log_{timestamp}.json"

        with open(output_path, 'w') as f:
            json.dump(self.submission_log, f, indent=2)

        print(f"\nSubmission log saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch submit protein pairs to AlphaFold3 Server"
    )
    parser.add_argument(
        '--api-key',
        required=True,
        help="AlphaFold Server API key"
    )
    parser.add_argument(
        '--type',
        choices=['real', 'random', 'all'],
        default='all',
        help="Which pairs to submit (default: all)"
    )
    parser.add_argument(
        '--delay',
        type=int,
        default=10,
        help="Delay in seconds between submissions (default: 10)"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Just list files without submitting"
    )

    args = parser.parse_args()

    # Initialize submitter
    submitter = AF3BatchSubmitter(args.api_key, args.delay)

    # Determine which directories to submit
    base_dir = Path(".")
    real_dir = base_dir / "af3_submissions_real"
    random_dir = base_dir / "af3_submissions_random"

    # Check directories exist
    if args.type in ['real', 'all'] and not real_dir.exists():
        print(f"Error: Directory not found: {real_dir}")
        sys.exit(1)

    if args.type in ['random', 'all'] and not random_dir.exists():
        print(f"Error: Directory not found: {random_dir}")
        sys.exit(1)

    # Submit jobs
    try:
        if args.type in ['real', 'all']:
            print("\n" + "="*70)
            print("SUBMITTING REAL PPIs")
            print("="*70)
            submitter.submit_directory(real_dir, args.dry_run)

        if args.type in ['random', 'all']:
            print("\n" + "="*70)
            print("SUBMITTING RANDOM PPIs")
            print("="*70)
            submitter.submit_directory(random_dir, args.dry_run)

        # Save log
        if not args.dry_run:
            submitter.save_log()

        print("\n" + "="*70)
        print("BATCH SUBMISSION COMPLETE!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Monitor job status on AlphaFold Server")
        print("  2. Wait for predictions to complete (may take hours)")
        print("  3. Download results using job IDs")
        print("  4. Run explainability analysis")

    except KeyboardInterrupt:
        print("\n\nSubmission interrupted by user.")
        submitter.print_summary()
        submitter.save_log()
        sys.exit(1)


if __name__ == "__main__":
    main()
