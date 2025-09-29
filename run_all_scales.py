#!/usr/bin/env python3
"""
Unified runner for all three scales of models (small, medium, large)
Provides a single interface to run analysis at different scales
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Run multiscale analysis at different model scales')
    parser.add_argument(
        '--scale',
        type=str,
        choices=['small', 'medium', 'large', 'all'],
        default='medium',
        help='Model scale to run analysis for'
    )
    parser.add_argument(
        '--modalities',
        type=str,
        nargs='+',
        default=['vision', 'language', 'speech'],
        help='Modalities to include in analysis'
    )

    args = parser.parse_args()

    if args.scale == 'all':
        scales = ['small', 'medium', 'large']
    else:
        scales = [args.scale]

    for scale in scales:
        print(f"\n{'='*80}")
        print(f"Running {scale.upper()}-SCALE analysis...")
        print(f"{'='*80}\n")

        if scale == 'small':
            from extract_small_features import main as run_small
            run_small()
        elif scale == 'medium':
            from run_medium_scale import run_medium_scale
            run_medium_scale()
        elif scale == 'large':
            from run_large_scale import run_large_scale
            run_large_scale()

    print(f"\n{'='*80}")
    print(f"All requested scales completed successfully!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()