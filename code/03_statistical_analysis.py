"""Analyse the fixed-dataset factorial experiment using paired seed blocks."""
from __future__ import annotations
import argparse
from pathlib import Path
from paired_inference import analyse, write_results

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'results' / 'full_results.csv')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'results' / 'stats')
    parser.add_argument('--expected-seeds', type=int, default=20)
    parser.add_argument('--bootstrap', type=int, default=5000)
    parser.add_argument('--bootstrap-seed', type=int, default=20261021)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--allow-partial-grid', action='store_true',
                        help='Allow identical complete Cartesian subgrids in every seed.')
    args = parser.parse_args()
    result = analyse(args.input, expected_seeds=args.expected_seeds,
                     n_boot=args.bootstrap, bootstrap_seed=args.bootstrap_seed,
                     alpha=args.alpha, allow_partial_grid=args.allow_partial_grid)
    write_results(result, args.out_dir)
    summary = result['summary']
    print(f"Analysed {summary['n_rows']} rows, {summary['n_seed_blocks']} seed blocks, "
          f"{summary['n_dataset_model_pairs']} dataset/model pairs.")
    print('Target: expected run metric over joint seed randomisation conditional on fixed datasets.')
    print(f'Wrote paired contrasts and descriptive summaries to {args.out_dir.resolve()}')


if __name__ == '__main__':
    main()
