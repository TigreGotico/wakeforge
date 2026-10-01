#!/usr/bin/env python3
"""ESP32 genetic search: find the best sub-10KB model via evolutionary optimization.

Uses ``run_micro_search`` with composite fitness = 0.7*f1 + 0.3*compactness.
Models exceeding the param budget are hard-rejected (fitness = -1).

Usage:
    python examples/35_esp32_genetic_search.py --metadata dataset.csv
    python examples/35_esp32_genetic_search.py --metadata dataset.csv --tier esp32_nano
"""
import argparse
import logging

from ww_trainer.sweep import run_micro_search


def main() -> None:
    parser = argparse.ArgumentParser(description="ESP32 genetic wake-word search")
    parser.add_argument("--metadata", required=True, help="Dataset CSV path")
    parser.add_argument("--tier", default="esp32_sweet",
                        choices=["esp32_nano", "esp32_sweet", "esp32_max"])
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output-dir", default="micro_search_results")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    result = run_micro_search(
        metadata_csv=args.metadata,
        tier_name=args.tier,
        population_size=args.population,
        generations=args.generations,
        output_dir=args.output_dir,
        device=args.device,
        epochs_per_trial=args.epochs,
    )

    print(f"\nBest config: {result['best_config']}")
    print(f"Best F1: {result['best_f1']:.4f}")
    print(f"Best fitness: {result['best_fitness']:.4f}")
    print(f"Param budget: {result['param_budget']}")


if __name__ == "__main__":
    main()
