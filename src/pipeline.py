"""
Main Pipeline Entry Point

Run anonymisation pipelines for evaluating privacy-utility trade-offs.

Pipelines available:
- discretisation: Uses binning/discretisation as anonymisation
- mondrian: Uses Mondrian k-anonymity algorithm

Usage:
    python -m src.pipeline                    # Run both pipelines
    python -m src.pipeline --discretisation   # Run only discretisation
    python -m src.pipeline --mondrian         # Run only Mondrian
"""

import argparse

from .pipeline_discretisation import main as run_discretisation_pipeline
from .pipeline_mondrian import main as run_mondrian_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run anonymisation pipelines for privacy-utility evaluation"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    parser.add_argument(
        "--discretisation", 
        action="store_true",
        help="Run only the discretisation pipeline"
    )
    parser.add_argument(
        "--mondrian", 
        action="store_true",
        help="Run only the Mondrian pipeline"
    )
    
    args = parser.parse_args()
    
    # If neither flag is set, run both
    run_disc = args.discretisation or (not args.discretisation and not args.mondrian)
    run_mond = args.mondrian or (not args.discretisation and not args.mondrian)
    
    if run_disc:
        run_discretisation_pipeline(args.config)
        print("\n")
    
    if run_mond:
        run_mondrian_pipeline(args.config)


if __name__ == "__main__":
    main()