#!/usr/bin/env python3
"""
Setup Reference Baseline Script

This script preserves a specific trained model as a fixed reference 
for consistent Elo comparisons across future experiments.

Usage:
    python scripts/setup_reference_baseline.py

This will look for the value_recovery_20250630 iteration 2 model and 
preserve it as "baseline_reference" for tournament use.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from yinsh_ml.utils.reference_models import ReferenceModelManager

def setup_logging():
    """Setup logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def find_model_path(project_root: Path, experiment_name: str, iteration: int):
    """Find the model checkpoint file."""
    possible_paths = [
        project_root / "results" / experiment_name / f"iteration_{iteration}" / f"checkpoint_iteration_{iteration}.pt",
        project_root / "results" / experiment_name / f"checkpoint_iteration_{iteration}.pt",
        project_root / f"checkpoint_iteration_{iteration}.pt",
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None

def main():
    """Main setup function."""
    setup_logging()
    logger = logging.getLogger("setup_reference_baseline")
    
    logger.info("Setting up reference baseline model...")
    
    # Initialize reference model manager
    reference_manager = ReferenceModelManager(project_root)
    
    # Check if baseline already exists
    existing_baselines = reference_manager.get_reference_models()
    if "baseline_reference" in existing_baselines:
        logger.info("Baseline reference already exists!")
        baseline_info = existing_baselines["baseline_reference"]
        logger.info(f"  Name: {baseline_info.name}")
        logger.info(f"  Experiment: {baseline_info.original_experiment}")
        logger.info(f"  Iteration: {baseline_info.iteration}")
        logger.info(f"  Elo: {baseline_info.elo_rating}")
        logger.info(f"  Description: {baseline_info.description}")
        
        response = input("\nOverwrite existing baseline? (y/n): ").lower()
        if response != 'y':
            logger.info("Keeping existing baseline.")
            return
        
        # Remove existing baseline
        reference_manager.remove_reference_model("baseline_reference")
        logger.info("Removed existing baseline.")
    
    # Look for value_recovery_20250630 iteration 2
    experiment_name = "value_recovery_20250630"
    iteration = 2
    
    model_path = find_model_path(project_root, experiment_name, iteration)
    
    if model_path is None:
        logger.error(f"Could not find {experiment_name} iteration {iteration} model!")
        logger.error("Searched in:")
        possible_paths = [
            project_root / "results" / experiment_name / f"iteration_{iteration}" / f"checkpoint_iteration_{iteration}.pt",
            project_root / "results" / experiment_name / f"checkpoint_iteration_{iteration}.pt",
            project_root / f"checkpoint_iteration_{iteration}.pt",
        ]
        for path in possible_paths:
            logger.error(f"  {path}")
        
        logger.info("\nTo manually set up a reference baseline:")
        logger.info("1. Find your desired model checkpoint file")
        logger.info("2. Use the ReferenceModelManager.preserve_model() method")
        logger.info("3. Specify the experiment name, iteration, and known Elo rating")
        return
    
    logger.info(f"Found model at: {model_path}")
    
    # Preserve the model as a reference baseline
    try:
        reference_name = reference_manager.preserve_model(
            experiment_name=experiment_name,
            iteration=iteration,
            model_path=model_path,
            elo_rating=1516.9,  # Known from analysis document
            reference_name="baseline_reference",
            description="Standard baseline from value_recovery_20250630 iteration 2 (Elo 1516.9)",
            games_per_tournament=150  # More games for important baseline
        )
        
        logger.info(f"✓ Successfully created reference baseline: {reference_name}")
        logger.info("This model will now be included in all future tournament comparisons")
        
    except Exception as e:
        logger.error(f"Failed to create reference baseline: {e}")
        return
    
    # List all reference models
    logger.info("\nAll reference models:")
    for name, info in reference_manager.get_reference_models().items():
        logger.info(f"  {name}: {info.description} (Elo {info.elo_rating})")

if __name__ == "__main__":
    main() 