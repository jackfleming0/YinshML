#!/usr/bin/env python3
"""
Example: Tournament Integration with Reference Models

This shows how to modify your existing training script to include
reference models in tournaments for consistent Elo comparisons.

Key Points:
- Reference models are preserved trained models (e.g., value_recovery_20250630 iteration 2)
- They provide consistent Elo baselines across different experiments
- All future tournaments automatically include these reference opponents
- No need for abstract "control agents" - just preserved real models
"""

import sys
import logging
from pathlib import Path

# Add project root to path  
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from yinsh_ml.utils.reference_models import ReferenceModelManager, integrate_reference_models_with_tournament


def example_training_script_modification():
    """
    Example showing how to modify your existing training script.
    
    This demonstrates the minimal changes needed to include reference
    models in your tournament system.
    """
    
    # Your existing imports and setup...
    # from yinsh_ml.training.trainer import TrainingManager
    # from yinsh_ml.training.tournament import TournamentManager
    
    print("=== Example Tournament Integration ===\n")
    
    # Step 1: Initialize reference model manager
    print("1. Initialize reference model manager:")
    reference_manager = ReferenceModelManager(project_root)
    print(f"   Reference models directory: {reference_manager.reference_dir}")
    
    # Step 2: List existing reference models
    print("\n2. Check existing reference models:")
    reference_models = reference_manager.get_reference_models()
    if reference_models:
        for name, info in reference_models.items():
            print(f"   - {name}: {info.description} (Elo {info.elo_rating})")
    else:
        print("   No reference models found. Run setup_reference_baseline.py first.")
    
    # Step 3: Show how to integrate with tournament manager
    print("\n3. Tournament integration example:")
    print("""
    # In your training script, after creating the tournament manager:
    
    tournament_manager = TournamentManager(
        training_dir=training_dir,
        device=device,
        results_file=results_file
    )
    
    # Add this single line to enable reference model integration:
    integrate_reference_models_with_tournament(tournament_manager, reference_manager)
    
    # Now all tournaments will automatically include reference models!
    tournament_manager.run_round_robin_tournament(current_iteration)
    """)
    
    # Step 4: Show what happens during tournaments
    print("\n4. What happens during tournaments:")
    print("""
    When you run a tournament, the system will:
    - Run normal round-robin between current models (unchanged)
    - Automatically load reference models 
    - Play additional matches: current models vs reference models
    - Update Elo ratings including reference model ratings
    - Log results showing both current and reference model performance
    
    Example output:
        Running round-robin tournament...
        iteration_5 vs iteration_4: iteration_5 wins 65%
        iteration_4 vs iteration_3: iteration_4 wins 72%
        
        Running matches against 1 reference models...
        Match: iteration_5 vs baseline_reference
        iteration_5 vs baseline_reference: 58.3% win rate
        
        Updated ratings (including references):
          iteration_3: 1489.2
          iteration_4: 1523.7
          iteration_5: 1547.8
        Reference models:
          baseline_reference: 1516.9
    """)


def example_adding_new_reference_models():
    """Example of how to add new reference models."""
    
    print("\n=== Example: Adding New Reference Models ===\n")
    
    reference_manager = ReferenceModelManager(project_root)
    
    print("To preserve a new model as a reference baseline:")
    print("""
    # If you want to preserve the best model from a new experiment:
    
    reference_manager.preserve_model(
        experiment_name="new_experiment_20250701",
        iteration=8,  # The best iteration from that experiment
        model_path=Path("results/new_experiment_20250701/checkpoint_iteration_8.pt"),
        elo_rating=1623.4,  # The known Elo from tournament results
        reference_name="strong_baseline",  # Custom name
        description="Strong model from new_experiment_20250701 iter 8",
        games_per_tournament=100  # How many games to play against this reference
    )
    """)
    
    print("After preserving, this model becomes available for all future tournaments.")


def main():
    """Main demonstration."""
    logging.basicConfig(level=logging.INFO)
    
    example_training_script_modification()
    example_adding_new_reference_models()
    
    print("\n=== Summary ===")
    print("""
    Reference Model System Benefits:
    ✓ Consistent Elo comparisons across experiments
    ✓ No need for abstract control agents - use real trained models
    ✓ Minimal code changes - just one integration line
    ✓ Automatic tournament inclusion
    ✓ Preserved model files for reproducibility
    
    Getting Started:
    1. Run: python scripts/setup_reference_baseline.py
    2. Add integration line to your training script
    3. All future tournaments include reference baseline automatically!
    """)


if __name__ == "__main__":
    main() 