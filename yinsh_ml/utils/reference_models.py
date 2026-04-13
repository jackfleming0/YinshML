"""
Reference Model System for Tournament Baselines

Preserves specific trained models as fixed opponents for consistent 
Elo comparisons across different experiments.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime

from ..network.wrapper import NetworkWrapper


@dataclass
class ReferenceModelInfo:
    """Information about a preserved reference model."""
    name: str
    original_experiment: str
    iteration: int
    elo_rating: float
    preservation_date: str
    model_path: str
    description: str = ""
    games_per_tournament: int = 100


class ReferenceModelManager:
    """Manages preserved reference models for tournament baselines."""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.reference_dir = self.project_root / "reference_models"
        self.reference_dir.mkdir(exist_ok=True)
        
        self.config_file = self.reference_dir / "reference_models.json"
        self.logger = logging.getLogger("ReferenceModelManager")
        
        # Load existing reference models
        self.reference_models = self._load_reference_models()
    
    def _load_reference_models(self) -> Dict[str, ReferenceModelInfo]:
        """Load reference model configurations from file."""
        if not self.config_file.exists():
            return {}
        
        try:
            with open(self.config_file, 'r') as f:
                data = json.load(f)
            
            models = {}
            for name, info in data.items():
                models[name] = ReferenceModelInfo(**info)
            
            self.logger.info(f"Loaded {len(models)} reference models")
            return models
            
        except Exception as e:
            self.logger.error(f"Error loading reference models: {e}")
            return {}
    
    def _save_reference_models(self):
        """Save reference model configurations to file."""
        try:
            data = {}
            for name, model_info in self.reference_models.items():
                data[name] = asdict(model_info)
            
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving reference models: {e}")
    
    def preserve_model(self, 
                      experiment_name: str,
                      iteration: int,
                      model_path: Path,
                      elo_rating: float,
                      reference_name: Optional[str] = None,
                      description: str = "",
                      games_per_tournament: int = 100) -> str:
        """
        Preserve a model as a reference baseline.
        
        Args:
            experiment_name: Name of the experiment (e.g., "value_recovery_20250630")
            iteration: Iteration number (e.g., 2)
            model_path: Path to the model checkpoint
            elo_rating: Known Elo rating of this model
            reference_name: Custom name for reference (defaults to experiment_iter_X)
            description: Human-readable description
            games_per_tournament: How many games to play against this reference
            
        Returns:
            The reference name used
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Generate reference name if not provided
        if reference_name is None:
            reference_name = f"{experiment_name}_iter_{iteration}"
        
        # Copy model to reference directory
        reference_model_path = self.reference_dir / f"{reference_name}.pt"
        shutil.copy2(model_path, reference_model_path)
        
        # Create reference model info
        model_info = ReferenceModelInfo(
            name=reference_name,
            original_experiment=experiment_name,
            iteration=iteration,
            elo_rating=elo_rating,
            preservation_date=datetime.now().isoformat(),
            model_path=str(reference_model_path),
            description=description or f"Reference model from {experiment_name} iteration {iteration}",
            games_per_tournament=games_per_tournament
        )
        
        # Store and save
        self.reference_models[reference_name] = model_info
        self._save_reference_models()
        
        self.logger.info(f"Preserved reference model '{reference_name}' with Elo {elo_rating}")
        return reference_name
    
    def get_reference_models(self) -> Dict[str, ReferenceModelInfo]:
        """Get all available reference models."""
        return self.reference_models.copy()
    
    def get_reference_model(self, name: str) -> Optional[ReferenceModelInfo]:
        """Get specific reference model info."""
        return self.reference_models.get(name)
    
    def load_reference_model(self, name: str, device: str = 'cpu') -> Optional[NetworkWrapper]:
        """Load a reference model for tournament use."""
        if name not in self.reference_models:
            self.logger.error(f"Reference model '{name}' not found")
            return None
        
        model_info = self.reference_models[name]
        model_path = Path(model_info.model_path)
        
        if not model_path.exists():
            self.logger.error(f"Reference model file not found: {model_path}")
            return None
        
        try:
            model = NetworkWrapper(device=device)
            model.load_model(str(model_path))
            self.logger.info(f"Loaded reference model '{name}' from {model_path}")
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading reference model '{name}': {e}")
            return None
    
    def remove_reference_model(self, name: str) -> bool:
        """Remove a reference model."""
        if name not in self.reference_models:
            self.logger.warning(f"Reference model '{name}' not found")
            return False
        
        model_info = self.reference_models[name]
        model_path = Path(model_info.model_path)
        
        try:
            # Remove model file
            if model_path.exists():
                model_path.unlink()
            
            # Remove from registry
            del self.reference_models[name]
            self._save_reference_models()
            
            self.logger.info(f"Removed reference model '{name}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing reference model '{name}': {e}")
            return False
    
    def list_reference_models(self) -> List[str]:
        """List all reference model names."""
        return list(self.reference_models.keys())
    
    def get_tournament_opponents(self) -> List[tuple]:
        """
        Get list of reference models for tournament inclusion.
        
        Returns:
            List of (model_name, model_info) tuples
        """
        return [(name, info) for name, info in self.reference_models.items()]
    
    def create_default_baseline(self):
        """
        Create a default baseline from value_recovery_20250630 iteration 2 if available.
        """
        # Look for the known good baseline
        possible_paths = [
            self.project_root / "results" / "value_recovery_20250630" / "iteration_2" / "checkpoint_iteration_2.pt",
            self.project_root / "results" / "value_recovery_20250630" / "checkpoint_iteration_2.pt",
        ]
        
        for path in possible_paths:
            if path.exists():
                try:
                    self.preserve_model(
                        experiment_name="value_recovery_20250630",
                        iteration=2,
                        model_path=path,
                        elo_rating=1516.9,  # Known from analysis
                        reference_name="baseline_reference",
                        description="Standard baseline from value_recovery_20250630 iteration 2",
                        games_per_tournament=150  # More games for important baseline
                    )
                    self.logger.info("Created default baseline reference model")
                    return True
                except Exception as e:
                    self.logger.error(f"Failed to create default baseline: {e}")
        
        self.logger.warning("Could not find value_recovery_20250630 iteration 2 for default baseline")
        return False


# Integration with existing tournament system
def integrate_reference_models_with_tournament(tournament_manager, reference_manager: ReferenceModelManager):
    """
    Helper function to integrate reference models with existing tournament system.
    
    This modifies the tournament to always include reference models as opponents.
    """
    
    # Store original run_round_robin_tournament method
    original_run_tournament = tournament_manager.run_round_robin_tournament
    
    def enhanced_run_tournament(current_iteration: int):
        """Enhanced tournament that includes reference models."""
        
        # Get reference models
        reference_models = reference_manager.get_tournament_opponents()
        
        if not reference_models:
            # No reference models, run normal tournament
            return original_run_tournament(current_iteration)
        
        # Load reference models
        loaded_references = {}
        for ref_name, ref_info in reference_models:
            ref_model = reference_manager.load_reference_model(ref_name, tournament_manager.device)
            if ref_model:
                loaded_references[ref_name] = (ref_model, ref_info)
        
        if not loaded_references:
            tournament_manager.logger.warning("No reference models could be loaded")
            return original_run_tournament(current_iteration)
        
        # Run normal round-robin tournament first
        original_run_tournament(current_iteration)
        
        # Now run additional matches against reference models
        tournament_manager.logger.info(f"\nRunning matches against {len(loaded_references)} reference models...")
        
        # Get current model paths
        model_paths = [
            (i, tournament_manager.training_dir / f"iteration_{i}" / f"checkpoint_iteration_{i}.pt")
            for i in range(current_iteration + 1)
            if (tournament_manager.training_dir / f"iteration_{i}" / f"checkpoint_iteration_{i}.pt").exists()
        ]
        
        if not model_paths:
            tournament_manager.logger.warning("No trained models found for reference comparison")
            return
        
        # Load current models
        current_models = {}
        for iter_num, path in model_paths:
            model_id = f"iteration_{iter_num}"
            current_models[model_id] = tournament_manager._load_model(path)
            tournament_manager.glicko_tracker.add_model(model_id)
        
        # Add reference models to Glicko tracker
        for ref_name, (ref_model, ref_info) in loaded_references.items():
            tournament_manager.glicko_tracker.add_model(ref_name)
            # Initialize with known Elo
            tournament_manager.glicko_tracker.players[ref_name].rating = ref_info.elo_rating
        
        # Play matches between current models and reference models
        reference_results = []
        
        for current_id, current_model in current_models.items():
            for ref_name, (ref_model, ref_info) in loaded_references.items():
                
                # Play both colors
                tournament_manager.logger.info(f"Match: {current_id} vs {ref_name}")
                
                # Current model as white
                result_white = tournament_manager._play_match(
                    white_model=current_model,
                    black_model=ref_model,
                    white_id=current_id,
                    black_id=ref_name
                )
                
                # Reference model as white  
                result_black = tournament_manager._play_match(
                    white_model=ref_model,
                    black_model=current_model,
                    white_id=ref_name,
                    black_id=current_id
                )
                
                # Record results
                tournament_manager.glicko_tracker.record_match(
                    current_id, ref_name,
                    white_wins=result_white.white_wins,
                    black_wins=result_white.black_wins,
                    draws=result_white.draws
                )
                
                tournament_manager.glicko_tracker.record_match(
                    ref_name, current_id,
                    white_wins=result_black.white_wins,
                    black_wins=result_black.black_wins,
                    draws=result_black.draws
                )
                
                reference_results.extend([result_white, result_black])
                
                # Log immediate results
                total_wins = result_white.white_wins + result_black.black_wins
                total_games = result_white.total_games() + result_black.total_games()
                win_rate = total_wins / total_games if total_games > 0 else 0
                tournament_manager.logger.info(f"{current_id} vs {ref_name}: {win_rate:.1%} win rate")
        
        # Update all ratings including references
        tournament_manager.glicko_tracker.update_ratings()
        
        # Log updated ratings
        tournament_manager.logger.info("\nUpdated ratings (including references):")
        all_ratings = {}
        for model_id in tournament_manager.glicko_tracker.players:
            all_ratings[model_id] = tournament_manager.glicko_tracker.get_rating(model_id)
        
        # Sort current models by iteration
        current_ratings = {k: v for k, v in all_ratings.items() if k.startswith('iteration_')}
        ref_ratings = {k: v for k, v in all_ratings.items() if not k.startswith('iteration_')}
        
        for model_id in sorted(current_ratings.keys(), key=lambda x: int(x.split('_')[-1])):
            rating = current_ratings[model_id]
            tournament_manager.logger.info(f"  {model_id}: {rating:.1f}")
        
        if ref_ratings:
            tournament_manager.logger.info("Reference models:")
            for ref_name, rating in ref_ratings.items():
                tournament_manager.logger.info(f"  {ref_name}: {rating:.1f}")
    
    # Replace the tournament method
    tournament_manager.run_round_robin_tournament = enhanced_run_tournament 