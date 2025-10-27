"""Game state recording system for Yinsh self-play."""

import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging

from ..game import GameState, Move, Player
from ..analysis.feature_extraction import FeatureExtractor, FeatureVector

logger = logging.getLogger(__name__)


@dataclass
class GameTurn:
    """Represents a single turn in a game."""
    turn_number: int
    current_player: str
    move: Dict[str, Any]  # Serialized move
    features: Dict[str, Any]  # Feature vector as dict
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class GameRecord:
    """Represents a complete game record."""
    game_id: str
    start_time: float
    end_time: float
    duration: float
    total_turns: int
    winner: Optional[str]
    final_score: Dict[str, int]
    turns: List[GameTurn]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def get_feature_matrix(self) -> List[List[float]]:
        """Get feature vectors as a matrix for ML training."""
        return [turn.features for turn in self.turns]
    
    def get_move_sequence(self) -> List[Dict[str, Any]]:
        """Get sequence of moves."""
        return [turn.move for turn in self.turns]


class GameRecorder:
    """Records game state data during self-play."""
    
    def __init__(self, output_dir: Union[str, Path] = "self_play_data", save_json: bool = True):
        """Initialize the game recorder.
        
        Args:
            output_dir: Directory to save game records
            save_json: Whether to save individual JSON files (disable when using parquet storage)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_json = save_json
        
        self.current_game: Optional[GameRecord] = None
        self.feature_extractor = FeatureExtractor()
        
        logger.info(f"Initialized GameRecorder with output directory: {self.output_dir}, save_json: {self.save_json}")
    
    def start_game(self, game_id: Optional[str] = None) -> str:
        """Start recording a new game.
        
        Args:
            game_id: Optional game ID. If None, generates one.
            
        Returns:
            Game ID for the started game
        """
        if game_id is None:
            game_id = f"game_{int(time.time() * 1000)}"
        
        self.current_game = GameRecord(
            game_id=game_id,
            start_time=time.time(),
            end_time=0.0,
            duration=0.0,
            total_turns=0,
            winner=None,
            final_score={"white": 0, "black": 0},
            turns=[],
            metadata={}
        )
        
        logger.info(f"Started recording game: {game_id}")
        return game_id
    
    def record_turn(self, game_state: GameState, move: Move) -> None:
        """Record a turn in the current game.
        
        Args:
            game_state: Current game state
            move: Move that was made
        """
        if self.current_game is None:
            logger.error("No active game to record turn")
            return
        
        # Extract features for current player
        features = self.feature_extractor.extract_all_features(game_state, game_state.current_player)
        
        # Create turn record
        turn = GameTurn(
            turn_number=len(self.current_game.turns) + 1,
            current_player=game_state.current_player.value,
            move=self._serialize_move(move),
            features=features.to_dict(),
            timestamp=time.time()
        )
        
        self.current_game.turns.append(turn)
        self.current_game.total_turns = len(self.current_game.turns)
        
        logger.debug(f"Recorded turn {turn.turn_number}: {move}")
    
    def end_game(self, game_state: GameState, winner: Optional[Player] = None) -> Optional[GameRecord]:
        """End the current game and return the complete record.
        
        Args:
            game_state: Final game state
            winner: Winner of the game, if any
            
        Returns:
            Complete game record, or None if no active game
        """
        if self.current_game is None:
            logger.error("No active game to end")
            return None
        
        # Update game record
        self.current_game.end_time = time.time()
        self.current_game.duration = self.current_game.end_time - self.current_game.start_time
        self.current_game.winner = winner.value if winner is not None else None
        self.current_game.final_score = {
            "white": game_state.white_score,
            "black": game_state.black_score
        }
        
        # Add metadata
        self.current_game.metadata.update({
            "final_phase": game_state.phase.value,
            "total_moves": len(game_state.move_history),
            "feature_count": len(self.current_game.turns[0].features) if self.current_game.turns else 0
        })
        
        # Save to file only if JSON saving is enabled
        if self.save_json:
            self._save_game_record(self.current_game)
        
        logger.info(f"Ended game {self.current_game.game_id}: {self.current_game.total_turns} turns, "
                   f"duration: {self.current_game.duration:.2f}s, winner: {self.current_game.winner}")
        
        # Return copy and clear current game
        game_record = self.current_game
        self.current_game = None
        return game_record
    
    def _serialize_move(self, move: Move) -> Dict[str, Any]:
        """Serialize a move to dictionary format.
        
        Args:
            move: Move to serialize
            
        Returns:
            Serialized move as dictionary
        """
        result = {
            "type": move.type.value,
            "player": move.player.value
        }
        
        if move.source:
            result["source"] = str(move.source)
        if move.destination:
            result["destination"] = str(move.destination)
        if move.markers:
            result["markers"] = [str(pos) for pos in move.markers]
        
        return result
    
    def _save_game_record(self, game_record: GameRecord) -> None:
        """Save a game record to file.
        
        Args:
            game_record: Game record to save
        """
        filename = f"{game_record.game_id}.json"
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(game_record.to_dict(), f, indent=2)
            logger.debug(f"Saved game record to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save game record: {e}")
    
    def load_game_record(self, game_id: str) -> Optional[GameRecord]:
        """Load a game record from file.
        
        Args:
            game_id: Game ID to load
            
        Returns:
            Game record, or None if not found
        """
        filename = f"{game_id}.json"
        filepath = self.output_dir / filename
        
        if not filepath.exists():
            logger.warning(f"Game record not found: {filepath}")
            return None
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Reconstruct GameRecord
            turns = [GameTurn(**turn_data) for turn_data in data['turns']]
            
            game_record = GameRecord(
                game_id=data['game_id'],
                start_time=data['start_time'],
                end_time=data['end_time'],
                duration=data['duration'],
                total_turns=data['total_turns'],
                winner=data['winner'],
                final_score=data['final_score'],
                turns=turns,
                metadata=data['metadata']
            )
            
            logger.debug(f"Loaded game record: {game_id}")
            return game_record
            
        except Exception as e:
            logger.error(f"Failed to load game record {game_id}: {e}")
            return None
    
    def list_game_records(self) -> List[str]:
        """List all available game record IDs.
        
        Returns:
            List of game IDs
        """
        game_files = list(self.output_dir.glob("*.json"))
        game_ids = [f.stem for f in game_files]
        logger.debug(f"Found {len(game_ids)} game records")
        return sorted(game_ids)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about recorded games.
        
        Returns:
            Dictionary with statistics
        """
        game_ids = self.list_game_records()
        
        if not game_ids:
            return {
                "total_games": 0,
                "total_turns": 0,
                "average_game_length": 0.0,
                "average_duration": 0.0
            }
        
        total_turns = 0
        total_duration = 0.0
        game_lengths = []
        
        for game_id in game_ids:
            game_record = self.load_game_record(game_id)
            if game_record:
                total_turns += game_record.total_turns
                total_duration += game_record.duration
                game_lengths.append(game_record.total_turns)
        
        return {
            "total_games": len(game_ids),
            "total_turns": total_turns,
            "average_game_length": sum(game_lengths) / len(game_lengths) if game_lengths else 0.0,
            "average_duration": total_duration / len(game_ids) if game_ids else 0.0,
            "shortest_game": min(game_lengths) if game_lengths else 0,
            "longest_game": max(game_lengths) if game_lengths else 0
        }
    
    def export_to_csv(self, output_file: Union[str, Path]) -> None:
        """Export all game data to CSV format for analysis.
        
        Args:
            output_file: Path to output CSV file
        """
        import pandas as pd
        
        all_turns = []
        game_ids = self.list_game_records()
        
        for game_id in game_ids:
            game_record = self.load_game_record(game_id)
            if game_record:
                for turn in game_record.turns:
                    row = {
                        'game_id': game_id,
                        'turn_number': turn.turn_number,
                        'current_player': turn.current_player,
                        'move_type': turn.move['type'],
                        'move_source': turn.move.get('source'),
                        'move_destination': turn.move.get('destination'),
                        'move_markers': str(turn.move.get('markers', [])),
                        'timestamp': turn.timestamp,
                        'winner': game_record.winner,
                        'game_duration': game_record.duration,
                        'total_turns': game_record.total_turns
                    }
                    
                    # Add feature columns
                    row.update(turn.features)
                    all_turns.append(row)
        
        if all_turns:
            df = pd.DataFrame(all_turns)
            df.to_csv(output_file, index=False)
            logger.info(f"Exported {len(all_turns)} turns to {output_file}")
        else:
            logger.warning("No game data to export")
