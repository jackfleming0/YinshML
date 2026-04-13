#!/usr/bin/env python3
"""Checkpoint manager for large-scale self-play data collection."""

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import yaml

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoints for large-scale self-play data collection."""
    
    def __init__(self, config_path: str, output_dir: str):
        """Initialize the checkpoint manager.
        
        Args:
            config_path: Path to the configuration file
            output_dir: Directory where self-play data is being written
        """
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.config = self._load_config()
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        logger.info(f"Checkpoint manager initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_game_count(self) -> int:
        """Get current number of completed games."""
        try:
            # Look for parquet files or game records
            parquet_files = list(self.output_dir.glob("*.parquet"))
            if parquet_files:
                # Estimate based on file count and batch size
                batch_size = self.config['storage']['parquet_batch_size']
                return len(parquet_files) * batch_size
            
            # Look for JSON game files
            json_files = list(self.output_dir.glob("game_*.json"))
            return len(json_files)
            
        except Exception as e:
            logger.warning(f"Could not determine game count: {e}")
            return 0
    
    def _get_data_files(self) -> List[Path]:
        """Get list of data files to include in checkpoint."""
        data_files = []
        
        # Add parquet files
        data_files.extend(self.output_dir.glob("*.parquet"))
        
        # Add JSON game files
        data_files.extend(self.output_dir.glob("game_*.json"))
        
        # Add status files
        data_files.extend(self.output_dir.glob("status_report.json"))
        data_files.extend(self.output_dir.glob("monitoring.log"))
        
        return data_files
    
    def create_checkpoint(self, description: Optional[str] = None) -> str:
        """Create a checkpoint of current progress.
        
        Args:
            description: Optional description for the checkpoint
            
        Returns:
            Checkpoint ID
        """
        checkpoint_id = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        checkpoint_path.mkdir(exist_ok=True)
        
        # Get current game count
        game_count = self._get_game_count()
        
        # Copy data files
        data_files = self._get_data_files()
        for file_path in data_files:
            if file_path.exists():
                dest_path = checkpoint_path / file_path.name
                shutil.copy2(file_path, dest_path)
                logger.info(f"Copied {file_path.name} to checkpoint")
        
        # Create checkpoint metadata
        metadata = {
            'checkpoint_id': checkpoint_id,
            'timestamp': datetime.now().isoformat(),
            'description': description or f"Checkpoint at {game_count} games",
            'game_count': game_count,
            'data_files': [f.name for f in data_files if f.exists()],
            'config': self.config
        }
        
        metadata_file = checkpoint_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Created checkpoint {checkpoint_id} with {game_count} games")
        return checkpoint_id
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints.
        
        Returns:
            List of checkpoint metadata
        """
        checkpoints = []
        
        for checkpoint_path in self.checkpoint_dir.iterdir():
            if checkpoint_path.is_dir():
                metadata_file = checkpoint_path / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        checkpoints.append(metadata)
                    except Exception as e:
                        logger.warning(f"Could not read metadata for {checkpoint_path.name}: {e}")
        
        # Sort by timestamp
        checkpoints.sort(key=lambda x: x['timestamp'], reverse=True)
        return checkpoints
    
    def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore from a checkpoint.
        
        Args:
            checkpoint_id: ID of the checkpoint to restore
            
        Returns:
            True if successful, False otherwise
        """
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint {checkpoint_id} not found")
            return False
        
        metadata_file = checkpoint_path / "metadata.json"
        if not metadata_file.exists():
            logger.error(f"Metadata file not found for checkpoint {checkpoint_id}")
            return False
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            logger.error(f"Could not read metadata for checkpoint {checkpoint_id}: {e}")
            return False
        
        # Copy files back to output directory
        for file_name in metadata['data_files']:
            src_path = checkpoint_path / file_name
            dest_path = self.output_dir / file_name
            
            if src_path.exists():
                shutil.copy2(src_path, dest_path)
                logger.info(f"Restored {file_name}")
            else:
                logger.warning(f"File {file_name} not found in checkpoint")
        
        logger.info(f"Restored checkpoint {checkpoint_id} with {metadata['game_count']} games")
        return True
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint.
        
        Args:
            checkpoint_id: ID of the checkpoint to delete
            
        Returns:
            True if successful, False otherwise
        """
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint {checkpoint_id} not found")
            return False
        
        try:
            shutil.rmtree(checkpoint_path)
            logger.info(f"Deleted checkpoint {checkpoint_id}")
            return True
        except Exception as e:
            logger.error(f"Could not delete checkpoint {checkpoint_id}: {e}")
            return False
    
    def cleanup_old_checkpoints(self, keep_count: int = 5) -> int:
        """Clean up old checkpoints, keeping only the most recent ones.
        
        Args:
            keep_count: Number of checkpoints to keep
            
        Returns:
            Number of checkpoints deleted
        """
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= keep_count:
            logger.info(f"No cleanup needed: {len(checkpoints)} checkpoints (keep: {keep_count})")
            return 0
        
        # Delete old checkpoints
        deleted_count = 0
        for checkpoint in checkpoints[keep_count:]:
            if self.delete_checkpoint(checkpoint['checkpoint_id']):
                deleted_count += 1
        
        logger.info(f"Cleaned up {deleted_count} old checkpoints")
        return deleted_count
    
    def get_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Get the latest checkpoint.
        
        Returns:
            Latest checkpoint metadata or None
        """
        checkpoints = self.list_checkpoints()
        return checkpoints[0] if checkpoints else None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Manage checkpoints for large-scale self-play')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory where self-play data is being written')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create checkpoint command
    create_parser = subparsers.add_parser('create', help='Create a new checkpoint')
    create_parser.add_argument('--description', type=str,
                              help='Description for the checkpoint')
    
    # List checkpoints command
    subparsers.add_parser('list', help='List all available checkpoints')
    
    # Restore checkpoint command
    restore_parser = subparsers.add_parser('restore', help='Restore from a checkpoint')
    restore_parser.add_argument('--checkpoint-id', type=str, required=True,
                               help='ID of the checkpoint to restore')
    
    # Delete checkpoint command
    delete_parser = subparsers.add_parser('delete', help='Delete a checkpoint')
    delete_parser.add_argument('--checkpoint-id', type=str, required=True,
                              help='ID of the checkpoint to delete')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old checkpoints')
    cleanup_parser.add_argument('--keep-count', type=int, default=5,
                               help='Number of checkpoints to keep (default: 5)')
    
    # Latest checkpoint command
    subparsers.add_parser('latest', help='Get the latest checkpoint')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Validate arguments
    if not Path(args.config).exists():
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    if not Path(args.output_dir).exists():
        logger.error(f"Output directory not found: {args.output_dir}")
        sys.exit(1)
    
    # Create checkpoint manager
    manager = CheckpointManager(args.config, args.output_dir)
    
    # Execute command
    if args.command == 'create':
        checkpoint_id = manager.create_checkpoint(args.description)
        print(f"Created checkpoint: {checkpoint_id}")
        
    elif args.command == 'list':
        checkpoints = manager.list_checkpoints()
        if not checkpoints:
            print("No checkpoints found")
        else:
            print("Available checkpoints:")
            for checkpoint in checkpoints:
                print(f"  {checkpoint['checkpoint_id']}: {checkpoint['description']} "
                      f"({checkpoint['game_count']} games, {checkpoint['timestamp']})")
                
    elif args.command == 'restore':
        if manager.restore_checkpoint(args.checkpoint_id):
            print(f"Successfully restored checkpoint {args.checkpoint_id}")
        else:
            print(f"Failed to restore checkpoint {args.checkpoint_id}")
            sys.exit(1)
            
    elif args.command == 'delete':
        if manager.delete_checkpoint(args.checkpoint_id):
            print(f"Successfully deleted checkpoint {args.checkpoint_id}")
        else:
            print(f"Failed to delete checkpoint {args.checkpoint_id}")
            sys.exit(1)
            
    elif args.command == 'cleanup':
        deleted_count = manager.cleanup_old_checkpoints(args.keep_count)
        print(f"Cleaned up {deleted_count} old checkpoints")
        
    elif args.command == 'latest':
        latest = manager.get_latest_checkpoint()
        if latest:
            print(f"Latest checkpoint: {latest['checkpoint_id']}")
            print(f"Description: {latest['description']}")
            print(f"Games: {latest['game_count']}")
            print(f"Timestamp: {latest['timestamp']}")
        else:
            print("No checkpoints found")


if __name__ == '__main__':
    main()
