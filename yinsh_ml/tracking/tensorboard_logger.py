"""
TensorBoard logging integration for YinshML experiment tracking.

Provides seamless integration between ExperimentTracker and TensorBoard
for real-time visualization of training metrics.
"""

import os
import threading
import logging
import queue
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
from dataclasses import dataclass
import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        SummaryWriter = None

# Import Yinsh-specific components for visualization
try:
    from .yinsh_visualizer import YinshBoardVisualizer
    from ..game.board import Board
    from ..game.game_state import GameState
    from ..game.types import Move
    from ..game.constants import Position
    YINSH_VISUALIZATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Yinsh visualization components not available: {e}")
    YINSH_VISUALIZATION_AVAILABLE = False
    YinshBoardVisualizer = None

logger = logging.getLogger(__name__)


class TensorBoardLoggerError(Exception):
    """Base exception for TensorBoardLogger operations."""
    pass


@dataclass
class TensorBoardMetric:
    """Data structure for queued TensorBoard metrics."""
    experiment_id: int
    metric_type: str  # 'scalar', 'histogram', 'text'
    metric_name: str
    value: Any
    iteration: Optional[int] = None
    timestamp: Optional[datetime] = None
    tag: Optional[str] = None  # For histogram/text logging


class TensorBoardLogger:
    """
    TensorBoard logger that integrates with the ExperimentTracker.
    
    Provides automatic TensorBoard logging for experiment metrics with
    proper directory organization and resource management.
    """
    
    def __init__(self, 
                 log_dir: Union[str, Path] = None,
                 config: Dict[str, Any] = None):
        """
        Initialize the TensorBoardLogger.
        
        Args:
            log_dir: Base directory for TensorBoard logs. If None, uses 'logs/'
            config: Configuration dictionary for TensorBoard settings
            
        Raises:
            TensorBoardLoggerError: If TensorBoard is not available
        """
        if SummaryWriter is None:
            raise TensorBoardLoggerError(
                "TensorBoard is not available. Please install tensorboard or tensorboardX."
            )
        
        # Configuration defaults
        self._config = {
            'enabled': True,
            'flush_secs': 120,  # Flush to disk every 2 minutes
            'max_queue': 10,    # Max number of events to queue before writing
            'purge_step': None, # Don't purge old data by default
            'filename_suffix': '',
            # Background logging configuration
            'background_logging': True,  # Enable background thread processing
            'queue_size': 1000,  # Maximum queue size for metrics
            'batch_size': 50,    # Number of metrics to process per batch
            'worker_timeout': 0.1,  # Worker thread timeout in seconds
            'queue_timeout': 0.01,  # Timeout for queue operations (non-blocking)
        }
        
        # Update with provided config
        if config:
            self._config.update(config)
        
        # Set up base log directory
        if log_dir is None:
            log_dir = Path.cwd() / 'logs'
        self._base_log_dir = Path(log_dir)
        self._base_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Active writers for different experiments
        self._writers: Dict[int, SummaryWriter] = {}
        self._writers_lock = threading.Lock()
        
        # Track metrics per experiment for step management
        self._experiment_steps: Dict[int, Dict[str, int]] = {}
        self._steps_lock = threading.Lock()
        
        # Background logging infrastructure
        self._background_enabled = self._config.get('background_logging', True)
        self._metric_queue = None
        self._worker_thread = None
        self._shutdown_event = threading.Event()
        self._stats = {
            'queued_metrics': 0,
            'processed_metrics': 0,
            'dropped_metrics': 0,
            'queue_full_count': 0,
            'worker_errors': 0
        }
        self._stats_lock = threading.Lock()
        
        # Initialize background logging if enabled
        if self._background_enabled:
            self._initialize_background_logging()
        
        logger.info(f"TensorBoardLogger initialized with log directory: {self._base_log_dir} (background_logging: {self._background_enabled})")
    
    def is_enabled(self) -> bool:
        """Check if TensorBoard logging is enabled."""
        return self._config.get('enabled', True)
    
    def _initialize_background_logging(self) -> None:
        """Initialize background logging thread and queue."""
        try:
            queue_size = self._config.get('queue_size', 1000)
            self._metric_queue = queue.Queue(maxsize=queue_size)
            
            # Start worker thread
            self._worker_thread = threading.Thread(
                target=self._background_worker,
                name="TensorBoardLogger-Worker",
                daemon=True
            )
            self._worker_thread.start()
            
            logger.info(f"Background logging initialized with queue size: {queue_size}")
            
        except Exception as e:
            logger.error(f"Failed to initialize background logging: {e}")
            self._background_enabled = False
            raise TensorBoardLoggerError(f"Background logging initialization failed: {e}")
    
    def _background_worker(self) -> None:
        """Background worker thread for processing queued metrics."""
        logger.info("TensorBoard background worker started")
        batch_size = self._config.get('batch_size', 50)
        worker_timeout = self._config.get('worker_timeout', 0.1)
        
        while not self._shutdown_event.is_set():
            try:
                # Collect a batch of metrics
                batch = []
                
                # Get first metric (blocking with timeout)
                try:
                    metric = self._metric_queue.get(timeout=worker_timeout)
                    batch.append(metric)
                except queue.Empty:
                    continue
                
                # Collect additional metrics non-blocking up to batch_size
                while len(batch) < batch_size:
                    try:
                        metric = self._metric_queue.get_nowait()
                        batch.append(metric)
                    except queue.Empty:
                        break
                
                # Process the batch
                self._process_metric_batch(batch)
                
                # Mark all metrics as done
                for _ in batch:
                    self._metric_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in background worker: {e}")
                with self._stats_lock:
                    self._stats['worker_errors'] += 1
        
        logger.info("TensorBoard background worker stopped")
    
    def _process_metric_batch(self, batch: list) -> None:
        """Process a batch of metrics."""
        for metric in batch:
            try:
                if metric.metric_type == 'scalar':
                    self._log_scalar_sync(
                        metric.experiment_id,
                        metric.metric_name,
                        metric.value,
                        metric.iteration
                    )
                elif metric.metric_type == 'histogram':
                    self._log_histogram_sync(
                        metric.experiment_id,
                        metric.tag or metric.metric_name,
                        metric.value,
                        metric.iteration
                    )
                elif metric.metric_type == 'text':
                    self._log_text_sync(
                        metric.experiment_id,
                        metric.tag or metric.metric_name,
                        metric.value,
                        metric.iteration
                    )
                
                with self._stats_lock:
                    self._stats['processed_metrics'] += 1
                    
            except Exception as e:
                logger.error(f"Error processing metric {metric}: {e}")
                with self._stats_lock:
                    self._stats['worker_errors'] += 1
    
    def get_log_dir(self, experiment_id: int) -> Path:
        """
        Get the log directory for a specific experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Path to the experiment's log directory
        """
        return self._base_log_dir / f"experiment_{experiment_id}"
    
    def _get_or_create_writer(self, experiment_id: int) -> Optional[SummaryWriter]:
        """
        Get or create a SummaryWriter for the specified experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            SummaryWriter instance or None if disabled
        """
        logger.debug(f"_get_or_create_writer called for experiment {experiment_id}, enabled={self.is_enabled()}")
        
        if not self.is_enabled():
            logger.debug("TensorBoard is disabled, returning None")
            return None
        
        with self._writers_lock:
            if experiment_id not in self._writers:
                try:
                    log_dir = self.get_log_dir(experiment_id)
                    logger.debug(f"Creating log directory: {log_dir}")
                    log_dir.mkdir(parents=True, exist_ok=True)
                    
                    logger.debug(f"Creating SummaryWriter with config: {self._config}")
                    writer = SummaryWriter(
                        log_dir=str(log_dir),
                        flush_secs=self._config.get('flush_secs', 120),
                        max_queue=self._config.get('max_queue', 10),
                        purge_step=self._config.get('purge_step'),
                        filename_suffix=self._config.get('filename_suffix', '')
                    )
                    
                    self._writers[experiment_id] = writer
                    logger.info(f"Created TensorBoard writer for experiment {experiment_id} at {log_dir}")
                    
                except Exception as e:
                    logger.error(f"Failed to create TensorBoard writer for experiment {experiment_id}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    return None
            else:
                logger.debug(f"Using existing writer for experiment {experiment_id}")
            
            return self._writers.get(experiment_id)
    
    def _get_step(self, experiment_id: int, metric_name: str, iteration: Optional[int] = None) -> int:
        """
        Get the step number for a metric, using iteration if provided or auto-incrementing.
        
        Args:
            experiment_id: Experiment ID
            metric_name: Name of the metric
            iteration: Optional explicit iteration number
            
        Returns:
            Step number to use for TensorBoard logging
        """
        with self._steps_lock:
            if experiment_id not in self._experiment_steps:
                self._experiment_steps[experiment_id] = {}
            
            exp_steps = self._experiment_steps[experiment_id]
            
            if iteration is not None:
                # Use explicit iteration
                step = iteration
                # Update our tracking to ensure future auto-increments don't conflict
                exp_steps[metric_name] = max(exp_steps.get(metric_name, -1), step)
            else:
                # Auto-increment
                step = exp_steps.get(metric_name, -1) + 1
                exp_steps[metric_name] = step
            
            return step
    
    def _log_scalar_sync(self, 
                        experiment_id: int, 
                        metric_name: str, 
                        value: float, 
                        iteration: Optional[int] = None) -> None:
        """
        Log a scalar metric to TensorBoard synchronously.
        
        Args:
            experiment_id: Experiment ID
            metric_name: Name of the metric
            value: Metric value
            iteration: Optional explicit step/iteration number
        """
        logger.debug(f"TensorBoard log_scalar called: enabled={self.is_enabled()}, experiment_id={experiment_id}, metric={metric_name}, value={value}")
        
        if not self.is_enabled():
            logger.debug("TensorBoard logging is disabled, skipping")
            return
        
        try:
            writer = self._get_or_create_writer(experiment_id)
            logger.debug(f"Writer created/retrieved for experiment {experiment_id}: {writer is not None}")
            
            if writer is None:
                logger.warning(f"No TensorBoard writer available for experiment {experiment_id}")
                return
            
            step = self._get_step(experiment_id, metric_name, iteration)
            writer.add_scalar(metric_name, value, step)
            
            logger.info(f"Logged scalar {metric_name}={value} at step {step} for experiment {experiment_id}")
            
        except Exception as e:
            logger.error(f"Failed to log scalar {metric_name} to TensorBoard for experiment {experiment_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _log_histogram_sync(self, 
                           experiment_id: int, 
                           tag: str, 
                           values: Any, 
                           iteration: Optional[int] = None) -> None:
        """
        Log a histogram to TensorBoard.
        
        Args:
            experiment_id: Experiment ID
            tag: Tag for the histogram
            values: Values to create histogram from
            iteration: Optional explicit step/iteration number
        """
        if not self.is_enabled():
            return
        
        try:
            writer = self._get_or_create_writer(experiment_id)
            if writer is None:
                return
            
            step = self._get_step(experiment_id, tag, iteration)
            writer.add_histogram(tag, values, step)
            
            logger.debug(f"Logged histogram {tag} at step {step} for experiment {experiment_id}")
            
        except Exception as e:
            logger.error(f"Failed to log histogram {tag} to TensorBoard for experiment {experiment_id}: {e}")
    
    def _log_text_sync(self, 
                      experiment_id: int, 
                      tag: str, 
                      text: str, 
                      iteration: Optional[int] = None) -> None:
        """
        Log text to TensorBoard.
        
        Args:
            experiment_id: Experiment ID
            tag: Tag for the text
            text: Text content to log
            iteration: Optional explicit step/iteration number
        """
        if not self.is_enabled():
            return
        
        try:
            writer = self._get_or_create_writer(experiment_id)
            if writer is None:
                return
            
            step = self._get_step(experiment_id, tag, iteration)
            writer.add_text(tag, text, step)
            
            logger.debug(f"Logged text {tag} at step {step} for experiment {experiment_id}")
            
        except Exception as e:
            logger.error(f"Failed to log text {tag} to TensorBoard for experiment {experiment_id}: {e}")
    
    def log_scalar(self, 
                  experiment_id: int, 
                  metric_name: str, 
                  value: float, 
                  iteration: Optional[int] = None,
                  timestamp: Optional[datetime] = None) -> None:
        """
        Log a scalar metric to TensorBoard.
        
        Uses background logging if enabled, otherwise falls back to synchronous logging.
        
        Args:
            experiment_id: Experiment ID
            metric_name: Name of the metric
            value: Metric value
            iteration: Optional explicit step/iteration number
            timestamp: Optional timestamp (stored for compatibility but not used by TensorBoard)
        """
        if not self.is_enabled():
            return
        
        if self._background_enabled and self._metric_queue is not None:
            # Queue for background processing
            metric = TensorBoardMetric(
                experiment_id=experiment_id,
                metric_type='scalar',
                metric_name=metric_name,
                value=value,
                iteration=iteration,
                timestamp=timestamp
            )
            
            try:
                queue_timeout = self._config.get('queue_timeout', 0.01)
                self._metric_queue.put(metric, timeout=queue_timeout)
                
                with self._stats_lock:
                    self._stats['queued_metrics'] += 1
                    
                logger.debug(f"Queued scalar metric {metric_name}={value} for experiment {experiment_id}")
                
            except queue.Full:
                # Queue is full, fall back to synchronous logging
                logger.warning(f"TensorBoard queue full, falling back to sync logging for {metric_name}")
                with self._stats_lock:
                    self._stats['queue_full_count'] += 1
                    self._stats['dropped_metrics'] += 1
                
                self._log_scalar_sync(experiment_id, metric_name, value, iteration)
        else:
            # Use synchronous logging
            self._log_scalar_sync(experiment_id, metric_name, value, iteration)
    
    def log_histogram(self, 
                     experiment_id: int, 
                     tag: str, 
                     values: Any, 
                     iteration: Optional[int] = None) -> None:
        """
        Log a histogram to TensorBoard.
        
        Uses background logging if enabled, otherwise falls back to synchronous logging.
        
        Args:
            experiment_id: Experiment ID
            tag: Tag for the histogram
            values: Values to create histogram from
            iteration: Optional explicit step/iteration number
        """
        if not self.is_enabled():
            return
        
        if self._background_enabled and self._metric_queue is not None:
            # Queue for background processing
            metric = TensorBoardMetric(
                experiment_id=experiment_id,
                metric_type='histogram',
                metric_name='',  # Not used for histograms
                value=values,
                iteration=iteration,
                tag=tag
            )
            
            try:
                queue_timeout = self._config.get('queue_timeout', 0.01)
                self._metric_queue.put(metric, timeout=queue_timeout)
                
                with self._stats_lock:
                    self._stats['queued_metrics'] += 1
                    
                logger.debug(f"Queued histogram {tag} for experiment {experiment_id}")
                
            except queue.Full:
                # Queue is full, fall back to synchronous logging
                logger.warning(f"TensorBoard queue full, falling back to sync logging for histogram {tag}")
                with self._stats_lock:
                    self._stats['queue_full_count'] += 1
                    self._stats['dropped_metrics'] += 1
                
                self._log_histogram_sync(experiment_id, tag, values, iteration)
        else:
            # Use synchronous logging
            self._log_histogram_sync(experiment_id, tag, values, iteration)
    
    def log_text(self, 
                experiment_id: int, 
                tag: str, 
                text: str, 
                iteration: Optional[int] = None) -> None:
        """
        Log text to TensorBoard.
        
        Uses background logging if enabled, otherwise falls back to synchronous logging.
        
        Args:
            experiment_id: Experiment ID
            tag: Tag for the text
            text: Text content to log
            iteration: Optional explicit step/iteration number
        """
        if not self.is_enabled():
            return
        
        if self._background_enabled and self._metric_queue is not None:
            # Queue for background processing
            metric = TensorBoardMetric(
                experiment_id=experiment_id,
                metric_type='text',
                metric_name='',  # Not used for text
                value=text,
                iteration=iteration,
                tag=tag
            )
            
            try:
                queue_timeout = self._config.get('queue_timeout', 0.01)
                self._metric_queue.put(metric, timeout=queue_timeout)
                
                with self._stats_lock:
                    self._stats['queued_metrics'] += 1
                    
                logger.debug(f"Queued text {tag} for experiment {experiment_id}")
                
            except queue.Full:
                # Queue is full, fall back to synchronous logging
                logger.warning(f"TensorBoard queue full, falling back to sync logging for text {tag}")
                with self._stats_lock:
                    self._stats['queue_full_count'] += 1
                    self._stats['dropped_metrics'] += 1
                
                self._log_text_sync(experiment_id, tag, text, iteration)
        else:
            # Use synchronous logging
            self._log_text_sync(experiment_id, tag, text, iteration)
    
    def flush(self, experiment_id: Optional[int] = None, timeout: float = 5.0) -> None:
        """
        Flush TensorBoard writer(s) to ensure data is written to disk.
        
        If background logging is enabled, waits for queue to be processed first.
        
        Args:
            experiment_id: Specific experiment to flush, or None to flush all
            timeout: Maximum time to wait for queue processing (seconds)
        """
        if not self.is_enabled():
            return
        
        # If background logging is enabled, wait for queue to be processed
        if self._background_enabled and self._metric_queue is not None:
            try:
                logger.debug("Waiting for background queue to be processed...")
                start_time = time.time()
                
                while not self._metric_queue.empty() and (time.time() - start_time) < timeout:
                    time.sleep(0.01)  # Small delay to avoid busy waiting
                
                if not self._metric_queue.empty():
                    logger.warning(f"Background queue still has {self._metric_queue.qsize()} items after {timeout}s timeout")
                
            except Exception as e:
                logger.error(f"Error waiting for background queue: {e}")
        
        # Flush the actual TensorBoard writers
        with self._writers_lock:
            if experiment_id is not None:
                # Flush specific experiment
                writer = self._writers.get(experiment_id)
                if writer:
                    try:
                        writer.flush()
                        logger.debug(f"Flushed TensorBoard writer for experiment {experiment_id}")
                    except Exception as e:
                        logger.error(f"Failed to flush TensorBoard writer for experiment {experiment_id}: {e}")
            else:
                # Flush all writers
                for exp_id, writer in self._writers.items():
                    try:
                        writer.flush()
                        logger.debug(f"Flushed TensorBoard writer for experiment {exp_id}")
                    except Exception as e:
                        logger.error(f"Failed to flush TensorBoard writer for experiment {exp_id}: {e}")
    
    def close_experiment(self, experiment_id: int) -> None:
        """
        Close and cleanup TensorBoard writer for a specific experiment.
        
        Args:
            experiment_id: Experiment ID to close
        """
        with self._writers_lock:
            writer = self._writers.pop(experiment_id, None)
            if writer:
                try:
                    writer.close()
                    logger.info(f"Closed TensorBoard writer for experiment {experiment_id}")
                except Exception as e:
                    logger.error(f"Failed to close TensorBoard writer for experiment {experiment_id}: {e}")
        
        # Clean up step tracking
        with self._steps_lock:
            self._experiment_steps.pop(experiment_id, None)
    
    def close_all(self) -> None:
        """Close all TensorBoard writers and cleanup resources."""
        # Shutdown background worker first
        if self._background_enabled and self._worker_thread is not None:
            logger.info("Shutting down background worker...")
            self._shutdown_event.set()
            
            # Wait for any remaining items to be processed
            if self._metric_queue is not None:
                try:
                    # Give worker time to process remaining items
                    self._worker_thread.join(timeout=2.0)
                    
                    if self._worker_thread.is_alive():
                        logger.warning("Background worker did not shut down gracefully")
                    else:
                        logger.info("Background worker shut down successfully")
                        
                except Exception as e:
                    logger.error(f"Error shutting down background worker: {e}")
        
        # Close TensorBoard writers
        with self._writers_lock:
            for experiment_id, writer in self._writers.items():
                try:
                    writer.close()
                    logger.debug(f"Closed TensorBoard writer for experiment {experiment_id}")
                except Exception as e:
                    logger.error(f"Failed to close TensorBoard writer for experiment {experiment_id}: {e}")
            
            self._writers.clear()
        
        # Clean up step tracking
        with self._steps_lock:
            self._experiment_steps.clear()
        
        logger.info("Closed all TensorBoard writers")
    
    def get_active_experiments(self) -> list:
        """Get list of experiment IDs with active TensorBoard writers."""
        with self._writers_lock:
            return list(self._writers.keys())
    
    def configure(self, **kwargs) -> None:
        """
        Update configuration settings.
        
        Args:
            **kwargs: Configuration key-value pairs to update
        """
        self._config.update(kwargs)
        logger.debug(f"TensorBoardLogger configuration updated: {kwargs}")
    
    def get_config(self, key: str = None) -> Union[Dict[str, Any], Any]:
        """
        Get configuration value(s).
        
        Args:
            key: Specific configuration key. If None, returns all config.
            
        Returns:
            Configuration value or entire configuration dictionary
        """
        if key is None:
            return self._config.copy()
        return self._config.get(key)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for background logging.
        
        Returns:
            Dictionary containing performance metrics
        """
        with self._stats_lock:
            stats = self._stats.copy()
        
        # Add additional runtime stats
        stats['background_enabled'] = self._background_enabled
        stats['worker_alive'] = self._worker_thread is not None and self._worker_thread.is_alive()
        stats['queue_size'] = self._metric_queue.qsize() if self._metric_queue else 0
        stats['queue_maxsize'] = self._config.get('queue_size', 1000)
        
        # Calculate derived metrics
        if stats['queued_metrics'] > 0:
            stats['processing_rate'] = stats['processed_metrics'] / stats['queued_metrics']
            stats['drop_rate'] = stats['dropped_metrics'] / stats['queued_metrics']
        else:
            stats['processing_rate'] = 0.0
            stats['drop_rate'] = 0.0
        
        return stats
    
    def reset_performance_stats(self) -> None:
        """Reset performance statistics counters."""
        with self._stats_lock:
            self._stats = {
                'queued_metrics': 0,
                'processed_metrics': 0,
                'dropped_metrics': 0,
                'queue_full_count': 0,
                'worker_errors': 0
            }
        logger.info("Performance statistics reset")
    
    def is_background_logging_healthy(self) -> bool:
        """
        Check if background logging is operating normally.
        
        Returns:
            True if background logging is healthy, False otherwise
        """
        if not self._background_enabled:
            return True  # Synchronous logging is always "healthy"
        
        # Check if worker thread is alive
        if self._worker_thread is None or not self._worker_thread.is_alive():
            return False
        
        # Check for excessive errors
        with self._stats_lock:
            error_rate = self._stats['worker_errors'] / max(1, self._stats['processed_metrics'])
            if error_rate > 0.1:  # More than 10% error rate
                return False
        
        # Check queue size
        if self._metric_queue and self._metric_queue.qsize() > (self._config.get('queue_size', 1000) * 0.8):
            return False  # Queue is more than 80% full
        
        return True
    
    # ===== YINSH-SPECIFIC VISUALIZATION METHODS =====
    
    def _get_visualizer(self) -> Optional[YinshBoardVisualizer]:
        """Get or create a Yinsh board visualizer instance."""
        if not YINSH_VISUALIZATION_AVAILABLE:
            logger.warning("Yinsh visualization not available. Skipping board visualization.")
            return None
        
        if not hasattr(self, '_visualizer'):
            self._visualizer = YinshBoardVisualizer()
        
        return self._visualizer
    
    def log_board_state(self,
                       experiment_id: int,
                       board: 'Board',
                       game_state: Optional['GameState'] = None,
                       valid_moves: Optional[List['Position']] = None,
                       attention_weights: Optional[Dict[str, float]] = None,
                       iteration: Optional[int] = None,
                       tag: str = "board_state") -> None:
        """
        Log a Yinsh board state as an image to TensorBoard.
        
        Args:
            experiment_id: Experiment ID
            board: Board object with current piece positions
            game_state: Optional game state for additional context
            valid_moves: Optional list of valid move positions to highlight
            attention_weights: Optional attention weights for positions
            iteration: Optional explicit step/iteration number
            tag: Tag for the image
        """
        if not self.is_enabled():
            return
        
        visualizer = self._get_visualizer()
        if not visualizer:
            return
        
        try:
            # Create title with game info
            title_parts = [tag]
            if game_state:
                title_parts.append(f"Phase: {game_state.phase.name}")
                title_parts.append(f"Player: {game_state.current_player.name}")
            title = " | ".join(title_parts)
            
            # Render board state
            img_array = visualizer.render_board_state(
                board=board,
                game_state=game_state,
                valid_moves=valid_moves,
                attention_weights=attention_weights,
                title=title
            )
            
            # Convert to CHW format for TensorBoard (channels first)
            if len(img_array.shape) == 3:
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
            else:
                img_tensor = torch.from_numpy(img_array)
            
            # Log as image
            writer = self._get_or_create_writer(experiment_id)
            if writer:
                step = self._get_step(experiment_id, tag, iteration)
                writer.add_image(tag, img_tensor, step)
                logger.debug(f"Logged board state image for experiment {experiment_id} at step {step}")
                
        except Exception as e:
            logger.error(f"Failed to log board state for experiment {experiment_id}: {e}")
    
    def log_game_trajectory(self,
                           experiment_id: int,
                           moves: List['Move'],
                           board_states: List['Board'],
                           iteration: Optional[int] = None,
                           tag: str = "game_trajectory") -> None:
        """
        Log a game trajectory showing key moves and board states.
        
        Args:
            experiment_id: Experiment ID
            moves: List of moves in chronological order
            board_states: List of board states corresponding to each move
            iteration: Optional explicit step/iteration number
            tag: Tag for the image
        """
        if not self.is_enabled():
            return
        
        visualizer = self._get_visualizer()
        if not visualizer:
            return
        
        try:
            # Render trajectory
            img_array = visualizer.render_move_trajectory(
                moves=moves,
                board_states=board_states,
                title=f"Game Trajectory ({len(moves)} moves)"
            )
            
            # Convert to CHW format for TensorBoard
            if len(img_array.shape) == 3:
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
            else:
                img_tensor = torch.from_numpy(img_array)
            
            # Log as image
            writer = self._get_or_create_writer(experiment_id)
            if writer:
                step = self._get_step(experiment_id, tag, iteration)
                writer.add_image(tag, img_tensor, step)
                logger.debug(f"Logged game trajectory for experiment {experiment_id} at step {step}")
                
        except Exception as e:
            logger.error(f"Failed to log game trajectory for experiment {experiment_id}: {e}")
    
    def log_attention_heatmap(self,
                             experiment_id: int,
                             attention_weights: Dict[str, float],
                             board: Optional['Board'] = None,
                             iteration: Optional[int] = None,
                             tag: str = "attention_heatmap") -> None:
        """
        Log neural network attention weights as a heatmap overlay on the board.
        
        Args:
            experiment_id: Experiment ID
            attention_weights: Dictionary mapping position strings to attention values
            board: Optional board state to show pieces
            iteration: Optional explicit step/iteration number
            tag: Tag for the image
        """
        if not self.is_enabled():
            return
        
        visualizer = self._get_visualizer()
        if not visualizer:
            return
        
        try:
            # Render attention heatmap
            img_array = visualizer.render_attention_heatmap(
                attention_weights=attention_weights,
                board=board,
                title="Neural Network Attention"
            )
            
            # Convert to CHW format for TensorBoard
            if len(img_array.shape) == 3:
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
            else:
                img_tensor = torch.from_numpy(img_array)
            
            # Log as image
            writer = self._get_or_create_writer(experiment_id)
            if writer:
                step = self._get_step(experiment_id, tag, iteration)
                writer.add_image(tag, img_tensor, step)
                logger.debug(f"Logged attention heatmap for experiment {experiment_id} at step {step}")
                
        except Exception as e:
            logger.error(f"Failed to log attention heatmap for experiment {experiment_id}: {e}")
    
    def log_phase_metrics(self,
                         experiment_id: int,
                         phase_metrics: Dict[str, Dict[str, float]],
                         iteration: Optional[int] = None,
                         tag: str = "phase_analysis") -> None:
        """
        Log performance metrics broken down by game phase.
        
        Args:
            experiment_id: Experiment ID
            phase_metrics: Dictionary with phase names as keys and metric dictionaries as values
            iteration: Optional explicit step/iteration number
            tag: Tag for the image
        """
        if not self.is_enabled():
            return
        
        visualizer = self._get_visualizer()
        if not visualizer:
            return
        
        try:
            # Log individual phase metrics as scalars
            step = self._get_step(experiment_id, tag, iteration)
            writer = self._get_or_create_writer(experiment_id)
            
            if writer:
                for phase_name, metrics in phase_metrics.items():
                    for metric_name, value in metrics.items():
                        scalar_tag = f"{tag}/{phase_name}_{metric_name}"
                        writer.add_scalar(scalar_tag, value, step)
                
                # Also create a visual analysis chart
                img_array = visualizer.render_phase_analysis(
                    phase_metrics=phase_metrics,
                    title="Performance by Game Phase"
                )
                
                # Convert to CHW format for TensorBoard
                if len(img_array.shape) == 3:
                    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                else:
                    img_tensor = torch.from_numpy(img_array)
                
                writer.add_image(f"{tag}_visualization", img_tensor, step)
                logger.debug(f"Logged phase metrics for experiment {experiment_id} at step {step}")
                
        except Exception as e:
            logger.error(f"Failed to log phase metrics for experiment {experiment_id}: {e}")
    
    def log_action_probabilities(self,
                                experiment_id: int,
                                action_probs: np.ndarray,
                                valid_actions: Optional[List[int]] = None,
                                iteration: Optional[int] = None,
                                tag: str = "action_probabilities") -> None:
        """
        Log action probability distributions as histograms.
        
        Args:
            experiment_id: Experiment ID
            action_probs: Array of action probabilities
            valid_actions: Optional list of valid action indices
            iteration: Optional explicit step/iteration number
            tag: Tag for the histogram
        """
        if not self.is_enabled():
            return
        
        try:
            writer = self._get_or_create_writer(experiment_id)
            if writer:
                step = self._get_step(experiment_id, tag, iteration)
                
                # Log full probability distribution
                writer.add_histogram(f"{tag}/all_actions", action_probs, step)
                
                # Log valid action probabilities separately if provided
                if valid_actions is not None:
                    valid_probs = action_probs[valid_actions]
                    writer.add_histogram(f"{tag}/valid_actions", valid_probs, step)
                    
                    # Log some statistics
                    writer.add_scalar(f"{tag}/max_prob", float(np.max(action_probs)), step)
                    writer.add_scalar(f"{tag}/entropy", float(-np.sum(action_probs * np.log(action_probs + 1e-8))), step)
                    if len(valid_actions) > 0:
                        writer.add_scalar(f"{tag}/valid_max_prob", float(np.max(valid_probs)), step)
                
                logger.debug(f"Logged action probabilities for experiment {experiment_id} at step {step}")
                
        except Exception as e:
            logger.error(f"Failed to log action probabilities for experiment {experiment_id}: {e}")
    
    def log_value_predictions(self,
                             experiment_id: int,
                             value_preds: np.ndarray,
                             target_values: Optional[np.ndarray] = None,
                             game_phases: Optional[List[str]] = None,
                             iteration: Optional[int] = None,
                             tag: str = "value_predictions") -> None:
        """
        Log value prediction distributions and accuracy metrics.
        
        Args:
            experiment_id: Experiment ID
            value_preds: Array of value predictions
            target_values: Optional array of target values for comparison
            game_phases: Optional list of game phases for each prediction
            iteration: Optional explicit step/iteration number
            tag: Tag for the metrics
        """
        if not self.is_enabled():
            return
        
        try:
            writer = self._get_or_create_writer(experiment_id)
            if writer:
                step = self._get_step(experiment_id, tag, iteration)
                
                # Log value prediction distribution
                writer.add_histogram(f"{tag}/predictions", value_preds, step)
                
                # Log statistics
                writer.add_scalar(f"{tag}/mean_prediction", float(np.mean(value_preds)), step)
                writer.add_scalar(f"{tag}/std_prediction", float(np.std(value_preds)), step)
                
                # Log accuracy metrics if targets provided
                if target_values is not None:
                    mse = np.mean((value_preds - target_values) ** 2)
                    mae = np.mean(np.abs(value_preds - target_values))
                    
                    writer.add_scalar(f"{tag}/mse", float(mse), step)
                    writer.add_scalar(f"{tag}/mae", float(mae), step)
                    
                    # Log correlation
                    if len(value_preds) > 1:
                        correlation = np.corrcoef(value_preds, target_values)[0, 1]
                        if not np.isnan(correlation):
                            writer.add_scalar(f"{tag}/correlation", float(correlation), step)
                
                # Log phase-specific metrics if phases provided
                if game_phases is not None:
                    unique_phases = set(game_phases)
                    for phase in unique_phases:
                        phase_mask = [p == phase for p in game_phases]
                        phase_preds = value_preds[phase_mask]
                        
                        if len(phase_preds) > 0:
                            writer.add_scalar(f"{tag}/{phase}_mean", float(np.mean(phase_preds)), step)
                            
                            if target_values is not None:
                                phase_targets = target_values[phase_mask]
                                phase_mse = np.mean((phase_preds - phase_targets) ** 2)
                                writer.add_scalar(f"{tag}/{phase}_mse", float(phase_mse), step)
                
                logger.debug(f"Logged value predictions for experiment {experiment_id} at step {step}")
                
        except Exception as e:
            logger.error(f"Failed to log value predictions for experiment {experiment_id}: {e}") 