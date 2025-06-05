"""Optimized trainer with zero-copy tensor operations."""

from typing import Dict, List, Tuple, Optional, Union
import logging
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

from ..memory.zero_copy import (
    ZeroCopyConfig,
    create_batch_from_numpy,
    get_persistent_buffer,
    release_buffer,
    safe_add_, safe_mul_, safe_copy_,
    zero_copy_context,
    get_zero_copy_statistics
)
from ..network.wrapper import NetworkWrapper
from .trainer import GameExperience  # Import the existing experience class


class ZeroCopyGameExperience(GameExperience):
    """Enhanced GameExperience with zero-copy batch formation."""
    
    def __init__(self, max_size: int = 10000, subsample_long_games: bool = True):
        """Initialize with zero-copy configuration."""
        super().__init__(max_size, subsample_long_games)
        
        # Configure zero-copy for training workloads
        self.zero_copy_config = ZeroCopyConfig(
            enable_shared_memory_tensors=True,
            enable_persistent_buffers=True,
            enable_inplace_operations=True,
            inplace_threshold_mb=0.5,  # Lower threshold for training tensors
            max_buffer_pool_size_mb=256
        )
    
    def sample_batch_zero_copy(self,
                              batch_size: int,
                              phase_weights: Optional[Dict[str, float]] = None,
                              device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch using zero-copy operations.
        
        Args:
            batch_size: Number of samples in the batch
            phase_weights: Optional phase-based sampling weights
            device: Target device for the batch
            
        Returns:
            Tuple of (states, move_probs, values) tensors
        """
        if len(self.states) < batch_size:
            # Fallback to original method for small batches
            return self.sample_batch(batch_size, phase_weights)
        
        with zero_copy_context(self.zero_copy_config):
            # Get sample indices (reuse existing logic)
            n = len(self.states)
            if phase_weights:
                # Apply phase weighting
                p = np.ones(n, dtype=np.float32)
                for i, phase in enumerate(self.phases):
                    p[i] *= phase_weights.get(phase, 1.0)
                p = p / p.sum()
                indices = np.random.choice(n, batch_size, replace=False, p=p)
            else:
                indices = np.random.choice(n, batch_size, replace=False)
            
            # Create batches using zero-copy operations
            try:
                # Collect numpy arrays
                state_arrays = [self.states[i] for i in indices]
                prob_arrays = [self.move_probs[i] for i in indices]
                value_list = [self.values[i] for i in indices]
                
                # Create batched tensors with minimal copying
                states = create_batch_from_numpy(state_arrays, torch.float32, device)
                probs = create_batch_from_numpy(prob_arrays, torch.float32, device)
                
                # Handle values separately (they're scalars)
                values = torch.tensor(value_list, dtype=torch.float32, device=device).unsqueeze(1)
                
                return states, probs, values
                
            except Exception as e:
                logging.warning(f"Zero-copy batch creation failed: {e}, falling back to standard method")
                return self.sample_batch(batch_size, phase_weights)


class OptimizedYinshTrainer:
    """Enhanced trainer with zero-copy tensor operations."""
    
    def __init__(self,
                 network: NetworkWrapper,
                 device: Optional[str] = None,
                 batch_size: int = 256,
                 l2_reg: float = 0.0,
                 metrics_logger=None,
                 value_head_lr_factor: float = 5.0,
                 value_loss_weights: Tuple[float, float] = (0.5, 0.5),
                 replay_buffer_path: Optional[str] = None,
                 enable_zero_copy: bool = True):
        """
        Initialize the optimized trainer.
        
        Args:
            network: NetworkWrapper instance
            device: Device to train on
            batch_size: Batch size for training
            l2_reg: L2 regularization coefficient
            metrics_logger: Optional MetricsLogger instance
            value_head_lr_factor: Factor to multiply base lr for value head
            value_loss_weights: Weights for combining MSE and CE loss
            replay_buffer_path: Path to replay buffer file
            enable_zero_copy: Whether to enable zero-copy optimizations
        """
        self.batch_size = batch_size
        self.enable_zero_copy = enable_zero_copy
        
        # Device setup
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else (
                    "mps" if torch.backends.mps.is_available() else "cpu"
                )
            )
        
        self.network = network
        self.network.network = self.network.network.to(self.device)
        self.metrics_logger = metrics_logger
        
        # Configure zero-copy operations
        if self.enable_zero_copy:
            self.zero_copy_config = ZeroCopyConfig(
                enable_shared_memory_tensors=False,  # Keep disabled for GPU tensors
                enable_persistent_buffers=True,
                enable_inplace_operations=True,
                inplace_threshold_mb=1.0,
                max_buffer_pool_size_mb=512
            )
            self.experience = ZeroCopyGameExperience()
        else:
            from .trainer import GameExperience
            self.experience = GameExperience()
        
        # Loss computation buffers for reuse
        self._loss_buffers = {}
        
        # Training configuration
        self.value_loss_weights = value_loss_weights
        self.value_head_lr_factor = value_head_lr_factor
        
        # Initialize optimizers (same as original)
        value_params = [p for n, p in self.network.network.named_parameters() if 'value_head' in n]
        policy_params = [p for n, p in self.network.network.named_parameters() if 'value_head' not in n]
        
        self.policy_optimizer = optim.Adam(policy_params, lr=0.001, weight_decay=1e-4)
        self.value_optimizer = optim.SGD(
            value_params,
            lr=0.0001 * value_head_lr_factor,
            momentum=0.9,
            weight_decay=1e-3
        )
        
        # Schedulers
        self.policy_scheduler = torch.optim.lr_scheduler.CyclicLR(
            self.policy_optimizer,
            base_lr=1e-5,
            max_lr=1e-4,
            step_size_up=500,
            mode='triangular2',
            cycle_momentum=False
        )
        
        self.value_scheduler = optim.lr_scheduler.CyclicLR(
            self.value_optimizer,
            base_lr=1e-5,
            max_lr=1e-4,
            step_size_up=500,
            mode='triangular2',
            cycle_momentum=True
        )
        
        # Training state
        self.current_iteration = 0
        self.l2_reg = l2_reg
        self.logger = logging.getLogger("OptimizedYinshTrainer")
        self.logger.setLevel(logging.INFO)
        
        # Metrics tracking
        self.policy_losses = []
        self.value_losses = []
        self.total_losses = []
        self.learning_rates = {'policy': [], 'value': []}
        self.value_accuracies = []
        self.move_accuracies = []
        self.temperature = 1.0
        
        # Load replay buffer if specified
        if replay_buffer_path is not None:
            from os import path as osp
            if osp.exists(replay_buffer_path):
                self.experience.load_buffer(replay_buffer_path)
            else:
                print(f"[Replay Buffer] File '{replay_buffer_path}' not found. Starting with empty buffer.")
    
    def _get_or_create_buffer(self, key: str, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """Get or create a persistent buffer for loss computation."""
        if not self.enable_zero_copy:
            return torch.zeros(shape, dtype=dtype, device=self.device)
        
        if key not in self._loss_buffers:
            self._loss_buffers[key] = get_persistent_buffer(shape, dtype, self.device)
        
        buffer = self._loss_buffers[key]
        
        # Resize buffer if needed
        if buffer.shape != shape:
            release_buffer(buffer)
            self._loss_buffers[key] = get_persistent_buffer(shape, dtype, self.device)
            buffer = self._loss_buffers[key]
        
        return buffer
    
    def _compute_policy_loss_optimized(self, 
                                     pred_logits: torch.Tensor,
                                     target_probs: torch.Tensor) -> torch.Tensor:
        """Compute policy loss with optimized tensor operations."""
        batch_size = pred_logits.size(0)
        
        if self.enable_zero_copy:
            # Use persistent buffers for intermediate computations
            log_probs_buffer = self._get_or_create_buffer(
                'log_probs', pred_logits.shape, pred_logits.dtype
            )
            
            # Compute log probabilities in-place
            F.log_softmax(pred_logits, dim=1, out=log_probs_buffer)
            
            # Compute cross-entropy loss using in-place operations
            loss_buffer = self._get_or_create_buffer(
                'policy_loss', (batch_size,), pred_logits.dtype
            )
            
            # Manual cross-entropy: -sum(target * log_pred)
            torch.sum(target_probs * log_probs_buffer, dim=1, out=loss_buffer)
            safe_mul_(loss_buffer, -1.0)
            
            return loss_buffer.mean()
        else:
            # Standard computation
            log_probs = F.log_softmax(pred_logits, dim=1)
            return -(target_probs * log_probs).sum(dim=1).mean()
    
    def _compute_value_loss_optimized(self,
                                    pred_values: torch.Tensor,
                                    target_values: torch.Tensor) -> torch.Tensor:
        """Compute value loss with optimized tensor operations."""
        if self.enable_zero_copy:
            # Use buffers for intermediate computations
            batch_size = pred_values.size(0)
            
            # MSE loss component
            mse_diff_buffer = self._get_or_create_buffer(
                'mse_diff', pred_values.shape, pred_values.dtype
            )
            
            # Compute (pred - target) in-place
            safe_copy_(mse_diff_buffer, pred_values)
            safe_add_(mse_diff_buffer, target_values, alpha=-1.0)  # diff = pred - target
            safe_mul_(mse_diff_buffer, mse_diff_buffer)  # square in-place
            
            mse_loss = mse_diff_buffer.mean()
            
            # BCE loss component
            value_probs = torch.sigmoid(pred_values)
            target_outcomes = (target_values > 0).float()
            bce_loss = F.binary_cross_entropy(value_probs, target_outcomes)
            
            # Combine losses
            return (self.value_loss_weights[0] * mse_loss + 
                   self.value_loss_weights[1] * bce_loss)
        else:
            # Standard computation
            mse_loss = F.mse_loss(pred_values, target_values)
            value_probs = torch.sigmoid(pred_values)
            target_outcomes = (target_values > 0).float()
            bce_loss = F.binary_cross_entropy(value_probs, target_outcomes)
            
            return (self.value_loss_weights[0] * mse_loss + 
                   self.value_loss_weights[1] * bce_loss)
    
    def train_step_optimized(self,
                           batch_size: int,
                           phase_weights: Optional[Dict[str, float]] = None) -> Tuple[float, float, float, Dict]:
        """Optimized training step with zero-copy operations."""
        if len(self.experience.states) < batch_size:
            return 0.0, 0.0, 0.0, {'top_1_accuracy': 0.0, 'top_3_accuracy': 0.0, 'top_5_accuracy': 0.0}
        
        self.network.network.train()
        
        # Sample batch with zero-copy if enabled
        if self.enable_zero_copy and hasattr(self.experience, 'sample_batch_zero_copy'):
            states, target_probs, target_values = self.experience.sample_batch_zero_copy(
                batch_size, phase_weights, self.device
            )
        else:
            states, target_probs, target_values = self.experience.sample_batch(batch_size, phase_weights)
            states = states.to(self.device)
            target_probs = target_probs.to(self.device)
            target_values = target_values.to(self.device)
        
        # Forward pass
        pred_logits, pred_values = self.network.network(states)
        
        # Compute losses with optimizations
        policy_loss = self._compute_policy_loss_optimized(pred_logits, target_probs)
        value_loss = self._compute_value_loss_optimized(pred_values, target_values)
        
        # L2 regularization (if enabled)
        l2_loss = 0.0
        if self.l2_reg > 0:
            for param in self.network.network.parameters():
                l2_loss += torch.norm(param, 2) ** 2
            l2_loss *= self.l2_reg
        
        total_loss = policy_loss + value_loss + l2_loss
        
        # Backward pass and optimization
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.network.network.parameters(), max_norm=1.0)
        
        self.policy_optimizer.step()
        self.value_optimizer.step()
        
        # Update learning rate schedulers
        self.policy_scheduler.step()
        self.value_scheduler.step()
        
        # Compute accuracies (reuse existing logic)
        with torch.no_grad():
            accuracies = self._calculate_accuracies(pred_values, target_values, pred_logits, target_probs)
        
        # Update metrics
        self.policy_losses.append(policy_loss.item())
        self.value_losses.append(value_loss.item())
        self.total_losses.append(total_loss.item())
        self.current_iteration += 1
        
        return policy_loss.item(), value_loss.item(), total_loss.item(), accuracies
    
    def _calculate_accuracies(self,
                            pred_values: torch.Tensor,
                            target_values: torch.Tensor,
                            pred_logits: torch.Tensor,
                            target_probs: torch.Tensor) -> Dict:
        """Calculate training accuracies (reused from original trainer)."""
        # Value accuracy
        pred_outcomes = (pred_values > 0).float()
        true_outcomes = (target_values > 0).float()
        value_accuracy = float((pred_outcomes == true_outcomes).float().mean().item())
        
        # Move accuracy
        pred_moves = torch.argmax(pred_logits, dim=1)
        true_moves = torch.argmax(target_probs, dim=1)
        move_accuracy = float((pred_moves == true_moves).float().mean().item())
        
        # Top-k accuracies
        _, top3_pred = torch.topk(pred_logits, 3, dim=1)
        _, top5_pred = torch.topk(pred_logits, 5, dim=1)
        
        top3_acc = float(torch.any(top3_pred == true_moves.unsqueeze(1), dim=1).float().mean().item())
        top5_acc = float(torch.any(top5_pred == true_moves.unsqueeze(1), dim=1).float().mean().item())
        
        return {
            'value_accuracy': value_accuracy,
            'top_1_accuracy': move_accuracy,
            'top_3_accuracy': top3_acc,
            'top_5_accuracy': top5_acc
        }
    
    def get_zero_copy_statistics(self) -> Dict:
        """Get zero-copy optimization statistics."""
        if self.enable_zero_copy:
            stats = get_zero_copy_statistics()
            return {
                'zero_copy_enabled': True,
                'tensor_factory_stats': {
                    'shared_memory_allocations': stats['tensor_factory'].shared_memory_allocations,
                    'buffer_reuses': stats['tensor_factory'].buffer_reuses,
                    'copy_avoided_count': stats['tensor_factory'].copy_avoided_count,
                    'fallback_copies': stats['tensor_factory'].fallback_copies
                },
                'inplace_stats': {
                    'inplace_operations': stats['inplace_operations'].inplace_operations,
                    'failed_inplace_operations': stats['inplace_operations'].failed_inplace_operations,
                    'copy_avoided_count': stats['inplace_operations'].copy_avoided_count
                },
                'batch_processor_stats': {
                    'copy_avoided_count': stats['batch_processor'].copy_avoided_count,
                    'fallback_copies': stats['batch_processor'].fallback_copies
                }
            }
        else:
            return {'zero_copy_enabled': False}
    
    def cleanup_buffers(self):
        """Clean up persistent buffers."""
        if self.enable_zero_copy:
            for buffer in self._loss_buffers.values():
                release_buffer(buffer)
            self._loss_buffers.clear()
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.cleanup_buffers()
        except:
            pass
    
    # Delegate other methods to maintain compatibility
    def add_game_experience(self, states: list, policies: list, outcome, discount_factor: float = 1.0):
        """Add game experience (delegates to experience buffer)."""
        return self.experience.add_game_experience(states, policies, outcome, discount_factor)
    
    def save_checkpoint(self, path: str, epoch: int):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.network.network.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'policy_scheduler_state_dict': self.policy_scheduler.state_dict(),
            'value_scheduler_state_dict': self.value_scheduler.state_dict(),
            'current_iteration': self.current_iteration,
            'zero_copy_stats': self.get_zero_copy_statistics() if self.enable_zero_copy else None
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> int:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.network.network.load_state_dict(checkpoint['model_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.policy_scheduler.load_state_dict(checkpoint['policy_scheduler_state_dict'])
        self.value_scheduler.load_state_dict(checkpoint['value_scheduler_state_dict'])
        self.current_iteration = checkpoint.get('current_iteration', 0)
        
        return checkpoint['epoch'] 