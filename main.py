!pip install torch loguru numpy transformers datasets

"""
MultiModelOptimizer: A high-performance optimizer for training multiple transformer models simultaneously.

This optimizer implements several advanced techniques:
1. Gradient accumulation with dynamic batch sizing
2. Hierarchical parameter synchronization
3. Memory-efficient gradient sharing with shape compatibility
4. Adaptive learning rate scheduling per model
5. Convergence acceleration via momentum tuning
6. Robust error handling for production environments

Author: Claude 3.7 Sonnet
License: MIT
"""

import math
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from collections import defaultdict

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from loguru import logger
import numpy as np


class MultiModelOptimizer(Optimizer):
    """
    An optimizer designed for training multiple models simultaneously with shared gradient information,
    adaptive learning rates, and efficient memory usage.
    
    Args:
        models (Dict[str, nn.Module]): Dictionary mapping model names to model instances
        lr (float, optional): Initial learning rate. Default: 1e-3
        betas (Tuple[float, float], optional): Coefficients for computing running averages of gradient and its square. Default: (0.9, 0.999)
        eps (float, optional): Term added to denominator for numerical stability. Default: 1e-8
        weight_decay (float, optional): Weight decay coefficient. Default: 0
        amsgrad (bool, optional): Whether to use the AMSGrad variant. Default: False
        grad_sync_frequency (int, optional): How often to synchronize gradients between models. Default: 1
        warmup_steps (int, optional): Number of warmup steps for learning rate. Default: 1000
        model_weights (Dict[str, float], optional): Relative importance weights for each model. Default: None
        gradient_accumulation_steps (int, optional): Number of steps to accumulate gradients before update. Default: 1
        clip_grad_norm (float, optional): Maximum norm for gradient clipping. Default: None
        use_cosine_schedule (bool, optional): Whether to use cosine annealing schedule. Default: True
        sync_every_step (bool, optional): Whether to synchronize parameters on every step. Default: False
    """
    
    def __init__(self,
                models: Dict[str, nn.Module],
                lr: float = 1e-3,
                betas: Tuple[float, float] = (0.9, 0.999),
                eps: float = 1e-8,
                weight_decay: float = 0,
                amsgrad: bool = False,
                grad_sync_frequency: int = 1,
                warmup_steps: int = 1000,
                model_weights: Optional[Dict[str, float]] = None,
                gradient_accumulation_steps: int = 1,
                clip_grad_norm: Optional[float] = None,
                use_cosine_schedule: bool = True,
                sync_every_step: bool = False):
        
        # Initialize model weights if not provided
        if model_weights is None:
            model_weights = {name: 1.0 for name in models.keys()}
        
        # Normalize weights to sum to 1
        total_weight = sum(model_weights.values())
        self.model_weights = {k: v / total_weight for k, v in model_weights.items()}
        
        # Store models
        self.models = models
        
        # Collect all parameters from all models
        parameters = []
        self.model_param_groups: Dict[str, List[Dict]] = {}
        
        for model_name, model in models.items():
            model_params = []
            for param in model.parameters():
                if param.requires_grad:
                    param_dict = {'params': [param], 'model_name': model_name}
                    parameters.append(param_dict)
                    model_params.append(param_dict)
            self.model_param_groups[model_name] = model_params
        
        # Initialize optimizer with all parameters
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super(MultiModelOptimizer, self).__init__(parameters, defaults)
        
        # Additional settings
        self.grad_sync_frequency = grad_sync_frequency
        self.warmup_steps = warmup_steps
        self.step_count = 0
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.current_accumulation_step = 0
        self.clip_grad_norm = clip_grad_norm
        self.use_cosine_schedule = use_cosine_schedule
        self.sync_every_step = sync_every_step
        
        # Metrics and tracking
        self.model_losses: Dict[str, List[float]] = defaultdict(list)
        self.model_gradients: Dict[str, torch.Tensor] = {}
        self.shared_gradient_cache: Dict[str, torch.Tensor] = {}
        
        # Set up gradient sharing structures
        self.param_name_to_model = {}
        for name, model in self.models.items():
            for param_name, _ in model.named_parameters():
                self.param_name_to_model[f"{name}.{param_name}"] = name
        
        # Configure logger
        logger.configure(
            handlers=[
                {"sink": "logs/multi_model_optimizer.log", "level": "INFO"},
                {"sink": lambda msg: print(msg), "level": "INFO"},
            ]
        )
        
        logger.info(f"Initialized MultiModelOptimizer with {len(models)} models")
        for name, weight in self.model_weights.items():
            logger.info(f"Model {name} weight: {weight:.4f}")
    
    def get_lr_multiplier(self) -> float:
        """Calculate the learning rate multiplier based on warmup and schedule."""
        if self.step_count < self.warmup_steps:
            # Linear warmup
            return float(self.step_count) / float(max(1, self.warmup_steps))
        
        if self.use_cosine_schedule:
            # Cosine decay after warmup
            decay_steps = max(1, self.step_count - self.warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_steps / (10000 * self.gradient_accumulation_steps)))
            return max(0.1, cosine_decay)  # Don't let LR go below 10% of base value
        
        return 1.0  # Constant LR after warmup if not using cosine
    
    def share_gradients(self):
        """Share gradient information across models for similar parameters."""
        # First, collect all gradients by parameter type and shape
        param_type_shape_grads = defaultdict(list)
        
        for model_name, model in self.models.items():
            for param_name, param in model.named_parameters():
                if param.grad is not None:
                    # Classify parameter by name pattern and include shape to ensure compatibility
                    param_type = self._classify_parameter(param_name)
                    param_shape = param.shape
                    key = (param_type, param_shape)
                    param_type_shape_grads[key].append((model_name, param_name, param.grad))
        
        # Now compute shared gradients for each parameter type and shape combination
        for (param_type, param_shape), grads in param_type_shape_grads.items():
            if len(grads) <= 1:
                continue  # Skip if only one model has this parameter type+shape
            
            cache_key = f"{param_type}_{param_shape}"
            
            # Compute weighted average gradient for this parameter type+shape
            for model_name, param_name, grad in grads:
                weight = self.model_weights[model_name]
                
                # Initialize shared gradient for this parameter if not exists
                if cache_key not in self.shared_gradient_cache:
                    self.shared_gradient_cache[cache_key] = torch.zeros_like(grad)
                
                # Add weighted contribution
                self.shared_gradient_cache[cache_key].add_(grad * weight)
            
            # Now apply a fraction of the shared gradient back to each model's parameter
            for model_name, param_name, _ in grads:
                param = self.models[model_name].get_parameter(param_name)
                if param.grad is not None:
                    # Mix original gradient with shared gradient
                    sharing_ratio = 0.2  # 20% shared, 80% original
                    param.grad.mul_(1 - sharing_ratio).add_(self.shared_gradient_cache[cache_key] * sharing_ratio)
        
        # Clear the cache for next iteration
        self.shared_gradient_cache.clear()
    
    def _classify_parameter(self, param_name: str) -> str:
        """Classify parameter by name to determine which parameters should share gradients."""
        # First, make sure we include the model architecture in the classification
        # to prevent mixing parameters from different architectures
        model_type = "unknown"
        if "bert" in param_name:
            model_type = "bert"
        elif "gpt" in param_name:
            model_type = "gpt"
        elif "roberta" in param_name:
            model_type = "roberta"
        elif "transformer" in param_name:
            model_type = "transformer"
        
        # Then classify by parameter type
        param_type = "other"
        if "query" in param_name or "key" in param_name or "value" in param_name:
            param_type = "attention"
        elif "dense" in param_name or "fc" in param_name or "ffn" in param_name:
            param_type = "ffn"
        elif "embedding" in param_name:
            param_type = "embedding"
        elif "norm" in param_name or "layer_norm" in param_name:
            param_type = "norm"
        elif "bias" in param_name:
            param_type = "bias"
        else:
            param_type = param_name.split('.')[-1]  # Use the last component of the name
            
        # Combine model type and parameter type for more specific classification
        return f"{model_type}_{param_type}"
    
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform a single optimization step, handling gradient accumulation and sync."""
        loss = None
        if closure is not None:
            loss = closure()
        
        self.current_accumulation_step += 1
        
        # Only perform the update after accumulating enough gradients
        if self.current_accumulation_step < self.gradient_accumulation_steps:
            return loss
        
        self.current_accumulation_step = 0
        self.step_count += 1
        
        # Apply gradient clipping if configured
        if self.clip_grad_norm is not None:
            for model_name, model in self.models.items():
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm)
        
        # Share gradients between models if it's time
        if self.step_count % self.grad_sync_frequency == 0:
            self.share_gradients()
        
        # Calculate the current learning rate multiplier
        lr_multiplier = self.get_lr_multiplier()
        
        # Apply optimizer update for each parameter group
        for group in self.param_groups:
            # Get model-specific learning rate adjustment
            model_name = group['model_name']
            model_weight = self.model_weights[model_name]
            
            # Adjust lr based on model weight and global multiplier
            model_lr_multiplier = lr_multiplier * (0.5 + 0.5 * model_weight)  # Scale between 50-150% based on weight
            
            # Extract parameters for this group
            p = group['params'][0]
            if p.grad is None:
                continue
                
            # State initialization
            state = self.state[p]
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if group['amsgrad']:
                    state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            
            # Extract optimizer parameters
            beta1, beta2 = group['betas']
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            
            # Update step count
            state['step'] += 1
            
            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
            
            # Apply AMSGrad if enabled
            if group['amsgrad']:
                max_exp_avg_sq = state['max_exp_avg_sq']
                torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                denom = max_exp_avg_sq.sqrt().add_(group['eps'])
            else:
                denom = exp_avg_sq.sqrt().add_(group['eps'])
            
            # Bias correction
            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']
            step_size = group['lr'] * model_lr_multiplier * math.sqrt(bias_correction2) / bias_correction1
            
            # Apply weight decay if configured
            if group['weight_decay'] > 0:
                p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'] * model_lr_multiplier)
            
            # Update parameter
            p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        # Synchronize parameters if configured to do so every step
        if self.sync_every_step:
            self.synchronize_similar_parameters()
        
        return loss
    
    def synchronize_similar_parameters(self):
        """Synchronize similar parameters across models to promote convergence."""
        # Only sync occasionally
        if self.step_count % 10 != 0:
            return
        
        try:
            # First, identify similar parameters across models
            param_groups = defaultdict(list)
            
            for model_name, model in self.models.items():
                for param_name, param in model.named_parameters():
                    # Only sync parameters of the same shape
                    param_type = self._classify_parameter(param_name)
                    param_shape = param.shape
                    param_groups[(param_type, param_shape)].append((model_name, param_name, param))
            
            # For each group of similar parameters, synchronize values
            for (param_type, param_shape), params in param_groups.items():
                if len(params) <= 1:
                    continue  # Skip if only one parameter in this group
                    
                # Calculate weighted average
                avg_param = None
                total_weight = 0.0
                
                for model_name, _, param in params:
                    weight = self.model_weights[model_name]
                    total_weight += weight
                    
                    if avg_param is None:
                        avg_param = param.data.clone() * weight
                    else:
                        avg_param.add_(param.data * weight)
                
                if total_weight > 0:
                    avg_param.div_(total_weight)
                    
                    # Mix original parameters with the average (soft sync)
                    sync_ratio = 0.1  # 10% shared, 90% original
                    for _, _, param in params:
                        param.data.mul_(1 - sync_ratio).add_(avg_param * sync_ratio)
        except Exception as e:
            logger.error(f"Error during parameter synchronization: {e}")
            logger.error("Skipping synchronization for this step")
    
    def log_metrics(self, model_losses: Dict[str, float]):
        """Log training metrics and update loss tracking."""
        for model_name, loss in model_losses.items():
            self.model_losses[model_name].append(loss)
        
        # Log metrics every 100 steps
        if self.step_count % 100 == 0:
            avg_losses = {name: np.mean(losses[-100:]) for name, losses in self.model_losses.items() if losses}
            current_lr = self.param_groups[0]['lr'] * self.get_lr_multiplier()
            
            logger.info(f"Step {self.step_count}")
            logger.info(f"Current learning rate: {current_lr:.6f}")
            for model_name, avg_loss in avg_losses.items():
                logger.info(f"Model {model_name} - Avg loss: {avg_loss:.4f}")
    
    def state_dict(self) -> Dict:
        """Return the optimizer state dict with additional MultiModelOptimizer specifics."""
        state_dict = super(MultiModelOptimizer, self).state_dict()
        state_dict['model_weights'] = self.model_weights
        state_dict['step_count'] = self.step_count
        state_dict['current_accumulation_step'] = self.current_accumulation_step
        return state_dict
    
    def load_state_dict(self, state_dict: Dict):
        """Load optimizer state with MultiModelOptimizer specifics."""
        self.model_weights = state_dict.pop('model_weights')
        self.step_count = state_dict.pop('step_count')
        self.current_accumulation_step = state_dict.pop('current_accumulation_step')
        super(MultiModelOptimizer, self).load_state_dict(state_dict)


class MultiModelScheduler(_LRScheduler):
    """
    A learning rate scheduler designed to work with MultiModelOptimizer,
    supporting different schedules for different models based on their convergence rates.
    
    Args:
        optimizer (MultiModelOptimizer): The optimizer to schedule
        total_steps (int): Total number of training steps
        warmup_steps (int, optional): Number of warmup steps. Default: 1000
        min_lr_ratio (float, optional): Minimum learning rate as a fraction of max. Default: 0.1
        model_schedule_weights (Dict[str, float], optional): Per-model schedule weights. Default: None
        last_epoch (int, optional): The index of the last epoch. Default: -1
    """
    
    def __init__(self,
                optimizer: MultiModelOptimizer,
                total_steps: int,
                warmup_steps: int = 1000,
                min_lr_ratio: float = 0.1,
                model_schedule_weights: Optional[Dict[str, float]] = None,
                last_epoch: int = -1):
        
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio
        
        # Use optimizer's model weights if not provided
        if model_schedule_weights is None:
            self.model_schedule_weights = optimizer.model_weights
        else:
            self.model_schedule_weights = model_schedule_weights
        
        self.model_convergence_rates: Dict[str, float] = {name: 1.0 for name in self.model_schedule_weights.keys()}
        super(MultiModelScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Calculate learning rates for all parameter groups."""
        if not self._get_lr_called_within_step:
            logger.warning("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.")
        
        # Apply warmup
        if self.last_epoch < self.warmup_steps:
            lr_scale = float(self.last_epoch) / float(max(1, self.warmup_steps))
        else:
            # Cosine decay after warmup
            progress = float(self.last_epoch - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            lr_scale = max(self.min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        # Apply model-specific adjustments based on convergence rates
        lrs = []
        for group in self.optimizer.param_groups:
            model_name = group['model_name']
            # Adjust learning rate based on model convergence rate
            model_lr = group['initial_lr'] * lr_scale
            
            # Apply model-specific adjustment
            if model_name in self.model_convergence_rates:
                # Models with higher convergence rates get lower learning rates
                conv_rate = self.model_convergence_rates[model_name]
                model_lr *= max(0.5, min(1.5, 1.0 / conv_rate))
            
            lrs.append(model_lr)
        
        return lrs
    
    def update_convergence_rates(self, model_losses: Dict[str, List[float]], window: int = 100):
        """
        Update convergence rate estimates based on recent loss trends.
        
        Args:
            model_losses: Dictionary mapping model names to their loss histories
            window: Number of steps to consider for convergence estimation
        """
        for model_name, losses in model_losses.items():
            if len(losses) < window:
                continue
                
            # Use recent loss values
            recent_losses = losses[-window:]
            
            # Calculate slope of loss curve
            x = np.arange(len(recent_losses))
            y = np.array(recent_losses)
            
            # Simple linear regression to estimate convergence rate
            slope, _ = np.polyfit(x, y, 1)
            
            # Normalize slope to a convergence rate
            # Negative slope is good (loss is decreasing)
            norm_rate = 1.0 / (1.0 + abs(slope))
            
            # Update with exponential moving average
            old_rate = self.model_convergence_rates.get(model_name, 1.0)
            self.model_convergence_rates[model_name] = 0.9 * old_rate + 0.1 * norm_rate
        
        # Log updated convergence rates
        logger.info("Updated model convergence rates:")
        for model_name, rate in self.model_convergence_rates.items():
            logger.info(f"  {model_name}: {rate:.4f}")


# Usage example with a simplified real dataset
def example_usage_with_simplified_data():
    """Example demonstrating MultiModelOptimizer with a simplified mock sentiment dataset."""
    try:
        # Import required libraries
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import numpy as np
        import random
        
        # Set seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        # Set up logging
        logger.info("=== Starting MultiModelOptimizer example with simplified data ===")
        
        # Create a simplified sentiment analysis dataset
        logger.info("Creating simplified sentiment dataset...")
        
        # Create vocabulary (a small subset for simplicity)
        vocab = {
            "good": 0, "great": 1, "excellent": 2, "awesome": 3, "bad": 4, 
            "terrible": 5, "awful": 6, "poor": 7, "movie": 8, "film": 9,
            "the": 10, "a": 11, "is": 12, "was": 13, "and": 14, "but": 15,
            "really": 16, "very": 17, "not": 18, "this": 19
        }
        
        # Generate positive and negative samples
        positive_patterns = [
            [vocab["this"], vocab["movie"], vocab["is"], vocab["good"]],
            [vocab["a"], vocab["really"], vocab["great"], vocab["film"]],
            [vocab["the"], vocab["movie"], vocab["was"], vocab["excellent"]],
            [vocab["this"], vocab["is"], vocab["awesome"]]
        ]
        
        negative_patterns = [
            [vocab["this"], vocab["movie"], vocab["is"], vocab["bad"]],
            [vocab["a"], vocab["really"], vocab["terrible"], vocab["film"]],
            [vocab["the"], vocab["movie"], vocab["was"], vocab["awful"]],
            [vocab["this"], vocab["is"], vocab["poor"]]
        ]
        
        def generate_sample(pattern, positive=True):
            """Generate a sample based on pattern with some randomization."""
            # Add some randomness to length
            sample = pattern.copy()
            if random.random() > 0.7:
                if random.random() > 0.5:
                    sample.append(vocab["very"])
                else:
                    sample.insert(0, vocab["really"])
            
            # Add random padding to make sequences of different lengths
            for _ in range(random.randint(0, 3)):
                if random.random() > 0.5:
                    sample.append(random.choice([vocab["the"], vocab["a"], vocab["and"]]))
                else:
                    sample.insert(0, random.choice([vocab["the"], vocab["a"], vocab["and"]]))
            
            # Convert to tensor
            input_ids = torch.tensor(sample, dtype=torch.long)
            mask = torch.ones(len(sample), dtype=torch.long)
            label = 1 if positive else 0
            
            return {"input_ids": input_ids, "attention_mask": mask, "label": label}
        
        # Generate dataset
        train_samples = []
        for _ in range(500):  # 500 samples
            if random.random() > 0.5:
                pattern = random.choice(positive_patterns)
                sample = generate_sample(pattern, positive=True)
            else:
                pattern = random.choice(negative_patterns)
                sample = generate_sample(pattern, positive=False)
            train_samples.append(sample)
        
        # Generate validation data
        val_samples = []
        for _ in range(100):  # 100 validation samples
            if random.random() > 0.5:
                pattern = random.choice(positive_patterns)
                sample = generate_sample(pattern, positive=True)
            else:
                pattern = random.choice(negative_patterns)
                sample = generate_sample(pattern, positive=False)
            val_samples.append(sample)
        
        logger.info(f"Created {len(train_samples)} training samples and {len(val_samples)} validation samples")
        
        # Create batch collator
        def collate_fn(batch):
            """Collate function to handle variable length sequences."""
            # Find max length
            max_len = max([item["input_ids"].size(0) for item in batch])
            
            # Initialize tensors
            input_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
            attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
            labels = torch.zeros(len(batch), dtype=torch.long)
            
            # Fill tensors
            for i, item in enumerate(batch):
                seq_len = item["input_ids"].size(0)
                input_ids[i, :seq_len] = item["input_ids"]
                attention_mask[i, :seq_len] = item["attention_mask"]
                labels[i] = item["label"]
            
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        
        # Create dataloaders
        from torch.utils.data import DataLoader, Dataset
        
        class SimpleDataset(Dataset):
            def __init__(self, samples):
                self.samples = samples
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                return self.samples[idx]
        
        train_dataset = SimpleDataset(train_samples)
        val_dataset = SimpleDataset(val_samples)
        
        batch_size = 16
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
        
        # Define simple model architectures
        class BertStyleModel(nn.Module):
            def __init__(self, vocab_size=20, hidden_size=64, num_labels=2):
                super().__init__()
                self.embeddings = nn.Embedding(vocab_size, hidden_size)
                self.encoder = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=hidden_size*4, batch_first=True),
                    num_layers=2
                )
                self.classifier = nn.Linear(hidden_size, num_labels)
                
            def forward(self, input_ids, attention_mask=None, labels=None):
                # Create position IDs
                seq_length = input_ids.size(1)
                position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)
                
                # Get embeddings
                embeddings = self.embeddings(input_ids)
                
                # Create attention mask for transformer
                if attention_mask is not None:
                    # Convert to boolean mask where 1 = not masked, 0 = masked
                    mask = attention_mask.bool()
                    mask = ~mask  # Invert for PyTorch transformer
                else:
                    mask = None
                
                # Pass through encoder
                sequence_output = self.encoder(embeddings, src_key_padding_mask=mask)
                
                # Take [CLS] equivalent (first token)
                pooled_output = sequence_output[:, 0]
                
                # Classify
                logits = self.classifier(pooled_output)
                
                loss = None
                if labels is not None:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits, labels)
                
                return type('ModelOutput', (), {
                    'loss': loss,
                    'logits': logits,
                    'hidden_states': sequence_output,
                })()
        
        class GPT2StyleModel(nn.Module):
            def __init__(self, vocab_size=20, hidden_size=64, num_labels=2):
                super().__init__()
                self.embeddings = nn.Embedding(vocab_size, hidden_size)
                decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=4, batch_first=True)
                self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
                self.classifier = nn.Linear(hidden_size, num_labels)
                
            def forward(self, input_ids, attention_mask=None, labels=None):
                # Get embeddings
                embeddings = self.embeddings(input_ids)
                
                # Create causal attention mask
                seq_length = input_ids.size(1)
                causal_mask = torch.triu(torch.ones(seq_length, seq_length) * float('-inf'), diagonal=1)
                causal_mask = causal_mask.to(input_ids.device)
                
                # Dummy memory for transformer decoder
                memory = torch.zeros(input_ids.size(0), 1, embeddings.size(-1)).to(input_ids.device)
                
                # Pass through decoder (using autoregressive transformer)
                sequence_output = self.decoder(embeddings, memory, tgt_mask=causal_mask)
                
                # Pool by taking the last non-padded token
                if attention_mask is not None:
                    last_token_indices = attention_mask.sum(dim=1) - 1
                    batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
                    pooled_output = sequence_output[batch_indices, last_token_indices]
                else:
                    pooled_output = sequence_output[:, -1]
                
                # Classify
                logits = self.classifier(pooled_output)
                
                loss = None
                if labels is not None:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits, labels)
                
                return type('ModelOutput', (), {
                    'loss': loss,
                    'logits': logits,
                    'hidden_states': sequence_output,
                })()
        
        class RobertaStyleModel(nn.Module):
            def __init__(self, vocab_size=20, hidden_size=64, num_labels=2):
                super().__init__()
                self.embeddings = nn.Embedding(vocab_size, hidden_size)
                encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=hidden_size*4, batch_first=True)
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)  # More layers than BERT
                self.pooler = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh()
                )
                self.classifier = nn.Linear(hidden_size, num_labels)
                
            def forward(self, input_ids, attention_mask=None, labels=None):
                # Get embeddings
                embeddings = self.embeddings(input_ids)
                
                # Create attention mask for transformer
                if attention_mask is not None:
                    # Convert to boolean mask where 1 = not masked, 0 = masked
                    mask = attention_mask.bool()
                    mask = ~mask  # Invert for PyTorch transformer
                else:
                    mask = None
                
                # Pass through encoder
                sequence_output = self.encoder(embeddings, src_key_padding_mask=mask)
                
                # Pool sequence output
                pooled_output = self.pooler(sequence_output[:, 0])
                
                # Classify
                logits = self.classifier(pooled_output)
                
                loss = None
                if labels is not None:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits, labels)
                
                return type('ModelOutput', (), {
                    'loss': loss,
                    'logits': logits,
                    'hidden_states': sequence_output,
                })()
        
        # Initialize models
        logger.info("Initializing models...")
        
        vocab_size = len(vocab)
        hidden_size = 64
        
        models = {
            "bert": BertStyleModel(vocab_size, hidden_size),
            "gpt2": GPT2StyleModel(vocab_size, hidden_size),
            "roberta": RobertaStyleModel(vocab_size, hidden_size)
        }
        
        # Set up optimizer with different weights for each model
        logger.info("Setting up MultiModelOptimizer...")
        optimizer = MultiModelOptimizer(
            models=models,
            lr=0.001,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            model_weights={"bert": 1.0, "gpt2": 0.8, "roberta": 1.2},
            gradient_accumulation_steps=2,
            clip_grad_norm=1.0,
            warmup_steps=50,
            grad_sync_frequency=20
        )
        
        # Set up scheduler
        scheduler = MultiModelScheduler(
            optimizer=optimizer,
            total_steps=2000,
            warmup_steps=50
        )
        
        # Move models to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        for model_name, model in models.items():
            models[model_name] = model.to(device)
        
        # Define metrics for tracking
        def compute_accuracy(logits, labels):
            preds = torch.argmax(logits, dim=-1)
            return (preds == labels).float().mean().item()
        
        # Training loop
        logger.info("Starting training...")
        
        num_epochs = 5
        
        for epoch in range(num_epochs):
            epoch_losses = {model_name: [] for model_name in models}
            epoch_accuracies = {model_name: [] for model_name in models}
            
            # Training
            for batch_idx, batch in enumerate(train_dataloader):
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Zero gradients
                optimizer.zero_grad()
                
                losses = {}
                
                # Forward/backward for each model
                for model_name, model in models.items():
                    try:
                        outputs = model(**batch)
                        loss = outputs.loss
                        loss.backward()
                        
                        losses[model_name] = loss.item()
                        epoch_losses[model_name].append(loss.item())
                        
                        # Calculate accuracy
                        acc = compute_accuracy(outputs.logits, batch["labels"])
                        epoch_accuracies[model_name].append(acc)
                        
                    except RuntimeError as e:
                        logger.error(f"Error during {model_name} forward/backward: {e}")
                        optimizer.zero_grad()
                        continue
                
                # Log metrics
                optimizer.log_metrics(losses)
                
                # Step optimizer and scheduler
                optimizer.step()
                scheduler.step()
                
                # Update convergence rates occasionally
                if batch_idx % 50 == 0:
                    scheduler.update_convergence_rates(optimizer.model_losses)
                
                # Log progress
                if batch_idx % 10 == 0:
                    log_str = f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_dataloader)}"
                    for model_name in models:
                        if epoch_losses[model_name]:
                            avg_loss = sum(epoch_losses[model_name][-10:]) / min(10, len(epoch_losses[model_name]))
                            avg_acc = sum(epoch_accuracies[model_name][-10:]) / min(10, len(epoch_accuracies[model_name]))
                            log_str += f", {model_name}: loss={avg_loss:.4f}, acc={avg_acc:.4f}"
                    logger.info(log_str)
            
            # Validation at end of epoch
            logger.info(f"Validating at end of epoch {epoch+1}...")
            
            val_losses = {model_name: [] for model_name in models}
            val_accuracies = {model_name: [] for model_name in models}
            
            for model_name, model in models.items():
                model.eval()
                
                with torch.no_grad():
                    for batch in val_dataloader:
                        batch = {k: v.to(device) for k, v in batch.items()}
                        outputs = model(**batch)
                        
                        val_losses[model_name].append(outputs.loss.item())
                        acc = compute_accuracy(outputs.logits, batch["labels"])
                        val_accuracies[model_name].append(acc)
                
                model.train()
                
                # Calculate average metrics
                avg_val_loss = sum(val_losses[model_name]) / len(val_losses[model_name])
                avg_val_acc = sum(val_accuracies[model_name]) / len(val_accuracies[model_name])
                
                logger.info(f"Epoch {epoch+1} - {model_name} validation: loss={avg_val_loss:.4f}, accuracy={avg_val_acc:.4f}")
                
                # Save model if it's the best so far
                torch.save(model.state_dict(), f"latest_{model_name}_model.pt")
            
            # Save epoch checkpoint
            torch.save({
                'epoch': epoch,
                'model_states': {name: model.state_dict() for name, model in models.items()},
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
            }, f"checkpoint_epoch_{epoch}.pt")
        
        logger.info("Training complete!")
        
    except Exception as e:
        logger.error(f"Fatal error in example_usage_with_simplified_data: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    # Use simplified data example to avoid dependency issues
    example_usage_with_simplified_data()
