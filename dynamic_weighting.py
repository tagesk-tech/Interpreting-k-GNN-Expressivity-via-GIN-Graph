"""
dynamic_weighting.py
Dynamic loss weighting scheme for GIN-Graph training.

Based on Equation 3 from the GIN-Graph paper:
λ(t) = λ_min + (λ_max - λ_min) * σ(k * (2*(t/T - p)/(1-p) - 1))

This ensures:
- Early training focuses on GAN loss (learning realistic graph structure)
- Later training shifts to GNN loss (maximizing class prediction probability)
"""

import math
from typing import Optional


class DynamicWeighting:
    """
    Dynamic weighting scheme for balancing GAN and GNN losses during training.
    
    The key insight from the paper is that:
    1. Early: Low λ → Focus on generating realistic graphs (GAN loss dominates)
    2. Late: High λ → Focus on class-specific patterns (GNN loss dominates)
    
    This prevents the generator from collapsing to invalid explanation graphs.
    """
    
    def __init__(
        self,
        total_iterations: int,
        min_lambda: float = 0.0,
        max_lambda: float = 1.0,
        p: float = 0.4,
        k: float = 10.0
    ):
        """
        Args:
            total_iterations: Total training iterations (T)
            min_lambda: Starting weight (λ_min, usually 0)
            max_lambda: Peak weight (λ_max, usually 1)
            p: Fraction of training before λ starts increasing significantly
               p=0.4 means first 40% of training focuses on structure
            k: Steepness of the sigmoid transition
        """
        self.T = total_iterations
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.p = p
        self.k = k
        self.current_step = 0
    
    def get_lambda(self, step: Optional[int] = None) -> float:
        """
        Get the lambda value for a given step.
        
        Args:
            step: Step number (if None, uses and increments internal counter)
            
        Returns:
            Lambda value in [min_lambda, max_lambda]
        """
        if step is None:
            step = self.current_step
            self.current_step += 1
        
        if self.p >= 1.0 or self.T == 0:
            return self.min_lambda
        
        # Progress ratio t/T
        progress = step / self.T
        
        # Normalized progress: maps [p, 1] to [0, 1]
        # Before p: negative values → sigmoid gives small λ
        # After p: positive values → sigmoid gives larger λ
        normalized_progress = (progress - self.p) / (1.0 - self.p)
        
        # Input to sigmoid: maps to approximately [-k, k]
        x = self.k * (2 * normalized_progress - 1)
        
        # Sigmoid
        sigmoid_val = 1 / (1 + math.exp(-x))
        
        # Scale to [min_lambda, max_lambda]
        lambda_val = self.min_lambda + (self.max_lambda - self.min_lambda) * sigmoid_val
        
        return lambda_val
    
    def reset(self) -> None:
        """Reset the internal step counter."""
        self.current_step = 0
    
    def get_schedule(self, num_points: int = 100) -> list:
        """
        Get the full lambda schedule for visualization.
        
        Args:
            num_points: Number of points to sample
            
        Returns:
            List of (step, lambda) tuples
        """
        schedule = []
        for i in range(num_points):
            step = int(i * self.T / num_points)
            lambda_val = self.get_lambda(step)
            schedule.append((step, lambda_val))
        return schedule


class ConstantWeighting:
    """Simple constant weighting for comparison."""
    
    def __init__(self, lambda_value: float = 0.5):
        self.lambda_value = lambda_value
    
    def get_lambda(self, step: Optional[int] = None) -> float:
        return self.lambda_value
    
    def reset(self) -> None:
        pass


class LinearWeighting:
    """Linear increase from min to max."""
    
    def __init__(
        self,
        total_iterations: int,
        min_lambda: float = 0.0,
        max_lambda: float = 1.0,
        warmup_fraction: float = 0.2
    ):
        self.T = total_iterations
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.warmup_fraction = warmup_fraction
        self.current_step = 0
    
    def get_lambda(self, step: Optional[int] = None) -> float:
        if step is None:
            step = self.current_step
            self.current_step += 1
        
        warmup_steps = int(self.warmup_fraction * self.T)
        
        if step < warmup_steps:
            return self.min_lambda
        
        progress = (step - warmup_steps) / (self.T - warmup_steps)
        return self.min_lambda + progress * (self.max_lambda - self.min_lambda)
    
    def reset(self) -> None:
        self.current_step = 0


if __name__ == "__main__":
    # Test and visualize the weighting schemes
    import matplotlib.pyplot as plt
    
    T = 1000
    
    dynamic = DynamicWeighting(T, p=0.4, k=10.0)
    linear = LinearWeighting(T, warmup_fraction=0.2)
    constant = ConstantWeighting(0.5)
    
    steps = list(range(T))
    dynamic_lambdas = [dynamic.get_lambda(s) for s in steps]
    linear_lambdas = [linear.get_lambda(s) for s in steps]
    constant_lambdas = [constant.get_lambda(s) for s in steps]
    
    plt.figure(figsize=(10, 5))
    plt.plot(steps, dynamic_lambdas, label='Dynamic (sigmoid)', linewidth=2)
    plt.plot(steps, linear_lambdas, label='Linear', linewidth=2)
    plt.plot(steps, constant_lambdas, label='Constant', linewidth=2)
    plt.xlabel('Training Step')
    plt.ylabel('Lambda (λ)')
    plt.title('GNN Loss Weight Schedules')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lambda_schedules.png', dpi=150)
    plt.close()
    
    print("Lambda schedule plot saved to lambda_schedules.png")
