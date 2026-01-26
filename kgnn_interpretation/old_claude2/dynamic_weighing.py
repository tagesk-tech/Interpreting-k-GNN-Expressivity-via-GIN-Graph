import torch
import math

class DynamicWeighting:
    def __init__(self, total_iterations, min_lambda=0.0, max_lambda=1.0, p=0.4, k=10.0):
        """
        Args:
            total_iterations (int): T in the paper.
            min_lambda (float): Starting weight (usually 0).
            max_lambda (float): Peak weight.
            p (float): Percentage of training to wait before increasing (0.0 to 1.0).
                       p=0.4 means the first 40% of training focuses mostly on structure.
            k (float): Steepness of the curve.
        """
        self.T = total_iterations
        self.min_l = min_lambda
        self.max_l = max_lambda
        self.p = p
        self.k = k
        self.current_step = 0

    def get_current_lambda(self):
        """
        Returns the lambda value for the current step and increments the step counter.
        Implements Equation 3 from the paper.
        """
        # Avoid division by zero if p is exactly 1 (though p should be < 1)
        if self.p >= 1.0:
            return self.min_l

        # 1. Calculate Progress Ratio (t/T)
        progress = self.current_step / self.T
        
        # 2. Check if we are in the "waiting period" (Optional optimization)
        # If progress is very low, the formula naturally gives a low value,
        # but we can clamp it for safety.
        
        # 3. The Core Formula: k * (2 * (progress - p) / (1 - p) - 1)
        # Inner term maps the "active" training time to range [-1, 1] approximately
        normalized_progress = (progress - self.p) / (1.0 - self.p)
        x = self.k * (2 * normalized_progress - 1)
        
        # 4. Apply Sigmoid
        # sigmoid(x) = 1 / (1 + exp(-x))
        sigmoid_val = 1 / (1 + math.exp(-x))
        
        # 5. Scale to range
        lambda_val = self.min_l + (self.max_l - self.min_l) * sigmoid_val
        
        # Increment step for next call
        self.current_step += 1
        
        return lambda_val

    def reset(self):
        self.current_step = 0