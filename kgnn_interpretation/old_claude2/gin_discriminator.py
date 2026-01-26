import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv

class GIN_Discriminator(nn.Module):
    def __init__(self, max_nodes, num_node_feats, hidden_dim=128):
        """
        Args:
            max_nodes (int): N (28 for MUTAG)
            num_node_feats (int): D (7 for MUTAG)
            hidden_dim (int): Internal hidden layer size
        """
        super(GIN_Discriminator, self).__init__()
        
        # 1. Architecture
        self.conv1 = DenseGCNConv(num_node_feats, hidden_dim)
        self.conv2 = DenseGCNConv(hidden_dim, hidden_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1) # Output: Scalar Score
        )
    
    def forward(self, x, adj):
        """
        Standard forward pass. 
        Returns the raw score D(x).
        """
        # Graph Convolution Layers
        h = F.relu(self.conv1(x, adj))
        h = F.relu(self.conv2(h, adj))
        
        # Pooling (Global Mean) - aggregates node features into one graph vector
        graph_embedding = torch.mean(h, dim=1) 
        
        # Final Score
        score = self.mlp(graph_embedding)
        return score

    # =================================================================
    #  Encapsulated Math: Gradient Penalty (The "Hard" Part)
    # =================================================================
    def compute_gradient_penalty(self, real_x, real_adj, fake_x, fake_adj, device='cpu'):
        """
        Calculates the Gradient Penalty term: lambda * (||\nabla D(\hat{x})|| - 1)^2
        """
        batch_size = real_x.size(0)
        
        # 1. Sample epsilon (random interpolation factor)
        # Shape must match for broadcasting: [Batch, 1, 1]
        epsilon = torch.rand(batch_size, 1, 1).to(device)
        
        # 2. Create Interpolated Graphs (\hat{x})
        # We mix Real and Fake data
        alpha_x = (epsilon * real_x + (1 - epsilon) * fake_x).requires_grad_(True)
        alpha_adj = (epsilon * real_adj + (1 - epsilon) * fake_adj).requires_grad_(True)
        
        # 3. Get the Discriminator's score for these mixed graphs
        interpolated_scores = self.forward(alpha_x, alpha_adj)
        
        # 4. Calculate Gradients (\nabla D(\hat{x}))
        # We ask PyTorch: "How much does the score change if we change the input?"
        grad_outputs = torch.ones_like(interpolated_scores, device=device)
        
        gradients = torch.autograd.grad(
            outputs=interpolated_scores,
            inputs=[alpha_x, alpha_adj],
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )
        
        # We combine gradients from X and Adj to get a single norm
        grad_x = gradients[0].view(batch_size, -1)
        grad_adj = gradients[1].view(batch_size, -1)
        grad_flat = torch.cat([grad_x, grad_adj], dim=1)
        
        # 5. Compute Norm and Penalty
        gradient_norm = grad_flat.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()
        
        return penalty

    # =================================================================
    #  Encapsulated Math: Total Loss
    # =================================================================
    def compute_loss(self, real_scores, fake_scores, gp, lambda_gp=10.0):
        """
        Calculates the full WGAN loss:
        L = E[Fake] - E[Real] + lambda * GP
        """
        # 1. Wasserstein Loss (Maximize Real, Minimize Fake)
        # Since we are minimizing loss: -(Real) + (Fake)
        w_loss = torch.mean(fake_scores) - torch.mean(real_scores)
        
        # 2. Add Penalty
        total_loss = w_loss + (lambda_gp * gp)
        
        return total_loss