import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# --- Import your custom modules ---
# Ensure these files are in the same directory
from gin_discriminator import GIN_Discriminator
from dynamic_weighing import DynamicWeighting
from gin_generator import GIN_Generator
from data_loader import load_mutag

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LATENT_DIM = 32
HIDDEN_DIM = 128
LR = 0.001
EPOCHS = 300
BATCH_SIZE = 64
TARGET_CLASS = 0  # 0 for Mutagen (Target), 1 for Non-Mutagen

# ==============================================================================
#  OPTION A: The Wrapper Class (New Addition)
# ==============================================================================
class DenseModelWrapper(torch.nn.Module):
    """
    Wraps a standard sparse GNN (which expects edge_index) to accept 
    dense inputs (x, adj) from the Generator.
    
    Crucial for Gradient Flow:
    It passes the values in 'adj' as 'edge_weight' to the GNN.
    """
    def __init__(self, sparse_model):
        super(DenseModelWrapper, self).__init__()
        self.sparse_model = sparse_model

    def forward(self, x, adj):
        """
        Args:
            x: Node features [Batch, N, Feats]
            adj: Continuous adjacency matrix [Batch, N, N]
        """
        # 1. Convert Dense Adj -> Sparse Edges
        # dense_to_sparse returns (edge_index, edge_attr)
        # edge_attr contains the continuous values (gradients flow here!)
        edge_index, edge_weight = dense_to_sparse(adj)
        
        # 2. Flatten x for the sparse GNN [Batch*N, Feats]
        # PyG sparse models usually expect a single large matrix of nodes
        batch_size, num_nodes, num_feats = x.size()
        x_flat = x.view(-1, num_feats)
        
        # 3. Create Batch Vector (assigns nodes to graphs)
        # [0, 0, ..., 1, 1, ...]
        batch_vec = torch.arange(batch_size, device=x.device).view(-1, 1).repeat(1, num_nodes).view(-1)
        
        # 4. Forward Pass through original model
        # Note: Your original model's forward() must accept edge_weight
        # e.g., def forward(self, x, edge_index, batch, edge_weight=None):
        return self.sparse_model(x_flat, edge_index, batch_vec, edge_weight=edge_weight)

# ==============================================================================
#  Training Function
# ==============================================================================
def train_gin_graph(pretrained_gnn):
    # 1. Setup Data
    dataset = load_mutag()
    # Filter dataset for ONLY the target class
    target_indices = [i for i, data in enumerate(dataset) if data.y.item() == TARGET_CLASS]
    target_dataset = dataset[target_indices]
    
    # Custom loader (using lambda for simple collate because we handle stacking manually)
    train_loader = torch.utils.data.DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: x)

    # 2. Initialize Models
    sample = dataset[0]
    max_nodes = 28  # MUTAG max nodes
    num_node_feats = dataset.num_node_features

    generator = GIN_Generator(LATENT_DIM, max_nodes, num_node_feats, HIDDEN_DIM).to(DEVICE)
    discriminator = GIN_Discriminator(max_nodes, num_node_feats, HIDDEN_DIM).to(DEVICE)
    
    # Pre-trained GNN Setup
    pretrained_gnn = pretrained_gnn.to(DEVICE)
    pretrained_gnn.eval() 
    for param in pretrained_gnn.parameters():
        param.requires_grad = False

    # 3. Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

    # 4. Weight Scheduler
    total_iters = EPOCHS * len(train_loader)
    weight_scheduler = DynamicWeighting(total_iters, p=0.4, k=10.0)

    print(f"Starting training on device: {DEVICE}")

    # --- MAIN LOOP ---
    for epoch in range(EPOCHS):
        for i, batch_list in enumerate(train_loader):
            
            # A. Prepare Real Batch (Stacking)
            real_x_list = []
            real_adj_list = []
            
            for data in batch_list:
                num_nodes = data.num_nodes
                x = data.x.float()
                # Zero padding for nodes
                x_padded = F.pad(x, (0, 0, 0, max_nodes - num_nodes))
                real_x_list.append(x_padded)
                
                # Dense Adj with padding
                adj = to_dense_adj(data.edge_index, max_num_nodes=max_nodes)[0]
                real_adj_list.append(adj)

            real_x = torch.stack(real_x_list).to(DEVICE)
            real_adj = torch.stack(real_adj_list).to(DEVICE)
            current_batch_size = real_x.size(0)

            # ============================================================
            #  PHASE 1: TRAIN DISCRIMINATOR
            # ============================================================
            optimizer_D.zero_grad()

            real_scores = discriminator(real_x, real_adj)

            z = torch.randn(current_batch_size, LATENT_DIM).to(DEVICE)
            fake_adj, fake_x = generator(z)
            fake_scores = discriminator(fake_x.detach(), fake_adj.detach())

            gp = discriminator.compute_gradient_penalty(real_x, real_adj, fake_x.detach(), fake_adj.detach(), device=DEVICE)
            d_loss = discriminator.compute_loss(real_scores, fake_scores, gp)
            
            d_loss.backward()
            optimizer_D.step()

            # ============================================================
            #  PHASE 2: TRAIN GENERATOR
            # ============================================================
            if i % 1 == 0:
                optimizer_G.zero_grad()
                
                # 1. Regenerate Fakes (with gradients)
                fake_adj, fake_x = generator(z)
                
                # 2. GAN Loss
                gan_scores = discriminator(fake_x, fake_adj)
                l_gan = -torch.mean(gan_scores)

                # 3. GNN Loss (Using the Wrapper logic implicitly or explicitly)
                gnn_logits = pretrained_gnn(fake_x, fake_adj) 
                
                target_labels = torch.full((current_batch_size,), TARGET_CLASS, device=DEVICE, dtype=torch.long)
                l_gnn = F.cross_entropy(gnn_logits, target_labels)

                # 4. Combine
                curr_lambda = weight_scheduler.get_current_lambda()
                total_g_loss = ((1 - curr_lambda) * l_gan) + (curr_lambda * l_gnn)
                
                total_g_loss.backward()
                optimizer_G.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | D Loss: {d_loss.item():.4f} | G Loss: {total_g_loss.item():.4f} | Lambda: {curr_lambda:.2f}")

    return generator

# ==============================================================================
#  Visualization Helper
# ==============================================================================
def visualize_explanations(generator, num_graphs=5):
    """
    Samples graphs from the generator, converts them to NetworkX, and plots them.
    Includes fixes for shape mismatches and self-loops.
    """
    generator.eval()
    z = torch.randn(num_graphs, LATENT_DIM).to(DEVICE)
    
    # Use temp=0.1 for cleaner discrete graphs
    fake_adj, fake_x = generator(z, temp=0.1) 
    
    fake_adj = fake_adj.cpu().detach().numpy()
    fake_x = fake_x.cpu().detach().numpy()
    
    atom_labels = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}
    
    fig, axes = plt.subplots(1, num_graphs, figsize=(15, 3))
    if num_graphs == 1: axes = [axes]

    for i in range(num_graphs):
        ax = axes[i]
        
        # 1. Create Graph from Adjacency > 0.8
        rows, cols = np.where(fake_adj[i] > 0.8)
        edges = zip(rows.tolist(), cols.tolist())
        G = nx.Graph()
        G.add_edges_from(edges)
        
        # 2. Cleanup: Remove self-loops FIRST
        G.remove_edges_from(nx.selfloop_edges(G))
        
        # 3. Cleanup: Remove isolated nodes SECOND (Crucial Fix)
        # We must do this BEFORE generating the color list
        G.remove_nodes_from(list(nx.isolates(G)))
        
        # 4. Generate Labels/Colors only for remaining nodes
        atom_indices = np.argmax(fake_x[i], axis=1)
        node_labels = {}
        node_colors = []
        
        # Now we iterate only over the nodes that actually exist in the final graph
        for node_idx in G.nodes():
            # Get atom type for this specific node index
            atom_type = atom_indices[node_idx]
            label = atom_labels.get(atom_type, '?')
            node_labels[node_idx] = label
            
            # Color logic
            if label == 'C': color = 'orange'
            elif label == 'N': color = 'cyan'
            elif label == 'O': color = 'red'
            elif label == 'F': color = 'green'
            elif label == 'I': color = 'purple'
            elif label == 'Cl': color = 'lightgreen'
            elif label == 'Br': color = 'brown'
            else: color = 'gray'
            node_colors.append(color)

        # 5. Draw
        if G.number_of_nodes() > 0:
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos, ax=ax, with_labels=True, labels=node_labels, 
                    node_color=node_colors, node_size=300, edge_color='black', font_size=10)
        else:
            ax.text(0.5, 0.5, "Empty Graph", ha='center')
            
        ax.set_title(f"Sample {i+1}")

    plt.tight_layout()
    plt.show()

# ==============================================================================
#  Main Execution
# ==============================================================================
if __name__ == "__main__":
    print("Loading Pre-trained GNN...")

    # --- OPTION A: Using the Wrapper ---
    # Assume 'OriginalSparseGNN' is your loaded model class
    # 1. Load your actual model
    # sparse_gnn = OriginalSparseGNN(...)
    # sparse_gnn.load_state_dict(torch.load('trained_1gnn.pt'))
    
    # 2. Wrap it for GIN-Graph
    # pretrained_gnn = DenseModelWrapper(sparse_gnn)

    # --- MOCK FOR DEMONSTRATION (Since we don't have your class file) ---
    class MockGNN(torch.nn.Module):
        def forward(self, x, adj):
            return torch.randn(x.size(0), 2).to(x.device)
    pretrained_gnn = MockGNN()
    # ------------------------------------------------------------------

    print("Training Generator...")
    trained_generator = train_gin_graph(pretrained_gnn)

    print("Visualizing Explanations...")
    visualize_explanations(trained_generator, num_graphs=5)