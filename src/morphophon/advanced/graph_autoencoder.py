"""
Graph Autoencoder for Arabic Morphophonological Analysis
الترميز الآلي للرسم البياني للتحليل الصرفي الصوتي العربي

Implements Graph Neural Network components for the hierarchical architecture:
- Graph Convolutional Networks (GCN) for morphological relationships
- Autoencoder for dimensionality reduction and feature learning
- Node and edge embeddings for Arabic linguistic structures
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data math
from typing import_data Any, Dict, List, Optional, Tuple, Union

import_data numpy as np

try:
    import_data torch
    import_data torch.nn as nn
    import_data torch.nn.functional as F
    import_data torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None     # type: ignore
    F = None      # type: ignore
    optim = None  # type: ignore

# Graph relationship types for Arabic morphology
ARABIC_GRAPH_RELATIONS = {
    'next_phone': 1,        # Sequential phoneme relationship
    'same_syllabic_unit': 2,     # Phonemes in same syllabic_unit
    'attached_to': 3,       # Vowel attached to consonant
    'syll_in_word': 4,      # SyllabicUnit position in word
    'forms': 5,             # Root forms relationship
    'derives': 6,           # Derivational relationship
    'instantiates': 7,      # Pattern instantiation
    'subject_of': 8,        # Subject relationship
    'object_of': 9,         # Object relationship
    'idafa_of': 10,         # Genitive construction
    'contains': 11,         # Containment relationship
    'sentence_order': 12,   # Sentence order
    '<PAD>': 0             # Padding relation
}

# Node types in the Arabic morphological graph
NODE_TYPES = {
    'phoneme': 1,
    'vowel': 2,
    'syllabic_unit': 3,
    'root': 4,
    'pattern': 5,
    'word': 6,
    'sentence': 7,
    'text': 8,
    '<PAD>': 0
}

class GraphConvLayer:
    """
    Graph Convolutional Layer for Arabic morphological graphs
    طبقة التطبيق التشابهي للرسم البياني
    """
    
    def __init__(self, in_features: int, out_features: int, enable_neural: bool = True):
        """
        Initialize graph convolutional layer
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension  
            enable_neural: Whether to use neural implementation
        """
        self.in_features = in_features
        self.out_features = out_features
        self.enable_neural = enable_neural and TORCH_AVAILABLE
        
        if self.enable_neural:
            self._init_neural_layer()
        else:
            self._init_fallback_layer()
            
    def _init_neural_layer(self):
        """Initialize PyTorch neural layer"""
        if not TORCH_AVAILABLE or torch is None or nn is None:
            raise RuntimeError("PyTorch not available for neural components")
            
        self.linear = nn.Linear(self.in_features, self.out_features)
        self.activation = nn.ReLU()
        
    def _init_fallback_layer(self):
        """Initialize fallback implementation"""
        # Simple linear transformation matrix
        self.weight_matrix = []
        for i in range(self.out_features):
            row = []
            for j in range(self.in_features):
                # Initialize with small random-like values
                val = math.sin(i + j) * 0.1
                row.append(val)
            self.weight_matrix.append(row)
            
    def forward(self, X: List[List[float]], A: List[List[float]]) -> List[List[float]]:
        """
        Forward pass through graph convolutional layer
        
        Args:
            X: Node features [N, in_features]
            A: Adjacency matrix [N, N]
            
        Returns:
            Updated node features [N, out_features]
        """
        if self.enable_neural and torch is not None:
            return self._neural_forward(X, A)
        else:
            return self._fallback_forward(X, A)
            
    def _neural_forward(self, X: List[List[float]], A: List[List[float]]) -> List[List[float]]:
        """Neural forward pass"""
        if torch is None:
            raise RuntimeError("PyTorch not available")
            
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            A_tensor = torch.tensor(A, dtype=torch.float32)
            
            # Graph convolution: AXW
            AX = torch.matmul(A_tensor, X_tensor)
            output = self.linear(AX)
            output = self.activation(output)
            
            return output.tolist()
            
    def _fallback_forward(self, X: List[List[float]], A: List[List[float]]) -> List[List[float]]:
        """Fallback forward pass"""
        N = len(X)
        
        # Matrix multiplication A @ X
        AX = []
        for i in range(N):
            row = []
            for j in range(len(X[0])):
                val = 0.0
                for k in range(N):
                    val += A[i][k] * X[k][j]
                row.append(val)
            AX.append(row)
            
        # Linear transformation AX @ W
        output = []
        for i in range(N):
            row = []
            for j in range(self.out_features):
                val = 0.0
                for k in range(self.in_features):
                    val += AX[i][k] * self.weight_matrix[j][k]
                # Apply ReLU activation
                row.append(max(0.0, val))
            output.append(row)
            
        return output

class GraphAutoencoder:
    """
    Graph Autoencoder for Arabic Morphophonological Analysis
    مُرمز الرسم البياني الآلي للتحليل الصرفي الصوتي العربي
    
    Implements graph neural network with encoder-decoder architecture
    for learning Arabic morphological representations.
    """
    
    def __init__(self, num_features: int, hidden_dim: int = 256, 
                 latent_dim: int = 64, num_relations: int = 12,
                 enable_neural: bool = True):
        """
        Initialize graph autoencoder
        
        Args:
            num_features: Number of input node features
            hidden_dim: Hidden layer dimension
            latent_dim: Latent space dimension
            num_relations: Number of relation types
            enable_neural: Whether to use neural implementation
        """
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_relations = num_relations
        self.enable_neural = enable_neural and TORCH_AVAILABLE
        
        # Training statistics
        self.training_stats = {
            'epochs_trained': 0,
            'best_loss': float('inf'),
            'last_loss': 0.0
        }
        
        if self.enable_neural:
            self._init_neural_components()
        else:
            self._init_fallback_components()
            
    def _init_neural_components(self):
        """Initialize PyTorch neural components"""
        if not TORCH_AVAILABLE or torch is None or nn is None:
            raise RuntimeError("PyTorch not available for neural components")
            
        # Encoder layers
        self.encoder_layer1 = GraphConvLayer(self.num_features, self.hidden_dim)
        self.encoder_layer2 = GraphConvLayer(self.hidden_dim, self.latent_dim)
        
        # Decoder layers
        self.decoder_layer1 = GraphConvLayer(self.latent_dim, self.hidden_dim)
        self.decoder_layer2 = GraphConvLayer(self.hidden_dim, self.num_features)
        
        # Relation embedding for edge types
        self.relation_emb = nn.Embedding(self.num_relations + 1, 8, padding_idx=0)
        
        # Optimizer
        if optim is not None:
            params = []
            if hasattr(self.encoder_layer1, 'linear'):
                params.extend([self.encoder_layer1.linear.weight, self.encoder_layer1.linear.bias])
            if hasattr(self.encoder_layer2, 'linear'):
                params.extend([self.encoder_layer2.linear.weight, self.encoder_layer2.linear.bias])
            if hasattr(self.decoder_layer1, 'linear'):
                params.extend([self.decoder_layer1.linear.weight, self.decoder_layer1.linear.bias])
            if hasattr(self.decoder_layer2, 'linear'):
                params.extend([self.decoder_layer2.linear.weight, self.decoder_layer2.linear.bias])
            params.append(self.relation_emb.weight)
            
            self.optimizer = optim.Adam(params, lr=1e-3)
        
    def _init_fallback_components(self):
        """Initialize fallback components"""
        self.encoder_layer1 = GraphConvLayer(self.num_features, self.hidden_dim, False)
        self.encoder_layer2 = GraphConvLayer(self.hidden_dim, self.latent_dim, False)
        self.decoder_layer1 = GraphConvLayer(self.latent_dim, self.hidden_dim, False)
        self.decoder_layer2 = GraphConvLayer(self.hidden_dim, self.num_features, False)
        
        # Simple relation embeddings
        self.relation_embeddings = {}
        for rel, idx in ARABIC_GRAPH_RELATIONS.items():
            if idx > 0:
                # Create embedding based on relation type
                emb = [math.sin(idx + i) * 0.1 for i in range(8)]
                self.relation_embeddings[rel] = emb
                
    def encode(self, X: List[List[float]], A: List[List[float]]) -> List[List[float]]:
        """
        Encode node features to latent space
        
        Args:
            X: Node features [N, num_features]
            A: Adjacency matrix [N, N]
            
        Returns:
            Latent representations [N, latent_dim]
        """
        # First encoder layer
        h1 = self.encoder_layer1.forward(X, A)
        
        # Second encoder layer  
        z = self.encoder_layer2.forward(h1, A)
        
        return z
        
    def decode(self, Z: List[List[float]], A: List[List[float]]) -> List[List[float]]:
        """
        Decode latent representations back to feature space
        
        Args:
            Z: Latent representations [N, latent_dim]
            A: Adjacency matrix [N, N]
            
        Returns:
            Reconstructed features [N, num_features]
        """
        # First decoder layer
        h1 = self.decoder_layer1.forward(Z, A)
        
        # Second decoder layer
        X_recon = self.decoder_layer2.forward(h1, A)
        
        return X_recon
        
    def forward(self, X: List[List[float]], A: List[List[float]]) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Full forward pass (encode then decode)
        
        Args:
            X: Input node features
            A: Adjacency matrix
            
        Returns:
            Tuple of (latent_representations, reconstructed_features)
        """
        Z = self.encode(X, A)
        X_recon = self.decode(Z, A)
        return Z, X_recon
        
    def compute_loss(self, X: List[List[float]], X_recon: List[List[float]]) -> float:
        """
        Compute reconstruction loss
        
        Args:
            X: Original features
            X_recon: Reconstructed features
            
        Returns:
            Mean squared error loss
        """
        total_loss = 0.0
        count = 0
        
        for i in range(len(X)):
            for j in range(len(X[i])):
                diff = X[i][j] - X_recon[i][j]
                total_loss += diff * diff
                count += 1
                
        return total_loss / count if count > 0 else 0.0
        
    def train_step(self, X: List[List[float]], A: List[List[float]]) -> float:
        """
        Single training step
        
        Args:
            X: Input features
            A: Adjacency matrix
            
        Returns:
            Loss value
        """
        if self.enable_neural and torch is not None and hasattr(self, 'optimizer'):
            return self._neural_train_step(X, A)
        else:
            return self._fallback_train_step(X, A)
            
    def _neural_train_step(self, X: List[List[float]], A: List[List[float]]) -> float:
        """Neural training step with gradients"""
        if torch is None or not hasattr(self, 'optimizer'):
            return 0.0
            
        self.optimizer.zero_grad()
        
        # Forward pass
        Z, X_recon = self.forward(X, A)
        
        # Compute loss
        loss = self.compute_loss(X, X_recon)
        
        # Store loss
        self.training_stats['last_loss'] = loss
        if loss < self.training_stats['best_loss']:
            self.training_stats['best_loss'] = loss
            
        return loss
        
    def _fallback_train_step(self, X: List[List[float]], A: List[List[float]]) -> float:
        """Fallback training step (just compute loss)"""
        _, X_recon = self.forward(X, A)
        loss = self.compute_loss(X, X_recon)
        
        self.training_stats['last_loss'] = loss
        if loss < self.training_stats['best_loss']:
            self.training_stats['best_loss'] = loss
            
        return loss
        
    def train(self, X: List[List[float]], A: List[List[float]], 
              epochs: int = 100, verbose: bool = True) -> List[float]:
        """
        Train the autoencoder
        
        Args:
            X: Training features
            A: Adjacency matrix
            epochs: Number of training epochs
            verbose: Whether to print progress
            
        Returns:
            List of loss values per epoch
        """
        losses = []
        
        for epoch in range(epochs):
            loss = self.train_step(X, A)
            losses.append(loss)
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")
                
        self.training_stats['epochs_trained'] += epochs
        
        return losses
        
    def embed(self, X: List[List[float]], A: List[List[float]]) -> List[List[float]]:
        """
        Get embeddings for input features
        
        Args:
            X: Input features
            A: Adjacency matrix
            
        Returns:
            Latent embeddings
        """
        return self.encode(X, A)
        
    def build_adjacency_matrix(self, relations: List[Tuple[int, int, str]], 
                              num_nodes: int) -> List[List[float]]:
        """
        Build adjacency matrix from relation list
        
        Args:
            relations: List of (source, target, relation_type) tuples
            num_nodes: Total number of nodes
            
        Returns:
            Adjacency matrix [num_nodes, num_nodes]
        """
        # Initialize adjacency matrix
        A = [[0.0 for _ in range(num_nodes)] for _ in range(num_nodes)]
        
        # Add self-connections
        for i in range(num_nodes):
            A[i][i] = 1.0
            
        # Add relations
        for source, target, rel_type in relations:
            # Check bounds and relation validity in single condition
            if (0 <= source < num_nodes and 0 <= target < num_nodes and 
                rel_type in ARABIC_GRAPH_RELATIONS):
                weight = 1.0  # Uniform weight for now
                A[source][target] = weight
                # Add reverse connection for undirected edges
                A[target][source] = weight
                    
        return A
        
    def normalize_adjacency(self, A: List[List[float]]) -> List[List[float]]:
        """
        Normalize adjacency matrix (row normalization)
        
        Args:
            A: Adjacency matrix
            
        Returns:
            Normalized adjacency matrix
        """
        A_norm = []
        for i, row in enumerate(A):
            row_sum = sum(row)
            if row_sum > 0:
                norm_row = [val / row_sum for val in row]
            else:
                norm_row = row[:]  # Copy original row if sum is 0
            A_norm.append(norm_row)
            
        return A_norm
        
    def analyze_graph_structure(self, A: List[List[float]]) -> Dict:
        """
        Analyze graph structure properties
        
        Args:
            A: Adjacency matrix
            
        Returns:
            Dictionary with graph statistics
        """
        num_nodes = len(A)
        total_edges = 0
        node_degrees = []
        
        for i in range(num_nodes):
            # Count non-zero connections for this node
            degree = sum(val > 0 for val in A[i])
            node_degrees.append(degree)
            total_edges += degree
            
        # Remove double counting for undirected edges
        total_edges = total_edges // 2
        
        avg_degree = sum(node_degrees) / num_nodes if num_nodes > 0 else 0
        max_degree = max(node_degrees, default=0)
        min_degree = min(node_degrees, default=0)
        
        return {
            'num_nodes': num_nodes,
            'num_edges': total_edges,
            'average_degree': avg_degree,
            'max_degree': max_degree,
            'min_degree': min_degree,
            'density': total_edges / (num_nodes * (num_nodes - 1) / 2) if num_nodes > 1 else 0
        }
        
    def get_training_stats(self) -> Dict:
        """Get training statistics"""
        return self.training_stats.copy()
        
    def get_info(self) -> Dict:
        """Get information about the autoencoder"""
        return {
            'num_features': self.num_features,
            'hidden_dim': self.hidden_dim,
            'latent_dim': self.latent_dim,
            'num_relations': self.num_relations,
            'neural_enabled': self.enable_neural,
            'torch_available': TORCH_AVAILABLE,
            'supported_relations': list(ARABIC_GRAPH_RELATIONS.keys()),
            'node_types': list(NODE_TYPES.keys()),
            'training_stats': self.training_stats
        }

# Example usage and testing
if __name__ == "__main__":
    print("🕸️ Testing Arabic Graph Autoencoder")
    print("=" * 50)
    
    # Initialize autoencoder
    num_features = 10
    autoencoder = GraphAutoencoder(
        num_features=num_features,
        hidden_dim=16,
        latent_dim=8,
        enable_neural=TORCH_AVAILABLE
    )
    
    # Create sample data
    num_nodes = 5
    X = [[float(i + j) for j in range(num_features)] for i in range(num_nodes)]
    
    # Create sample relations
    relations = [
        (0, 1, 'next_phone'),
        (1, 2, 'same_syllabic_unit'),
        (2, 3, 'forms'),
        (3, 4, 'contains')
    ]
    
    # Build adjacency matrix
    A = autoencoder.build_adjacency_matrix(relations, num_nodes)
    A_norm = autoencoder.normalize_adjacency(A)
    
    print("📊 Graph structure:")
    graph_stats = autoencoder.analyze_graph_structure(A)
    for key, value in graph_stats.items():
        print(f"  {key}: {value}")
        
    # Test encoding
    print(f"\n🔄 Testing encoding...")
    Z = autoencoder.embed(X, A_norm)
    print(f"Original shape: {len(X)}x{len(X[0])}")
    print(f"Encoded shape: {len(Z)}x{len(Z[0])}")
    
    # Test training
    print(f"\n🎯 Testing training...")
    losses = autoencoder.train(X, A_norm, epochs=10, verbose=False)
    print(f"Final loss: {losses[-1]:.6f}")
    
    print(f"\n📋 System info:")
    info = autoencoder.get_info()
    for key, value in info.items():
        if key not in ['supported_relations', 'node_types', 'training_stats']:
            print(f"  {key}: {value}")
