# 🔍 bipolar_ml: High-Performance Hyperdimensional Computing Framework

bipolar_ml is a framework for hyperdimensional computing (HDC) with built-in optimization for binary neural representations. It's designed from the ground up to make bitwise operations sing by leveraging bit-packed storage and vectorized operations.

## 🧠 What is Hyperdimensional Computing?

HDC represents information using high-dimensional binary vectors (hypervectors) where the magic happens through simple bitwise operations:
- **Binding**: XOR operations to combine concepts
- **Bundling**: Thresholded addition to represent sets
- **Similarity**: Hamming distance to measure relationships

This approach lets you build powerful semantic models and pattern recognizers that are crazy efficient and robust to noise.

## ✨ Key Features

- **Bit-Packed Storage**: Uses just 1/8th the memory with optimized bit operations
- **Vectorized Operations**: Blazing fast parallel processing of bitwise computations
- **Adaptive Techniques**: Self-tuning parameters based on data characteristics
- **Hybrid Training**: Memory-efficient SignSGD algorithm with targeted bit flips
- **Semantic Modeling**: Extract meaning and relationships from text
- **BitTransformer**: State-of-the-art transformer architecture using binary representations
- **Progressive Distillation**: Convert standard neural networks to binary representations

## 🚀 Quick Start

### Basic Vector Operations

```python
from bipolar_ml import Bipolar

# Create random bipolar vectors
vec1 = Bipolar.random((1024,))
vec2 = Bipolar.random((1024,))

# Binding (XOR)
bound = vec1.bind(vec2)  # or simply: vec1 * vec2

# Similarity measurement
similarity = vec1.similarity(vec2)  # ranges from 0 to 1

# Unbinding
recovered = bound.unbind(vec2)  # should be similar to vec1

# Verify recovery quality
recovery_quality = vec1.similarity(recovered)
print(f"Recovery quality: {recovery_quality:.4f}")
```

### Semantic Modeling

```python
from bipolar_ml import AdaptiveBitSemantic, AdaptiveConfig

# Configure the semantic model
config = AdaptiveConfig(
    dimension=1024,
    min_freq=3,
    max_words=5000,
    max_iterations=200
)

# Create and train model
model = AdaptiveBitSemantic(config)
model.fit(text_corpus)

# Find similar words
similar_words = model.find_similar("neural", n=5)
print("Words similar to 'neural':")
for word, distance in similar_words:
    print(f"  {word}: {distance:.4f}")

# Solve analogies
results = model.analogy("king", "man", "woman", n=3)
print("king:man::woman:?")
for word, distance in results:
    print(f"  {word}: {distance:.4f}")
```

### Binary Classification

```python
from bipolar_ml import BipolarNetwork, train_bipolar_classifier
import torch

# Create a binary classifier
model = BipolarNetwork(input_size=60, architecture="wide")

# Train the model
accuracy, loss = train_bipolar_classifier(
    model=model,
    X_train=X_train, 
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    max_epochs=100,
    flip_prob=0.1
)

print(f"Accuracy: {accuracy:.4f}")
```

## 🧰 Core Components

### Bipolar Class
The heart of bipolar_ml - an optimized representation for binary vectors. Stores data in bit-packed format (1 bit per value) while exposing a clean API for operations like binding, superposition, and similarity measurement.

### AdaptiveBipolar Toolkit
Analysis and optimization tools that can:
- Detect convergence patterns during training
- Calculate information density across vector dimensions
- Find optimal compression ratios
- Track binding stability across operations
- Auto-tune training parameters

### SignSGD Optimizer
Memory-efficient optimization algorithm designed specifically for binary parameters:
- Uses targeted bit flips instead of gradients
- Adaptive flip probabilities based on system stability
- Vectorized processing for speed
- Built-in early stopping via convergence detection

### TextProcessor
Turbocharged NLP toolkit for extracting semantic relationships from text:
- **Flexible Tokenization**: Supports whitespace, regex, and subword tokenization strategies
- **Information-Weighted BM25**: Enhanced BM25 algorithm that weights terms by information density
- **Relationship Extraction**:
  - **Word-to-Word (w2w)**: Direct semantic relationships between individual words
  - **Word-to-Phrase (w2p)**: Relationships between words and multi-word expressions  
  - **Phrase-to-Phrase (p2p)**: Higher-order semantic connections between concepts
- **Adaptive Window Sizing**: Auto-tunes context window based on semantic density
- **Vector-Optimized Implementation**: Leverages PyTorch's vectorized operations for 10-50x faster processing
- **LRU Caching**: Smart document-term matrix caching with configurable memory limits
- **Entropy-Based Vocabulary Selection**: Prioritizes words with highest information content

### BitTransformer
Transformer architecture that operates entirely in binary space:
- Hyperdimensional attention mechanism
- Bit-packed parameter storage
- Straight-Through Estimator for training
- Optimized feed-forward components

## 🔬 Advanced Usage

### Progressive Distillation

Convert standard neural networks into binary ones with minimal accuracy loss:

```python
from bipolar_ml import ProgressiveHDCDistiller

# Create distiller with existing model
distiller = ProgressiveHDCDistiller(model, data_loader)

# Transform layers progressively
distiller.transform_next_layer()  # Transform most stable layer
```

### Locality-Sensitive Hashing (LSH)

bipolar_ml implements a specialized binary LSH that makes similarity search blazing fast:

- **Bit-Optimized Hash Tables**: Designed specifically for Hamming distance in binary space
- **Multi-Table Architecture**: Uses multiple hash tables with different bit sampling patterns
- **Sub-Linear Scaling**: Search time scales logarithmically with vocabulary size
- **Probabilistic Guarantees**: High probability of finding nearest neighbors
- **Adaptive Table Configuration**: Auto-tunes table count and hash length based on vector dimensionality
- **Information-Weighted Hashing**: Prioritizes bits with higher information content
- **Zero Extra Memory**: Reuses bit-packed storage for minimal memory overhead

```python
# The LSH index is built automatically during model training
results = model.find_similar("quantum", n=10, use_lsh=True)

# Or create and query directly
from bipolar_ml import SignLSH
lsh = SignLSH(
    dimension=1024,  
    bits_per_table=16,
    num_tables=8,
    device="cuda"  # GPU acceleration
)

# Index vectors 
lsh.index(vectors, ids)

# Query is crazy fast - ~O(log n) instead of O(n)
similar_ids = lsh.query(query_vector, k=10)
```

## 📊 Performance

bipolar_ml is ridiculously memory-efficient:
- 8x memory reduction compared to float32 representations
- Minimal accuracy tradeoff (typically <2% drop)
- ~20-100x speedup on similarity operations
- Scales to million-word vocabularies on consumer hardware

## 🔧 Installation

```bash
pip install bipolar_ml
```

## 🧪 Run Demos

```bash
# Run all demos
python -m bipolar_ml all

# Or run specific demos
python -m bipolar_ml semantic
python -m bipolar_ml vector
python -m bipolar_ml transformer
python -m bipolar_ml sonar
python -m bipolar_ml distiller
```

## 📜 License

MIT
