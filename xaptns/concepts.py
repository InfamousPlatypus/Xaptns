import numpy as np
import torch
from typing import List, Dict, Any

class ConceptMapper:
    """
    The Decoder: Maps coordinates to human-readable concepts using Sparse Autoencoders (SAE).
    """
    def __init__(self, sae_weights_path: str = None):
        # In a real scenario, we'd load a pre-trained SAE (e.g., from Gemma Scope)
        # For MVP, we'll use a toy dictionary if no path is provided.
        self.dictionary = self._load_toy_dictionary()

    def _load_toy_dictionary(self):
        # Mock dictionary: latent feature index -> label
        return {
            10: "Quantum Annealing",
            42: "Transformer Attention",
            105: "Persistent Homology",
            256: "Differential Privacy",
            512: "Protein Folding"
        }

    def decode(self, vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Disentangles the vector into human-readable concept labels.
        """
        # Simulated SAE activation:
        # In reality, this would be: activations = ReLU( (v - b_dec) @ W_enc + b_enc )
        # Here we'll just mock it.

        # Mocking activations based on vector indices for reproducibility in demo
        indices = np.argsort(vector.flatten())[-top_k:]

        concepts = []
        for idx in indices:
            label = self.dictionary.get(idx, f"Latent Concept #{idx}")
            concepts.append({
                "index": int(idx),
                "label": label,
                "activation": float(vector.flatten()[idx])
            })

        return concepts

if __name__ == "__main__":
    mapper = ConceptMapper()
    test_vec = np.random.rand(768)
    test_vec[10] = 2.0
    test_vec[42] = 1.5
    concepts = mapper.decode(test_vec)
    print("Decoded concepts:")
    for c in concepts:
        print(f" - {c['label']} (Activation: {c['activation']:.2f})")
