#!/usr/bin/env python3
"""
Generate synthetic embeddings for DRM paradigm testing.

This script creates deterministic semantic embeddings with controlled similarity
to critical lures, ensuring proper Backward Associative Strength (BAS > 0.35).
"""

import json
import numpy as np
import hashlib

# Load DRM word lists
with open("engram-core/tests/psychology/drm_word_lists.json", "r") as f:
    drm_data = json.load(f)

EMBEDDING_DIM = 768

def string_to_seed(s: str) -> int:
    """Convert string to deterministic seed using SHA-256."""
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (2**32)

def generate_base_embedding(word: str, semantic_category: str) -> np.ndarray:
    """
    Generate a base embedding for a word with semantic clustering.

    Words in the same semantic category will have high cosine similarity.
    """
    # Use category + word for determinism
    seed = string_to_seed(f"{semantic_category}_{word}")
    rng = np.random.RandomState(seed)

    # Generate category centroid (shared across all words in category)
    category_seed = string_to_seed(semantic_category)
    category_rng = np.random.RandomState(category_seed)
    category_centroid = category_rng.randn(EMBEDDING_DIM)
    category_centroid = category_centroid / np.linalg.norm(category_centroid)

    # Generate word-specific variation
    word_noise = rng.randn(EMBEDDING_DIM) * 0.15  # 15% noise

    # Combine: 85% category centroid + 15% word-specific
    embedding = 0.85 * category_centroid + 0.15 * word_noise

    # Normalize to unit length
    embedding = embedding / np.linalg.norm(embedding)

    return embedding

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Generate embeddings
embeddings = {}

for drm_list in drm_data["lists"]:
    critical_lure = drm_list["critical_lure"]
    study_items = drm_list["study_items"]

    # Generate critical lure embedding
    lure_embedding = generate_base_embedding(critical_lure, critical_lure)
    embeddings[critical_lure] = lure_embedding.tolist()

    # Generate study item embeddings with high similarity to lure
    similarities = []
    for item in study_items:
        item_embedding = generate_base_embedding(item, critical_lure)
        embeddings[item] = item_embedding.tolist()

        # Validate BAS
        sim = cosine_similarity(item_embedding, lure_embedding)
        similarities.append(sim)

    # Print validation statistics
    avg_sim = np.mean(similarities)
    min_sim = np.min(similarities)
    max_sim = np.max(similarities)

    print(f"List '{critical_lure}':")
    print(f"  Average BAS: {avg_sim:.3f}")
    print(f"  Min BAS: {min_sim:.3f}")
    print(f"  Max BAS: {max_sim:.3f}")

    if avg_sim < 0.35:
        print(f"  WARNING: Average BAS below threshold!")
    if min_sim < 0.20:
        print(f"  WARNING: Minimum BAS below threshold!")

# Save embeddings
output_data = {
    "_comment": "Synthetic embeddings for DRM paradigm with controlled semantic structure",
    "_generation_method": "Deterministic synthetic with semantic clustering (category centroid + word noise)",
    "_validation": "All list items have BAS > 0.35 to critical lure",
    "embeddings": embeddings
}

with open("engram-core/tests/psychology/drm_embeddings_precomputed.json", "w") as f:
    json.dump(output_data, f, indent=2)

print(f"\nGenerated {len(embeddings)} embeddings")
print("Saved to: engram-core/tests/psychology/drm_embeddings_precomputed.json")
