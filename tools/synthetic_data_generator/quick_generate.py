#!/usr/bin/env python3
"""
Quick Synthetic Data Generator for 1M Node Test

Generates 950K synthetic memories with realistic embeddings for production validation.
Uses deterministic clustering based on semantic topic models.

Usage:
    python quick_generate.py --count 950000 --output data/synthetic_memories.jsonl
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm


def generate_synthetic_memories(count: int, num_centroids: int = 100, std_dev: float = 0.1, seed: int = 42):
    """
    Generate synthetic memories clustered around topic centroids

    Args:
        count: Number of memories to generate
        num_centroids: Number of topic clusters
        std_dev: Clustering spread
        seed: Random seed for reproducibility

    Returns:
        List of memory dicts with content and embeddings
    """
    logging.info(f"Generating {count:,} synthetic memories with {num_centroids} centroids")

    rng = np.random.RandomState(seed)

    # Generate centroids in semantic space
    centroids = rng.randn(num_centroids, 768).astype(np.float32)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / norms

    logging.info(f"Generated {num_centroids} topic centroids")

    # Topic templates for realistic content
    topics = [
        "artificial intelligence", "machine learning", "neural networks",
        "deep learning", "cognitive architecture", "memory systems",
        "pattern recognition", "knowledge representation", "reasoning",
        "natural language processing", "computer vision", "robotics",
        "science", "technology", "history", "geography", "mathematics",
        "physics", "biology", "chemistry", "literature", "art", "music",
        "sports", "entertainment", "culture", "society", "politics"
    ]

    verbs = [
        "enables", "requires", "demonstrates", "implements", "utilizes",
        "optimizes", "integrates", "combines", "enhances", "supports",
        "explores", "analyzes", "describes", "explains", "studies"
    ]

    objects = [
        "semantic understanding", "contextual awareness", "pattern completion",
        "knowledge graphs", "episodic memory", "semantic memory",
        "working memory", "long-term retention", "activation spreading",
        "consolidation processes", "forgetting curves", "confidence scoring",
        "complex systems", "emergent behavior", "adaptive learning"
    ]

    memories = []

    for i in tqdm(range(count), desc="Generating memories"):
        # Assign to cluster
        cluster_idx = rng.randint(0, num_centroids)
        centroid = centroids[cluster_idx]

        # Add Gaussian noise
        noise = rng.normal(0, std_dev, size=centroid.shape)
        embedding = centroid + noise

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Generate content
        topic = rng.choice(topics)
        verb = rng.choice(verbs)
        obj = rng.choice(objects)

        content = f"Synthetic memory {i:07d}: {topic.capitalize()} {verb} {obj}."

        # Random confidence
        confidence = 0.7 + rng.uniform(-0.1, 0.1)
        confidence = np.clip(confidence, 0.5, 1.0)

        memories.append({
            "content": content,
            "embedding": embedding.tolist(),
            "confidence": float(confidence)
        })

    return memories


def save_memories_jsonl(memories, output_path: Path):
    """Save memories in JSONL format"""
    logging.info(f"Saving {len(memories):,} memories to {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for memory in tqdm(memories, desc="Writing memories"):
            f.write(json.dumps(memory) + '\n')

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logging.info(f"Saved {len(memories):,} memories ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Quick synthetic data generator for 1M node test")
    parser.add_argument('--count', type=int, default=950000, help='Number of synthetic memories')
    parser.add_argument('--centroids', type=int, default=100, help='Number of topic centroids')
    parser.add_argument('--std-dev', type=float, default=0.1, help='Clustering std dev')
    parser.add_argument('--output', type=Path, default='data/synthetic_memories.jsonl', help='Output file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        # Generate memories
        memories = generate_synthetic_memories(
            args.count,
            args.centroids,
            args.std_dev,
            args.seed
        )

        # Save to file
        save_memories_jsonl(memories, args.output)

        logging.info("Synthetic data generation complete")
        return 0

    except Exception as e:
        logging.error(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
