#!/usr/bin/env python3
"""
Synthetic Data Generator for Engram

Generates synthetic memories with embeddings clustered around real centroids.
This creates a large dataset (100K-1M nodes) with realistic semantic structure
for performance testing.

Usage:
    python generate_synthetic.py --centroids centroids.npy --count 950000 --output synthetic_memories.jsonl
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm


def load_centroids(centroid_path: Path) -> np.ndarray:
    """Load centroids from disk"""
    logging.info(f"Loading centroids from {centroid_path}")
    centroids = np.load(centroid_path)
    logging.info(f"Loaded {len(centroids)} centroids of dimension {centroids.shape[1]}")
    return centroids


def generate_clustered_embeddings(
    centroids: np.ndarray,
    count: int,
    std_dev: float = 0.1,
    seed: int = 42
) -> np.ndarray:
    """
    Generate embeddings clustered around centroids

    Args:
        centroids: Centroid array (num_centroids, 768)
        count: Number of embeddings to generate
        std_dev: Standard deviation for Gaussian noise
        seed: Random seed

    Returns:
        Array of synthetic embeddings (count, 768)
    """
    logging.info(f"Generating {count:,} synthetic embeddings with std_dev={std_dev}")

    rng = np.random.RandomState(seed)

    # Assign each embedding to a cluster (uniform distribution)
    cluster_assignments = rng.randint(0, len(centroids), size=count)

    # Generate embeddings by adding Gaussian noise to centroids
    embeddings = np.zeros((count, centroids.shape[1]), dtype=np.float32)

    for i in tqdm(range(count), desc="Generating embeddings"):
        centroid_idx = cluster_assignments[i]
        centroid = centroids[centroid_idx]

        # Add Gaussian noise
        noise = rng.normal(0, std_dev, size=centroid.shape)
        embeddings[i] = centroid + noise

        # Normalize to unit length (common for sentence embeddings)
        norm = np.linalg.norm(embeddings[i])
        if norm > 0:
            embeddings[i] /= norm

    logging.info(f"Generated {len(embeddings):,} synthetic embeddings")

    return embeddings


def generate_synthetic_content(count: int, seed: int = 42) -> List[str]:
    """
    Generate synthetic content strings

    Args:
        count: Number of content strings to generate
        seed: Random seed

    Returns:
        List of synthetic content strings
    """
    logging.info(f"Generating {count:,} synthetic content strings")

    rng = np.random.RandomState(seed)

    # Templates for synthetic content
    topics = [
        "artificial intelligence", "machine learning", "neural networks",
        "deep learning", "cognitive architecture", "memory systems",
        "pattern recognition", "knowledge representation", "reasoning",
        "natural language processing", "computer vision", "robotics"
    ]

    verbs = [
        "enables", "requires", "demonstrates", "implements", "utilizes",
        "optimizes", "integrates", "combines", "enhances", "supports"
    ]

    objects = [
        "semantic understanding", "contextual awareness", "pattern completion",
        "knowledge graphs", "episodic memory", "semantic memory",
        "working memory", "long-term retention", "activation spreading",
        "consolidation processes", "forgetting curves", "confidence scoring"
    ]

    content_list = []
    for i in range(count):
        topic = rng.choice(topics)
        verb = rng.choice(verbs)
        obj = rng.choice(objects)

        content = f"Synthetic memory {i:07d}: {topic.capitalize()} {verb} {obj}."
        content_list.append(content)

    return content_list


def save_synthetic_memories(
    embeddings: np.ndarray,
    content_list: List[str],
    output_path: Path,
    base_confidence: float = 0.7
):
    """
    Save synthetic memories in JSONL format for bulk ingestion

    Args:
        embeddings: Embedding array (count, 768)
        content_list: List of content strings
        output_path: Output JSONL file path
        base_confidence: Base confidence score (adds small random variation)
    """
    logging.info(f"Saving {len(embeddings):,} synthetic memories to {output_path}")

    rng = np.random.RandomState(42)

    with open(output_path, 'w') as f:
        for i, (embedding, content) in enumerate(tqdm(
            zip(embeddings, content_list),
            total=len(embeddings),
            desc="Writing memories"
        )):
            # Add small random variation to confidence
            confidence = base_confidence + rng.uniform(-0.1, 0.1)
            confidence = np.clip(confidence, 0.5, 1.0)

            memory = {
                "content": content,
                "embedding": embedding.tolist(),
                "confidence": float(confidence)
            }

            f.write(json.dumps(memory) + '\n')

    logging.info(f"Saved {len(embeddings):,} memories to {output_path}")

    # Print file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logging.info(f"Output file size: {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Engram memories")
    parser.add_argument('--centroids', type=Path, required=True, help='Centroid file (*.npy)')
    parser.add_argument('--count', type=int, default=950000, help='Number of synthetic memories')
    parser.add_argument('--std-dev', type=float, default=0.1, help='Gaussian noise std dev')
    parser.add_argument('--output', type=Path, default='synthetic_memories.jsonl', help='Output JSONL file')
    parser.add_argument('--confidence', type=float, default=0.7, help='Base confidence score')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        # Load centroids
        centroids = load_centroids(args.centroids)

        # Generate embeddings
        embeddings = generate_clustered_embeddings(
            centroids,
            args.count,
            args.std_dev,
            args.seed
        )

        # Generate content
        content_list = generate_synthetic_content(args.count, args.seed)

        # Save memories
        save_synthetic_memories(
            embeddings,
            content_list,
            args.output,
            args.confidence
        )

        logging.info("Synthetic data generation complete")
        logging.info(f"Generated {args.count:,} memories from {len(centroids)} centroids")

        return 0

    except Exception as e:
        logging.error(f"Synthetic data generation failed: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
