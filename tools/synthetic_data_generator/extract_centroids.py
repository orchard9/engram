#!/usr/bin/env python3
"""
Centroid Extraction Tool for Engram

Extracts embedding centroids from existing Engram memories using k-means clustering.
These centroids are used to generate realistic synthetic data clustered around
actual semantic patterns.

Usage:
    python extract_centroids.py --endpoint http://localhost:7432 --num-centroids 100 --output centroids.npy
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import httpx
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm


def fetch_all_embeddings(endpoint: str, batch_size: int = 1000) -> Tuple[List[str], np.ndarray]:
    """
    Fetch all embeddings from Engram database

    Args:
        endpoint: Engram API endpoint
        batch_size: Number of memories to fetch per request

    Returns:
        Tuple of (memory_ids, embeddings_array)
    """
    logging.info(f"Fetching embeddings from {endpoint}")

    client = httpx.Client(timeout=30.0)

    # Get all memories (paginated if needed)
    # Note: This assumes GET /api/v1/memories returns all memories
    # If pagination is needed, this would need to be updated
    response = client.get(f"{endpoint}/api/v1/memories")
    response.raise_for_status()

    data = response.json()
    memories = data.get('memories', [])

    if not memories:
        raise ValueError("No memories found in database")

    logging.info(f"Found {len(memories)} memories")

    # Extract embeddings
    memory_ids = []
    embeddings = []

    for memory in tqdm(memories, desc="Extracting embeddings"):
        if 'embedding' in memory and memory['embedding']:
            memory_ids.append(memory.get('id', 'unknown'))
            embeddings.append(memory['embedding'])

    embeddings_array = np.array(embeddings, dtype=np.float32)

    logging.info(f"Extracted {len(embeddings)} embeddings of dimension {embeddings_array.shape[1]}")

    return memory_ids, embeddings_array


def compute_centroids(embeddings: np.ndarray, num_centroids: int, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cluster centroids using MiniBatch k-means

    Args:
        embeddings: Array of embeddings (N, 768)
        num_centroids: Number of centroids to compute
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (centroids, labels)
    """
    logging.info(f"Computing {num_centroids} centroids using MiniBatch k-means")

    # Use MiniBatchKMeans for efficiency with large datasets
    kmeans = MiniBatchKMeans(
        n_clusters=num_centroids,
        random_state=random_state,
        batch_size=1024,
        max_iter=100,
        verbose=1
    )

    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_

    # Print cluster statistics
    unique, counts = np.unique(labels, return_counts=True)
    logging.info(f"Cluster sizes: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")

    return centroids, labels


def save_centroids(centroids: np.ndarray, output_path: Path, metadata: dict = None):
    """
    Save centroids to disk with optional metadata

    Args:
        centroids: Centroid array (num_centroids, 768)
        output_path: Output file path
        metadata: Optional metadata dict
    """
    logging.info(f"Saving {len(centroids)} centroids to {output_path}")

    # Save centroids as numpy array
    np.save(output_path, centroids)

    # Save metadata if provided
    if metadata:
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logging.info(f"Saved metadata to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract embedding centroids from Engram")
    parser.add_argument('--endpoint', default='http://localhost:7432', help='Engram API endpoint')
    parser.add_argument('--num-centroids', type=int, default=100, help='Number of centroids to compute')
    parser.add_argument('--output', type=Path, default='centroids.npy', help='Output file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        # Fetch embeddings from Engram
        memory_ids, embeddings = fetch_all_embeddings(args.endpoint)

        # Compute centroids
        centroids, labels = compute_centroids(embeddings, args.num_centroids, args.seed)

        # Save centroids
        metadata = {
            'num_centroids': args.num_centroids,
            'num_source_embeddings': len(embeddings),
            'embedding_dim': embeddings.shape[1],
            'seed': args.seed,
            'endpoint': args.endpoint
        }
        save_centroids(centroids, args.output, metadata)

        logging.info("Centroid extraction complete")
        logging.info(f"Generated {len(centroids)} centroids from {len(embeddings)} embeddings")

        return 0

    except Exception as e:
        logging.error(f"Centroid extraction failed: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
