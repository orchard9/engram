#!/usr/bin/env python3
"""
Bulk Ingestion Tool for Engram

Fast bulk ingestion of synthetic memories from JSONL files with parallel requests
and progress tracking.

Usage:
    python bulk_ingest.py --input synthetic_memories.jsonl --endpoint http://localhost:7432 --concurrent 20
"""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import List, Dict

import httpx
from tqdm import tqdm


class BulkIngestor:
    """Async bulk ingestor with rate limiting and error handling"""

    def __init__(self, endpoint: str, concurrent_requests: int = 20):
        self.endpoint = endpoint.rstrip('/')
        self.semaphore = asyncio.Semaphore(concurrent_requests)
        self.client = None
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'start_time': 0
        }

    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.stats['start_time'] = time.time()
        return self

    async def __aexit__(self, *args):
        if self.client:
            await self.client.aclose()

    async def store_memory(self, memory: Dict) -> bool:
        """Store a single memory"""
        async with self.semaphore:
            try:
                response = await self.client.post(
                    f"{self.endpoint}/api/v1/memories",
                    json=memory
                )
                response.raise_for_status()
                return True
            except httpx.HTTPError as e:
                logging.debug(f"Failed to store memory: {e}")
                return False

    async def ingest_batch(self, memories: List[Dict], progress_bar=None):
        """Ingest a batch of memories in parallel"""
        tasks = [self.store_memory(memory) for memory in memories]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            self.stats['total'] += 1
            if result is True:
                self.stats['success'] += 1
            else:
                self.stats['failed'] += 1

        if progress_bar:
            progress_bar.update(len(memories))
            elapsed = time.time() - self.stats['start_time']
            throughput = self.stats['success'] / elapsed if elapsed > 0 else 0
            progress_bar.set_postfix({
                'throughput': f'{throughput:.0f} mem/s',
                'errors': self.stats['failed']
            })

    def print_summary(self):
        """Print ingestion summary"""
        elapsed = time.time() - self.stats['start_time']
        throughput = self.stats['success'] / elapsed if elapsed > 0 else 0

        print("\n" + "="*60)
        print("BULK INGESTION COMPLETE")
        print("="*60)
        print(f"Total memories:     {self.stats['total']:,}")
        print(f"Successfully ingested: {self.stats['success']:,}")
        print(f"Failed:             {self.stats['failed']:,}")
        print(f"Elapsed time:       {elapsed:.1f}s")
        print(f"Throughput:         {throughput:.0f} memories/sec")
        print("="*60)


def load_jsonl_memories(input_path: Path, limit: int = None) -> List[Dict]:
    """Load memories from JSONL file"""
    logging.info(f"Loading memories from {input_path}")

    memories = []
    with open(input_path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            memory = json.loads(line)
            memories.append(memory)

    logging.info(f"Loaded {len(memories):,} memories")
    return memories


async def main():
    parser = argparse.ArgumentParser(description="Bulk ingest synthetic memories into Engram")
    parser.add_argument('--input', type=Path, required=True, help='Input JSONL file')
    parser.add_argument('--endpoint', default='http://localhost:7432', help='Engram API endpoint')
    parser.add_argument('--concurrent', type=int, default=20, help='Concurrent API requests')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for parallel ingestion')
    parser.add_argument('--limit', type=int, help='Limit number of memories to ingest')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        # Load memories
        memories = load_jsonl_memories(args.input, args.limit)

        # Ingest in batches
        async with BulkIngestor(args.endpoint, args.concurrent) as ingestor:
            # Verify Engram is reachable
            try:
                response = await ingestor.client.get(f"{args.endpoint}/health")
                response.raise_for_status()
                logging.info(f"Connected to Engram at {args.endpoint}")
            except Exception as e:
                logging.error(f"Failed to connect to Engram: {e}")
                return 1

            # Ingest batches with progress bar
            progress = tqdm(total=len(memories), desc="Ingesting memories", unit="mem")

            for i in range(0, len(memories), args.batch_size):
                batch = memories[i:i + args.batch_size]
                await ingestor.ingest_batch(batch, progress)

            progress.close()

            # Print summary
            ingestor.print_summary()

        return 0 if ingestor.stats['failed'] == 0 else 1

    except Exception as e:
        logging.error(f"Bulk ingestion failed: {e}")
        return 1


if __name__ == '__main__':
    exit(asyncio.run(main()))
