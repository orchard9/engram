#!/usr/bin/env python3
"""
Streaming Bulk Ingestion Tool for Engram

Memory-efficient bulk ingestion that streams JSONL files without loading everything into RAM.
Designed for large datasets (100K-1M+ memories).

Usage:
    python streaming_ingest.py --input synthetic_memories.jsonl --endpoint http://localhost:7432
"""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, AsyncIterator

import httpx
from tqdm import tqdm


class StreamingIngestor:
    """Async bulk ingestor with streaming and rate limiting"""

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

    async def ingest_stream(self, memory_stream: AsyncIterator[Dict], total_count: int = None):
        """Ingest memories from an async stream"""
        progress = tqdm(total=total_count, desc="Ingesting memories", unit="mem")
        batch = []
        batch_size = 1000

        async for memory in memory_stream:
            batch.append(memory)

            if len(batch) >= batch_size:
                await self._process_batch(batch, progress)
                batch = []

        # Process final batch
        if batch:
            await self._process_batch(batch, progress)

        progress.close()

    async def _process_batch(self, batch, progress_bar):
        """Process a batch of memories in parallel"""
        tasks = [self.store_memory(memory) for memory in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            self.stats['total'] += 1
            if result is True:
                self.stats['success'] += 1
            else:
                self.stats['failed'] += 1

        if progress_bar is not None:
            progress_bar.update(len(batch))
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
        print(f"Total memories:        {self.stats['total']:,}")
        print(f"Successfully ingested: {self.stats['success']:,}")
        print(f"Failed:                {self.stats['failed']:,}")
        print(f"Elapsed time:          {elapsed/60:.1f} minutes ({elapsed:.0f}s)")
        print(f"Throughput:            {throughput:.0f} memories/sec")
        print("="*60)


async def stream_jsonl_file(file_path: Path, limit: int = None) -> AsyncIterator[Dict]:
    """Asynchronously stream JSONL file line by line"""
    count = 0
    with open(file_path) as f:
        for line in f:
            if limit and count >= limit:
                break
            try:
                memory = json.loads(line)
                yield memory
                count += 1
            except json.JSONDecodeError as e:
                logging.warning(f"Skipping invalid JSON line {count}: {e}")
                continue


def count_lines(file_path: Path) -> int:
    """Count lines in file for progress tracking"""
    logging.info(f"Counting lines in {file_path}...")
    with open(file_path) as f:
        count = sum(1 for _ in f)
    logging.info(f"Found {count:,} lines")
    return count


async def main():
    parser = argparse.ArgumentParser(description="Streaming bulk ingest for large datasets")
    parser.add_argument('--input', type=Path, required=True, help='Input JSONL file')
    parser.add_argument('--endpoint', default='http://localhost:7432', help='Engram API endpoint')
    parser.add_argument('--concurrent', type=int, default=20, help='Concurrent API requests')
    parser.add_argument('--limit', type=int, help='Limit number of memories to ingest')
    parser.add_argument('--skip-count', action='store_true', help='Skip line counting (faster startup)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        # Count lines for progress bar (unless skipped)
        total_count = None
        if not args.skip_count:
            total_count = count_lines(args.input)
            if args.limit and args.limit < total_count:
                total_count = args.limit

        # Ingest using streaming
        async with StreamingIngestor(args.endpoint, args.concurrent) as ingestor:
            # Verify Engram is reachable
            try:
                response = await ingestor.client.get(f"{args.endpoint}/health")
                response.raise_for_status()
                logging.info(f"Connected to Engram at {args.endpoint}")
            except Exception as e:
                logging.error(f"Failed to connect to Engram: {e}")
                return 1

            # Stream and ingest
            memory_stream = stream_jsonl_file(args.input, args.limit)
            await ingestor.ingest_stream(memory_stream, total_count)

            # Print summary
            ingestor.print_summary()

        return 0 if ingestor.stats['failed'] == 0 else 1

    except Exception as e:
        logging.error(f"Bulk ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(asyncio.run(main()))
