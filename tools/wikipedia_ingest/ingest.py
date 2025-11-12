#!/usr/bin/env python3
"""
Wikipedia Ingestion Tool for Engram

Production-quality tool for ingesting Wikipedia articles into Engram
for realistic performance testing and demonstration.

Usage:
    python ingest.py --endpoint http://localhost:7432 --limit 1000
"""

import argparse
import asyncio
import gzip
import json
import logging
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
from urllib.request import urlretrieve

import httpx
import mwparserfromhell
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# Constants
SIMPLE_WIKI_URL = "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"
DEFAULT_BATCH_SIZE = 32
DEFAULT_CONCURRENT_REQUESTS = 10
MIN_ARTICLE_LENGTH = 50  # Lowered for Simple English Wikipedia
CHECKPOINT_INTERVAL = 1000


@dataclass
class Article:
    """Represents a parsed Wikipedia article"""
    title: str
    text: str
    timestamp: str
    id: int


@dataclass
class IngestStats:
    """Tracks ingestion statistics"""
    total_articles: int = 0
    ingested: int = 0
    failed: int = 0
    skipped: int = 0
    start_time: float = 0

    def throughput(self) -> float:
        """Calculate articles per second"""
        elapsed = time.time() - self.start_time
        return self.ingested / elapsed if elapsed > 0 else 0


class WikipediaParser:
    """Stream parser for Wikipedia XML dumps"""

    def __init__(self, dump_path: Path):
        self.dump_path = dump_path
        self.namespace = "{http://www.mediawiki.org/xml/export-0.11/}"

    def parse_articles(self, limit: Optional[int] = None) -> List[Article]:
        """
        Stream parse Wikipedia dump and yield articles

        Args:
            limit: Maximum number of articles to parse (None for all)

        Yields:
            Article objects
        """
        articles = []
        count = 0

        # Determine file opener based on extension
        if self.dump_path.suffix == '.bz2':
            import bz2
            opener = bz2.open
        elif self.dump_path.suffix == '.gz':
            opener = gzip.open
        else:
            opener = open

        with opener(self.dump_path, 'rb') as f:
            # Stream parse XML
            for event, elem in ET.iterparse(f, events=('end',)):
                if elem.tag == f'{self.namespace}page':
                    article = self._parse_page(elem)
                    if article and self._should_include(article):
                        articles.append(article)
                        count += 1

                        if limit and count >= limit:
                            break

                    # Clear element to free memory
                    elem.clear()

        return articles

    def _parse_page(self, page_elem) -> Optional[Article]:
        """Parse a single page element"""
        try:
            # Extract fields
            title_elem = page_elem.find(f'{self.namespace}title')
            id_elem = page_elem.find(f'{self.namespace}id')
            revision = page_elem.find(f'{self.namespace}revision')

            if revision is None:
                return None

            text_elem = revision.find(f'{self.namespace}text')
            timestamp_elem = revision.find(f'{self.namespace}timestamp')

            if any(x is None for x in [title_elem, id_elem, text_elem, timestamp_elem]):
                return None

            # Parse wikitext to plain text
            wikitext = text_elem.text or ""
            plain_text = self._wikitext_to_plain(wikitext)

            return Article(
                title=title_elem.text,
                text=plain_text,
                timestamp=timestamp_elem.text,
                id=int(id_elem.text)
            )
        except Exception as e:
            logging.warning(f"Failed to parse page: {e}")
            return None

    def _wikitext_to_plain(self, wikitext: str) -> str:
        """Convert wikitext to plain text"""
        try:
            # Parse wikitext
            parsed = mwparserfromhell.parse(wikitext)

            # Strip templates, references, etc.
            text = parsed.strip_code()

            # Clean up whitespace
            text = re.sub(r'\n\n+', '\n\n', text)
            text = text.strip()

            return text
        except Exception:
            # Fallback: basic cleaning
            text = re.sub(r'\[\[([^\]|]+\|)?([^\]]+)\]\]', r'\2', wikitext)
            text = re.sub(r'\{\{[^}]+\}\}', '', text)
            text = re.sub(r'<[^>]+>', '', text)
            return text.strip()

    def _should_include(self, article: Article) -> bool:
        """Filter articles based on quality criteria"""
        # Skip short articles (stubs)
        if len(article.text) < MIN_ARTICLE_LENGTH:
            return False

        # Skip redirects
        if article.text.lower().startswith('#redirect'):
            return False

        # Skip disambiguation pages
        if 'disambiguation' in article.title.lower():
            return False

        # Skip Wikipedia meta pages
        if any(prefix in article.title for prefix in ['Wikipedia:', 'Template:', 'Category:', 'File:', 'Help:']):
            return False

        return True


class EmbeddingGenerator:
    """Generate embeddings using sentence-transformers"""

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        logging.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # Validate dimension matches Engram
        if self.embedding_dim != 768:
            raise ValueError(f"Model dimension {self.embedding_dim} != 768 (Engram EMBEDDING_DIM)")

    def generate_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts

        Args:
            texts: List of text strings

        Returns:
            Array of embeddings (batch_size, 768)
        """
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    def generate(self, text: str) -> List[float]:
        """Generate embedding for single text"""
        embedding = self.model.encode([text], show_progress_bar=False, convert_to_numpy=True)[0]
        return embedding.tolist()


class EngramClient:
    """Async HTTP client for Engram API"""

    def __init__(self, endpoint: str, concurrent_requests: int = DEFAULT_CONCURRENT_REQUESTS):
        self.endpoint = endpoint.rstrip('/')
        self.semaphore = asyncio.Semaphore(concurrent_requests)
        self.client = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, *args):
        if self.client:
            await self.client.aclose()

    async def store_memory(self, content: str, embedding: List[float], confidence: float = 0.8) -> bool:
        """
        Store a memory in Engram

        Args:
            content: Memory text content
            embedding: 768-dimensional embedding vector
            confidence: Confidence score (0.0-1.0)

        Returns:
            True if successful, False otherwise
        """
        async with self.semaphore:
            try:
                response = await self.client.post(
                    f"{self.endpoint}/api/v1/memories",
                    json={
                        "content": content,
                        "embedding": embedding,
                        "confidence": confidence
                    }
                )
                response.raise_for_status()
                return True
            except httpx.HTTPError as e:
                logging.debug(f"Failed to store memory: {e}")
                return False

    async def get_memory_count(self) -> int:
        """Get current count of memories in database"""
        try:
            response = await self.client.get(f"{self.endpoint}/api/v1/memories")
            response.raise_for_status()
            memories = response.json()
            return len(memories.get('memories', []))
        except httpx.HTTPError:
            return 0


class CheckpointManager:
    """Manages ingestion checkpoints for resume capability"""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, article_ids: set, stats: IngestStats):
        """Save checkpoint with ingested article IDs and stats"""
        checkpoint = {
            'article_ids': list(article_ids),
            'stats': {
                'total_articles': stats.total_articles,
                'ingested': stats.ingested,
                'failed': stats.failed,
                'skipped': stats.skipped
            },
            'timestamp': time.time()
        }

        checkpoint_path = self.checkpoint_dir / 'latest.json'
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f)

    def load(self) -> Optional[Dict[str, Any]]:
        """Load latest checkpoint"""
        checkpoint_path = self.checkpoint_dir / 'latest.json'
        if not checkpoint_path.exists():
            return None

        with open(checkpoint_path) as f:
            return json.load(f)


async def ingest_articles(
    articles: List[Article],
    embedder: EmbeddingGenerator,
    client: EngramClient,
    stats: IngestStats,
    batch_size: int,
    checkpoint_mgr: Optional[CheckpointManager] = None,
    resume_article_ids: Optional[set] = None
):
    """
    Ingest articles into Engram with batched embedding generation and parallel requests

    Args:
        articles: List of parsed Wikipedia articles
        embedder: Embedding generator
        client: Engram API client
        stats: Statistics tracker
        batch_size: Batch size for embedding generation
        checkpoint_mgr: Checkpoint manager for resume support
        resume_article_ids: Set of already-ingested article IDs to skip
    """
    resume_article_ids = resume_article_ids or set()
    ingested_ids = set(resume_article_ids)

    progress = tqdm(total=len(articles), desc="Ingesting articles", unit="articles")
    progress.update(len(resume_article_ids))

    for i in range(0, len(articles), batch_size):
        batch = articles[i:i + batch_size]

        # Skip already ingested articles
        batch = [a for a in batch if a.id not in resume_article_ids]
        if not batch:
            continue

        # Generate embeddings in batch
        texts = [f"{a.title}\n\n{a.text[:1000]}" for a in batch]  # Truncate long articles
        embeddings = embedder.generate_batch(texts)

        # Store memories in parallel
        tasks = [
            client.store_memory(
                content=f"{article.title}\n\n{article.text}",
                embedding=embedding.tolist(),
                confidence=min(0.9, 0.6 + len(article.text) / 10000)  # Higher confidence for longer articles
            )
            for article, embedding in zip(batch, embeddings)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Update stats
        for article, success in zip(batch, results):
            if success is True:
                stats.ingested += 1
                ingested_ids.add(article.id)
            elif isinstance(success, Exception):
                stats.failed += 1
                logging.error(f"Failed to ingest '{article.title}': {success}")
            else:
                stats.failed += 1

        progress.update(len(batch))
        progress.set_postfix({
            'throughput': f'{stats.throughput():.1f} art/s',
            'errors': stats.failed
        })

        # Checkpoint periodically
        if checkpoint_mgr and stats.ingested % CHECKPOINT_INTERVAL == 0:
            checkpoint_mgr.save(ingested_ids, stats)

    progress.close()

    # Final checkpoint
    if checkpoint_mgr:
        checkpoint_mgr.save(ingested_ids, stats)


def download_wikipedia_dump(output_path: Path) -> bool:
    """Download Wikipedia dump if not already present"""
    if output_path.exists():
        logging.info(f"Wikipedia dump already exists: {output_path}")
        return True

    logging.info(f"Downloading Wikipedia dump from {SIMPLE_WIKI_URL}")
    logging.info("This may take several minutes (600MB download)...")

    try:
        def progress_hook(count, block_size, total_size):
            if total_size > 0:
                percent = count * block_size * 100 / total_size
                sys.stdout.write(f"\rDownload progress: {percent:.1f}%")
                sys.stdout.flush()

        urlretrieve(SIMPLE_WIKI_URL, output_path, reporthook=progress_hook)
        print()  # New line after progress
        logging.info(f"Download complete: {output_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to download Wikipedia dump: {e}")
        return False


async def main():
    parser = argparse.ArgumentParser(description="Ingest Wikipedia into Engram")
    parser.add_argument('--endpoint', default='http://localhost:7432', help='Engram API endpoint')
    parser.add_argument('--limit', type=int, help='Limit number of articles to ingest')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help='Embedding batch size')
    parser.add_argument('--concurrent', type=int, default=DEFAULT_CONCURRENT_REQUESTS, help='Concurrent API requests')
    parser.add_argument('--download-only', action='store_true', help='Only download Wikipedia dump')
    parser.add_argument('--resume', help='Resume from checkpoint file')
    parser.add_argument('--dump-path', type=Path, help='Path to Wikipedia dump (auto-download if not provided)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Create directories
    data_dir = Path(__file__).parent / 'data'
    data_dir.mkdir(exist_ok=True)

    checkpoint_dir = Path(__file__).parent / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)

    # Determine dump path
    dump_path = args.dump_path or (data_dir / 'simplewiki-latest-pages-articles.xml.bz2')

    # Download Wikipedia dump if needed
    if not download_wikipedia_dump(dump_path):
        return 1

    if args.download_only:
        logging.info("Download-only mode: Exiting")
        return 0

    # Load checkpoint if resuming
    checkpoint_mgr = CheckpointManager(checkpoint_dir)
    resume_checkpoint = None
    resume_article_ids = set()

    if args.resume:
        resume_checkpoint = checkpoint_mgr.load()
        if resume_checkpoint:
            resume_article_ids = set(resume_checkpoint['article_ids'])
            logging.info(f"Resuming from checkpoint: {len(resume_article_ids)} articles already ingested")

    # Parse Wikipedia dump
    logging.info(f"Parsing Wikipedia dump: {dump_path}")
    parser = WikipediaParser(dump_path)
    articles = parser.parse_articles(limit=args.limit)

    logging.info(f"Parsed {len(articles)} articles (after filtering)")

    if not articles:
        logging.error("No articles to ingest")
        return 1

    # Initialize embedder
    logging.info("Initializing embedding model (first run downloads ~420MB model)")
    embedder = EmbeddingGenerator()

    # Ingest articles
    stats = IngestStats(
        total_articles=len(articles),
        start_time=time.time()
    )

    async with EngramClient(args.endpoint, args.concurrent) as client:
        # Verify Engram is reachable
        try:
            count = await client.get_memory_count()
            logging.info(f"Connected to Engram (current memories: {count})")
        except Exception as e:
            logging.error(f"Failed to connect to Engram at {args.endpoint}: {e}")
            return 1

        # Ingest articles
        await ingest_articles(
            articles=articles,
            embedder=embedder,
            client=client,
            stats=stats,
            batch_size=args.batch_size,
            checkpoint_mgr=checkpoint_mgr,
            resume_article_ids=resume_article_ids
        )

    # Print summary
    elapsed = time.time() - stats.start_time
    print("\n" + "="*60)
    print("INGESTION COMPLETE")
    print("="*60)
    print(f"Total articles:     {stats.total_articles:,}")
    print(f"Successfully ingested: {stats.ingested:,}")
    print(f"Failed:             {stats.failed:,}")
    print(f"Elapsed time:       {elapsed:.1f}s")
    print(f"Throughput:         {stats.throughput():.1f} articles/sec")
    print("="*60)

    return 0 if stats.failed == 0 else 1


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
