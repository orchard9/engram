//! Approximate Annoy implementation for ANN benchmarks.
//!
//! This module implements a lightweight, Annoy-inspired random projection
//! forest so Engram's benchmarks can compare against a second baseline without
//! relying on mocks or external native dependencies. The algorithm follows the
//! Annoy design: build multiple trees by recursively splitting vectors with
//! random hyperplanes, then perform best-first traversal to gather candidate
//! neighbours.

use super::ann_common::AnnIndex;
use anyhow::{Result, ensure};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::convert::TryFrom;

/// Default leaf size before a subtree stops splitting.
const DEFAULT_LEAF_SIZE: usize = 64;
/// Minimum norm for a random hyperplane direction before considering it valid.
const MIN_NORMAL_NORM: f32 = 1.0e-6;
/// Base seed used for deterministic tree construction.
const BASE_TREE_SEED: u64 = 0xA77A_D00D_u64;

/// Annoy-style ANN index implemented in pure Rust for reproducible benchmarks.
pub struct AnnoyAnnIndex {
    dimension: usize,
    n_trees: usize,
    leaf_size: usize,
    search_k: usize,
    vectors: Vec<[f32; 768]>,
    trees: Vec<AnnoyTree>,
}

impl AnnoyAnnIndex {
    /// Construct a new Annoy-style index using default parameters.
    pub fn new(dimension: usize, n_trees: usize) -> Result<Self> {
        Self::with_params(dimension, n_trees, DEFAULT_LEAF_SIZE, 0)
    }

    /// Construct a new index with full parameter control.
    pub fn with_params(
        dimension: usize,
        n_trees: usize,
        leaf_size: usize,
        search_k: usize,
    ) -> Result<Self> {
        ensure!(dimension > 0, "Annoy dimension must be greater than zero");
        ensure!(n_trees > 0, "Annoy requires at least one tree");
        ensure!(leaf_size > 0, "Leaf size must be greater than zero");

        let derived_search_k = if search_k == 0 {
            // Following Annoy defaults, explore roughly n_trees * leaf_size * 2 nodes.
            n_trees
                .saturating_mul(leaf_size)
                .saturating_mul(2)
                .max(leaf_size)
        } else {
            search_k
        };

        Ok(Self {
            dimension,
            n_trees,
            leaf_size,
            search_k: derived_search_k,
            vectors: Vec::new(),
            trees: Vec::new(),
        })
    }

    /// Compute cosine similarity between two vectors.
    fn cosine_similarity(a: &[f32; 768], b: &[f32; 768]) -> f32 {
        let mut dot_product = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;

        for (x, y) in a.iter().zip(b.iter()) {
            dot_product += x * y;
            norm_a += x * x;
            norm_b += y * y;
        }

        if norm_a <= 0.0 || norm_b <= 0.0 {
            return 0.0;
        }

        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom <= 0.0 {
            0.0
        } else {
            dot_product / denom
        }
    }
}

impl AnnIndex for AnnoyAnnIndex {
    fn build(&mut self, vectors: &[[f32; 768]]) -> Result<()> {
        ensure!(
            self.dimension == 768,
            "Annoy index configured for {0} dimensions but dataset uses 768",
            self.dimension
        );

        self.vectors.clear();
        self.vectors.extend_from_slice(vectors);
        self.trees.clear();

        if self.vectors.is_empty() {
            return Ok(());
        }

        let indices: Vec<usize> = (0..self.vectors.len()).collect();
        let max_depth = calculate_max_depth(self.vectors.len(), self.leaf_size);

        for tree_id in 0..self.n_trees {
            let offset = u64::try_from(tree_id).unwrap_or(u64::MAX);
            let seed = BASE_TREE_SEED.wrapping_add(offset);
            let builder = AnnoyTreeBuilder::new(
                &self.vectors,
                self.dimension,
                self.leaf_size,
                max_depth,
                seed,
            );
            let tree = builder.build(&indices)?;
            self.trees.push(tree);
        }

        Ok(())
    }

    fn search(&mut self, query: &[f32; 768], k: usize) -> Vec<(usize, f32)> {
        if self.vectors.is_empty() || self.trees.is_empty() || k == 0 {
            return Vec::new();
        }

        let mut candidates = HashSet::new();
        let candidate_limit = usize::min(self.search_k.max(k), self.vectors.len());

        for tree in &self.trees {
            tree.collect_candidates(query, self.search_k, &mut candidates);
            if candidates.len() >= candidate_limit {
                break;
            }
        }

        if candidates.is_empty() {
            // Degenerate case: fall back to all vectors to avoid empty result.
            candidates.extend(0..self.vectors.len());
        }

        let mut scored: Vec<(usize, f32)> = candidates
            .into_iter()
            .map(|idx| (idx, Self::cosine_similarity(query, &self.vectors[idx])))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        scored.truncate(k);
        scored
    }

    fn memory_usage(&self) -> usize {
        let vector_bytes = self.vectors.len() * self.dimension * std::mem::size_of::<f32>();

        let mut tree_bytes = 0usize;
        for tree in &self.trees {
            tree_bytes = tree_bytes.saturating_add(tree.memory_usage());
        }

        vector_bytes.saturating_add(tree_bytes)
    }

    fn name(&self) -> &'static str {
        "Annoy"
    }
}

/// Compute the maximum recursion depth for tree construction.
fn calculate_max_depth(len: usize, leaf_size: usize) -> usize {
    if len <= 1 || leaf_size == 0 {
        return 0;
    }

    let ratio = (len as f64 / leaf_size as f64).max(1.0);
    // Approximate Annoy depth: 2 * log2(n / leaf_size).
    (2.0 * ratio.log2()).ceil() as usize + 1
}

/// Single Annoy-style tree storing recursively split nodes.
struct AnnoyTree {
    nodes: Vec<TreeNode>,
    root: usize,
}

impl AnnoyTree {
    fn collect_candidates(
        &self,
        query: &[f32; 768],
        search_k: usize,
        candidates: &mut HashSet<usize>,
    ) {
        let mut heap = BinaryHeap::new();
        heap.push(SearchNode {
            priority: 0.0,
            node_index: self.root,
        });

        let mut visited = 0usize;
        let visit_limit = search_k.max(1);

        while let Some(node) = heap.pop() {
            if visited >= visit_limit {
                break;
            }
            visited = visited.saturating_add(1);

            match &self.nodes[node.node_index] {
                TreeNode::Leaf { points } => {
                    candidates.extend(points.iter().copied());
                }
                TreeNode::Split {
                    normal,
                    threshold,
                    left,
                    right,
                } => {
                    let projection = dot_product(normal, query);
                    let (near, far) = if projection <= *threshold {
                        (*left, *right)
                    } else {
                        (*right, *left)
                    };

                    heap.push(SearchNode {
                        priority: 0.0,
                        node_index: near,
                    });

                    let margin = (projection - threshold).abs();
                    heap.push(SearchNode {
                        priority: margin,
                        node_index: far,
                    });
                }
            }
        }
    }

    fn memory_usage(&self) -> usize {
        let mut total = 0usize;
        for node in &self.nodes {
            match node {
                TreeNode::Leaf { points } => {
                    total = total.saturating_add(points.len() * std::mem::size_of::<usize>());
                }
                TreeNode::Split { normal, .. } => {
                    total = total.saturating_add(normal.len() * std::mem::size_of::<f32>());
                    total = total.saturating_add(2 * std::mem::size_of::<usize>());
                    total = total.saturating_add(std::mem::size_of::<f32>());
                }
            }
        }
        total
    }
}

/// Tree node used internally by the Annoy tree structure.
enum TreeNode {
    Leaf {
        points: Vec<usize>,
    },
    Split {
        normal: Vec<f32>,
        threshold: f32,
        left: usize,
        right: usize,
    },
}

/// Builder responsible for constructing a single Annoy-style tree.
struct AnnoyTreeBuilder<'a> {
    vectors: &'a [[f32; 768]],
    dimension: usize,
    leaf_size: usize,
    max_depth: usize,
    rng: StdRng,
    nodes: Vec<TreeNode>,
}

impl<'a> AnnoyTreeBuilder<'a> {
    fn new(
        vectors: &'a [[f32; 768]],
        dimension: usize,
        leaf_size: usize,
        max_depth: usize,
        seed: u64,
    ) -> Self {
        Self {
            vectors,
            dimension,
            leaf_size,
            max_depth,
            rng: StdRng::seed_from_u64(seed),
            nodes: Vec::new(),
        }
    }

    fn build(mut self, indices: &[usize]) -> Result<AnnoyTree> {
        let root = self.build_node(indices.to_vec(), 0)?;
        Ok(AnnoyTree {
            nodes: self.nodes,
            root,
        })
    }

    fn build_node(&mut self, indices: Vec<usize>, depth: usize) -> Result<usize> {
        if indices.len() <= self.leaf_size || depth >= self.max_depth || indices.len() <= 1 {
            let node_index = self.nodes.len();
            self.nodes.push(TreeNode::Leaf { points: indices });
            return Ok(node_index);
        }

        let (mut normal, threshold) = self.hyperplane(&indices)?;

        let mut left = Vec::new();
        let mut right = Vec::new();
        left.reserve(indices.len());
        right.reserve(indices.len());

        for &idx in &indices {
            let point = &self.vectors[idx];
            let projection = dot_product(&normal, point);
            if projection <= threshold {
                left.push(idx);
            } else {
                right.push(idx);
            }
        }

        // Fallback: if split is degenerate, use median split on random dimension.
        if left.is_empty() || right.is_empty() {
            let dimension_idx = self.rng.gen_range(0..self.dimension);
            let mut sorted = indices;
            sorted.sort_by(|a, b| {
                self.vectors[*a][dimension_idx]
                    .partial_cmp(&self.vectors[*b][dimension_idx])
                    .unwrap_or(Ordering::Equal)
            });

            let mid = sorted.len() / 2;
            if mid == 0 || mid == sorted.len() {
                let node_index = self.nodes.len();
                self.nodes.push(TreeNode::Leaf { points: sorted });
                return Ok(node_index);
            }

            let right_vec = sorted.split_off(mid);
            let left_vec = sorted;
            let threshold_val = self.vectors[right_vec[0]][dimension_idx];

            normal = vec![0.0f32; self.dimension];
            if let Some(component) = normal.get_mut(dimension_idx) {
                *component = 1.0;
            }

            left = left_vec;
            right = right_vec;

            // Replace threshold with axis-aligned split.
            return self.finish_split(normal, threshold_val, left, right, depth + 1);
        }

        self.finish_split(normal, threshold, left, right, depth + 1)
    }

    fn finish_split(
        &mut self,
        normal: Vec<f32>,
        threshold: f32,
        left: Vec<usize>,
        right: Vec<usize>,
        depth: usize,
    ) -> Result<usize> {
        let left_index = self.build_node(left, depth)?;
        let right_index = self.build_node(right, depth)?;

        let node_index = self.nodes.len();
        self.nodes.push(TreeNode::Split {
            normal,
            threshold,
            left: left_index,
            right: right_index,
        });
        Ok(node_index)
    }

    fn hyperplane(&mut self, indices: &[usize]) -> Result<(Vec<f32>, f32)> {
        ensure!(
            indices.len() >= 2,
            "Need at least two points to construct a hyperplane"
        );

        let idx_a = self.sample_index(indices);
        let mut idx_b = self.sample_index(indices);
        if idx_a == idx_b {
            idx_b = indices[(idx_b + 1) % indices.len()];
        }

        let vector_a = &self.vectors[idx_a];
        let vector_b = &self.vectors[idx_b];

        let mut normal: Vec<f32> = vector_a
            .iter()
            .zip(vector_b.iter())
            .map(|(a, b)| a - b)
            .collect();

        if !normalise(&mut normal) {
            normal = self.random_unit_vector();
        }

        let threshold = 0.5 * (dot_product(&normal, vector_a) + dot_product(&normal, vector_b));
        Ok((normal, threshold))
    }

    fn sample_index(&mut self, indices: &[usize]) -> usize {
        let range = 0..indices.len();
        let choice = self.rng.gen_range(range);
        indices[choice]
    }

    fn random_unit_vector(&mut self) -> Vec<f32> {
        let mut direction = vec![0.0f32; self.dimension];
        for component in &mut direction {
            *component = self.rng.gen_range(-1.0f32..1.0f32);
        }

        if normalise(&mut direction) {
            direction
        } else {
            let mut fallback = vec![0.0f32; self.dimension];
            if let Some(first) = fallback.first_mut() {
                *first = 1.0;
            }
            fallback
        }
    }
}

/// Normalise a mutable vector in-place, returning whether normalisation succeeded.
fn normalise(vector: &mut [f32]) -> bool {
    let mut norm_sq = 0.0f32;
    for component in vector.iter() {
        norm_sq += component * component;
    }

    if norm_sq <= MIN_NORMAL_NORM {
        return false;
    }

    let norm = norm_sq.sqrt();
    for component in vector.iter_mut() {
        *component /= norm;
    }
    true
}

/// Dot product between a vector slice and a fixed-size query vector.
fn dot_product(weights: &[f32], vector: &[f32; 768]) -> f32 {
    weights.iter().zip(vector.iter()).map(|(w, v)| w * v).sum()
}

/// Node used when traversing the tree during search.
struct SearchNode {
    priority: f32,
    node_index: usize,
}

impl PartialEq for SearchNode {
    fn eq(&self, other: &Self) -> bool {
        self.node_index == other.node_index && self.priority == other.priority
    }
}

impl Eq for SearchNode {}

impl PartialOrd for SearchNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering to make BinaryHeap behave like a min-heap based on priority.
        other
            .priority
            .total_cmp(&self.priority)
            .then_with(|| self.node_index.cmp(&other.node_index))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_and_search_returns_reasonable_results() {
        let mut index = AnnoyAnnIndex::new(768, 5).expect("failed to create Annoy index");

        let vectors: Vec<[f32; 768]> = (0..200)
            .map(|i| {
                let mut v = [0.0f32; 768];
                v[0] = i as f32;
                v
            })
            .collect();

        index.build(&vectors).expect("build should succeed");

        let mut query = [0.0f32; 768];
        query[0] = 42.0;

        let results = index.search(&query, 5);
        assert!(!results.is_empty());
        assert!(results.iter().any(|(idx, _)| *idx == 42));
    }

    #[test]
    fn memory_usage_is_non_zero_after_build() {
        let mut index =
            AnnoyAnnIndex::with_params(768, 3, 32, 100).expect("failed to create Annoy index");

        let vectors: Vec<[f32; 768]> = vec![[0.5f32; 768]; 50];
        index.build(&vectors).expect("build should succeed");

        assert!(index.memory_usage() >= vectors.len() * 768 * std::mem::size_of::<f32>());
    }

    #[test]
    fn with_params_respects_custom_configuration() {
        let index =
            AnnoyAnnIndex::with_params(768, 7, 16, 200).expect("failed to create Annoy index");
        assert_eq!(index.dimension, 768);
        assert_eq!(index.n_trees, 7);
        assert_eq!(index.leaf_size, 16);
        assert_eq!(index.search_k, 200);
    }
}
