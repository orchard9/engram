//! Entorhinal cortex-inspired context gathering with grid cell dynamics.

use crate::{Episode, memory::Memory};
use std::collections::HashMap;

/// Grid module for spatial/temporal indexing
#[derive(Debug, Clone)]
pub struct GridModule {
    /// Grid spacing (scale of the grid)
    pub scale: f32,
    
    /// Grid orientation in radians
    pub orientation: f32,
    
    /// Grid phase offset (x, y)
    pub phase: (f32, f32),
    
    /// Individual field width
    pub field_width: f32,
}

impl GridModule {
    /// Create a new grid module
    pub fn new(scale: f32, orientation: f32, phase: (f32, f32)) -> Self {
        Self {
            scale,
            orientation,
            phase,
            field_width: scale * 0.3, // Field width is typically 30% of grid spacing
        }
    }
    
    /// Calculate grid cell activation at a position
    pub fn activation(&self, position: (f32, f32)) -> f32 {
        // Hexagonal grid pattern
        let rotated_x = position.0 * self.orientation.cos() - position.1 * self.orientation.sin();
        let rotated_y = position.0 * self.orientation.sin() + position.1 * self.orientation.cos();
        
        let shifted_x = rotated_x - self.phase.0;
        let shifted_y = rotated_y - self.phase.1;
        
        // Generate hexagonal pattern using three cosine waves
        let angle1: f32 = 0.0;
        let angle2: f32 = 2.0 * std::f32::consts::PI / 3.0;
        let angle3: f32 = 4.0 * std::f32::consts::PI / 3.0;
        
        let wave1 = ((2.0 * std::f32::consts::PI * 
            (shifted_x * angle1.cos() + shifted_y * angle1.sin())) / self.scale).cos();
        let wave2 = ((2.0 * std::f32::consts::PI * 
            (shifted_x * angle2.cos() + shifted_y * angle2.sin())) / self.scale).cos();
        let wave3 = ((2.0 * std::f32::consts::PI * 
            (shifted_x * angle3.cos() + shifted_y * angle3.sin())) / self.scale).cos();
        
        // Combine waves and normalize
        let activation = (wave1 + wave2 + wave3) / 3.0;
        (activation + 1.0) / 2.0 // Normalize to [0, 1]
    }
}

/// Entorhinal cortex-inspired context gathering
pub struct EntorhinalContext {
    /// Multiple grid modules at different scales
    grid_modules: Vec<GridModule>,
    
    /// Temporal context window (in seconds)
    temporal_window: f64,
    
    /// Spatial context radius
    spatial_radius: f32,
    
    /// Episode context cache
    context_cache: HashMap<String, Vec<Episode>>,
}

impl EntorhinalContext {
    /// Create a new entorhinal context system
    pub fn new() -> Self {
        // Create grid modules at multiple scales (like in real entorhinal cortex)
        let scales = vec![30.0, 42.0, 59.0, 83.0, 117.0]; // Geometric progression
        let mut grid_modules = Vec::new();
        
        for (i, scale) in scales.into_iter().enumerate() {
            let orientation = (i as f32) * std::f32::consts::PI / 6.0; // Different orientations
            let phase = ((i as f32) * 7.3, (i as f32) * 11.7); // Pseudo-random phases
            grid_modules.push(GridModule::new(scale, orientation, phase));
        }
        
        Self {
            grid_modules,
            temporal_window: 3600.0, // 1 hour window
            spatial_radius: 100.0,
            context_cache: HashMap::new(),
        }
    }
    
    /// Gather temporal context around an episode
    pub fn gather_temporal_context(&self, target_time: chrono::DateTime<chrono::Utc>, 
                                  episodes: &[Episode]) -> Vec<Episode> {
        let mut context_episodes = Vec::new();
        
        for episode in episodes {
            let time_diff = (target_time - episode.when).num_seconds().abs() as f64;
            if time_diff <= self.temporal_window {
                context_episodes.push(episode.clone());
            }
        }
        
        // Sort by temporal proximity
        context_episodes.sort_by_key(|ep| {
            (target_time - ep.when).num_seconds().abs()
        });
        
        context_episodes
    }
    
    /// Gather spatial context (if location information available)
    pub fn gather_spatial_context(&self, target_location: Option<&str>, 
                                 episodes: &[Episode]) -> Vec<Episode> {
        if let Some(target) = target_location {
            episodes.iter()
                .filter(|ep| {
                    if let Some(ref loc) = ep.where_location {
                        // Simple string similarity for now
                        loc.contains(target) || target.contains(loc)
                    } else {
                        false
                    }
                })
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }
    
    /// Gather semantic context based on content similarity
    pub fn gather_semantic_context(&self, target_content: &str, 
                                  memories: &[Memory]) -> Vec<Memory> {
        let mut context_memories = Vec::new();
        
        for memory in memories {
            if let Some(ref content) = memory.content {
                if content.contains(target_content) || target_content.contains(content.as_str()) {
                    context_memories.push(memory.clone());
                }
            }
        }
        
        // Sort by activation level
        context_memories.sort_by(|a, b| {
            b.activation().partial_cmp(&a.activation()).unwrap()
        });
        
        context_memories
    }
    
    /// Compute grid code for a spatiotemporal position
    pub fn compute_grid_code(&self, position: (f32, f32)) -> Vec<f32> {
        self.grid_modules.iter()
            .map(|module| module.activation(position))
            .collect()
    }
    
    /// Find episodes with similar grid codes
    pub fn find_similar_grid_patterns(&self, target_code: &[f32], 
                                     episodes: &[Episode]) -> Vec<(Episode, f32)> {
        let mut similar_episodes = Vec::new();
        
        for episode in episodes {
            // Compute similarity based on embedding (simplified)
            let similarity = self.grid_code_similarity(target_code, &episode.embedding[..target_code.len()]);
            if similarity > 0.5 {
                similar_episodes.push((episode.clone(), similarity));
            }
        }
        
        // Sort by similarity
        similar_episodes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similar_episodes
    }
    
    /// Compute similarity between grid codes
    fn grid_code_similarity(&self, code1: &[f32], code2: &[f32]) -> f32 {
        if code1.len() != code2.len() {
            return 0.0;
        }
        
        let dot: f32 = code1.iter().zip(code2.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm1: f32 = code1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = code2.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm1 > 0.0 && norm2 > 0.0 {
            dot / (norm1 * norm2)
        } else {
            0.0
        }
    }
    
    /// Integrate multiple context sources
    pub fn integrate_contexts(&self, temporal: Vec<Episode>, 
                            spatial: Vec<Episode>, 
                            semantic: Vec<Memory>) -> Vec<Episode> {
        let mut integrated = HashMap::new();
        let mut scores = HashMap::new();
        
        // Add temporal context with weight
        for episode in temporal {
            *scores.entry(episode.id.clone()).or_insert(0.0) += 1.0;
            integrated.insert(episode.id.clone(), episode);
        }
        
        // Add spatial context with weight
        for episode in spatial {
            *scores.entry(episode.id.clone()).or_insert(0.0) += 0.8;
            integrated.insert(episode.id.clone(), episode);
        }
        
        // Add semantic context (convert memories to episodes)
        for memory in semantic {
            *scores.entry(memory.id.clone()).or_insert(0.0) += 0.6;
            // Create episode from memory if not already present
            if !integrated.contains_key(&memory.id) {
                let episode = Episode::new(
                    memory.id.clone(),
                    memory.created_at,
                    memory.content.unwrap_or_else(|| "Memory".to_string()),
                    memory.embedding,
                    memory.confidence,
                );
                integrated.insert(memory.id.clone(), episode);
            }
        }
        
        // Sort by combined score
        let mut result: Vec<(String, Episode, f32)> = integrated.into_iter()
            .map(|(id, ep)| (id.clone(), ep, scores[&id]))
            .collect();
        result.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        
        result.into_iter().map(|(_, ep, _)| ep).collect()
    }
    
    /// Path integration for sequential memory navigation
    pub fn path_integrate(&self, start_position: (f32, f32), 
                         movements: &[(f32, f32)]) -> Vec<(f32, f32)> {
        let mut path = vec![start_position];
        let mut current = start_position;
        
        for movement in movements {
            current.0 += movement.0;
            current.1 += movement.1;
            path.push(current);
        }
        
        path
    }
    
    /// Border cell activation for boundary detection
    pub fn border_activation(&self, position: (f32, f32), boundary: f32) -> f32 {
        let distance_to_boundary = (position.0.abs().max(position.1.abs()) - boundary).abs();
        (-distance_to_boundary / 10.0).exp() // Exponential decay from boundary
    }
}

impl Default for EntorhinalContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_grid_module() {
        let module = GridModule::new(30.0, 0.0, (0.0, 0.0));
        
        // Test activation at origin
        let activation = module.activation((0.0, 0.0));
        assert!(activation >= 0.0 && activation <= 1.0);
        
        // Test activation at grid point
        let activation = module.activation((30.0, 0.0));
        assert!(activation >= 0.0 && activation <= 1.0);
    }
    
    #[test]
    fn test_entorhinal_context() {
        let context = EntorhinalContext::new();
        assert_eq!(context.grid_modules.len(), 5);
        
        // Test grid code computation
        let code = context.compute_grid_code((10.0, 20.0));
        assert_eq!(code.len(), 5);
        for value in code {
            assert!(value >= 0.0 && value <= 1.0);
        }
    }
    
    #[test]
    fn test_path_integration() {
        let context = EntorhinalContext::new();
        let movements = vec![(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)];
        let path = context.path_integrate((0.0, 0.0), &movements);
        
        assert_eq!(path.len(), 4);
        assert_eq!(path[0], (0.0, 0.0));
        assert_eq!(path[1], (1.0, 0.0));
        assert_eq!(path[2], (1.0, 1.0));
        assert_eq!(path[3], (0.0, 1.0));
    }
}