# Task 008: Hierarchical Spreading

## Objective
Implement hierarchical spreading patterns that flow both upward (episode to concept) and downward (concept to episode) with appropriate activation strengths.

## Background
Hierarchical spreading enables generalization (upward) and instantiation (downward) in memory networks, modeling how specific memories activate general concepts and vice versa.

## Requirements
1. Design asymmetric spreading strengths for upward/downward
2. Implement multi-hop spreading through concept hierarchies
3. Add spreading decay over hierarchical distance
4. Support spreading path tracking for explainability
5. Optimize for deep hierarchies

## Technical Specification

### Files to Create
- `engram-core/src/activation/hierarchical.rs` - Hierarchical spreading

### Hierarchical Spreading Design
```rust
pub struct HierarchicalSpreading {
    upward_strength: f32,   // Episode -> Concept (typically 0.8)
    downward_strength: f32, // Concept -> Episode (typically 0.6)
    lateral_strength: f32,  // Concept -> Concept (typically 0.4)
    max_depth: usize,
}

impl HierarchicalSpreading {
    pub fn spread_hierarchical(
        &self,
        seeds: Vec<(NodeId, f32)>,
    ) -> HierarchicalSpreadResult {
        let mut frontier = BinaryHeap::new();
        let mut visited = HashSet::new();
        let mut activations = HashMap::new();
        let mut paths = HashMap::new();
        
        // Initialize frontier
        for (node_id, activation) in seeds {
            frontier.push(SpreadNode {
                id: node_id.clone(),
                activation,
                depth: 0,
                path: vec![node_id.clone()],
            });
        }
        
        while let Some(current) = frontier.pop() {
            if current.depth >= self.max_depth || visited.contains(&current.id) {
                continue;
            }
            
            visited.insert(current.id.clone());
            activations.insert(current.id.clone(), current.activation);
            paths.insert(current.id.clone(), current.path.clone());
            
            // Spread based on node type
            let next_nodes = match self.graph.get_node_type(&current.id) {
                MemoryNodeType::Episode { .. } => {
                    self.spread_upward(&current)
                }
                MemoryNodeType::Concept { .. } => {
                    self.spread_downward_and_lateral(&current)
                }
            };
            
            frontier.extend(next_nodes);
        }
        
        HierarchicalSpreadResult {
            activations,
            paths,
            max_depth_reached: visited.iter()
                .map(|id| paths[id].len())
                .max()
                .unwrap_or(0),
        }
    }
}
```

### Spreading Strength Calculation
```rust
impl HierarchicalSpreading {
    fn calculate_spread_strength(
        &self,
        source_type: &MemoryNodeType,
        target_type: &MemoryNodeType,
        binding_strength: f32,
        depth: usize,
    ) -> f32 {
        let base_strength = match (source_type, target_type) {
            (MemoryNodeType::Episode { .. }, MemoryNodeType::Concept { .. }) => {
                self.upward_strength
            }
            (MemoryNodeType::Concept { .. }, MemoryNodeType::Episode { .. }) => {
                self.downward_strength
            }
            (MemoryNodeType::Concept { .. }, MemoryNodeType::Concept { .. }) => {
                self.lateral_strength
            }
            _ => 0.0,
        };
        
        // Apply depth decay
        let depth_decay = 0.8_f32.powi(depth as i32);
        base_strength * binding_strength * depth_decay
    }
}
```

## Implementation Notes
- Use priority queue for breadth-first spreading
- Track paths for explainability
- Consider using SIMD for batch distance calculations
- Implement spreading cutoff for efficiency

## Testing Approach
1. Unit tests for directional spreading
2. Multi-level hierarchy tests
3. Path tracking validation
4. Performance with deep hierarchies

## Acceptance Criteria
- [ ] Upward spreading stronger than downward
- [ ] Multi-hop spreading works correctly
- [ ] Path tracking provides full provenance
- [ ] Performance scales sub-linearly with depth
- [ ] Configurable strength parameters

## Dependencies
- Task 007 (Fan Effect Spreading)
- Task 005 (Binding Formation)

## Estimated Time
3 days