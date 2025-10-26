//! Runnable examples demonstrating Engram's query language
//!
//! This file contains comprehensive examples of all query operations with
//! explanations and expected outcomes. These examples are designed to compile
//! and run, serving as both documentation and validation.
//!
//! Run with: `cargo run --example query_examples`

use engram_core::Confidence;
use engram_core::query::parser::Parser;
use engram_core::query::parser::ast::{
    ConfidenceThreshold, ConsolidateQuery, Constraint, EpisodeSelector, NodeIdentifier, Pattern,
    Query, RecallQueryBuilder, SpreadQuery,
};
use std::borrow::Cow;
use std::time::Duration;

fn main() {
    println!("Engram Query Language Examples\n");
    println!("================================\n");

    // Run all examples
    recall_examples();
    spread_examples();
    predict_examples();
    imagine_examples();
    consolidate_examples();
    builder_pattern_examples();
    error_examples();

    println!("\nAll examples completed successfully!");
}

/// RECALL query examples - retrieving memories
fn recall_examples() {
    println!("RECALL QUERIES");
    println!("--------------\n");

    // Example 1: Simple node ID recall
    {
        println!("Example 1: Recall by node ID");
        let query = "RECALL episode_123";
        let ast = Parser::parse(query).expect("valid query");

        match ast {
            Query::Recall(recall) => {
                println!("  Query: {}", query);
                println!("  Pattern: NodeId");
                if let Pattern::NodeId(id) = &recall.pattern {
                    println!("  Node ID: {}", id.as_str());
                }
                println!("  Constraints: {}", recall.constraints.len());
                println!();
            }
            _ => panic!("Expected Recall query"),
        }
    }

    // Example 2: Embedding similarity search
    {
        println!("Example 2: Recall by embedding similarity");
        // Generate 768-dimensional embedding for demonstration
        let embedding_vec: Vec<String> = (0..768)
            .map(|i| format!("{:.3}", (i as f32 / 768.0)))
            .collect();
        let embedding_str = embedding_vec.join(", ");
        let query = format!("RECALL [{}] THRESHOLD 0.85", embedding_str);
        let ast = Parser::parse(&query).expect("valid query");

        match ast {
            Query::Recall(recall) => {
                println!("  Query: RECALL [<768-dimensional vector>] THRESHOLD 0.85");
                if let Pattern::Embedding { vector, threshold } = &recall.pattern {
                    println!("  Vector dimensions: {}", vector.len());
                    println!("  Similarity threshold: {}", threshold);
                }
                println!();
            }
            _ => panic!("Expected Recall query"),
        }
    }

    // Example 3: Content match
    {
        println!("Example 3: Recall by content match");
        let query = r#"RECALL "neural networks and deep learning""#;
        let ast = Parser::parse(query).expect("valid query");

        match ast {
            Query::Recall(recall) => {
                println!("  Query: {}", query);
                if let Pattern::ContentMatch(content) = &recall.pattern {
                    println!("  Search text: {}", content);
                }
                println!();
            }
            _ => panic!("Expected Recall query"),
        }
    }

    // Example 4: With confidence threshold
    {
        println!("Example 4: Recall with confidence filter");
        let query = "RECALL episode CONFIDENCE > 0.7";
        let ast = Parser::parse(query).expect("valid query");

        match ast {
            Query::Recall(recall) => {
                println!("  Query: {}", query);
                println!(
                    "  Has confidence threshold: {}",
                    recall.confidence_threshold.is_some()
                );
                if let Some(ConfidenceThreshold::Above(conf)) = recall.confidence_threshold {
                    println!("  Minimum confidence: {}", conf.raw());
                }
                println!();
            }
            _ => panic!("Expected Recall query"),
        }
    }

    // Example 5: Complex query with multiple constraints
    {
        println!("Example 5: Recall with multiple constraints");
        let query = r#"RECALL episode WHERE confidence > 0.8 content CONTAINS "machine learning""#;
        let ast = Parser::parse(query).expect("valid query");

        match ast {
            Query::Recall(recall) => {
                println!("  Query: {}", query);
                println!("  Number of constraints: {}", recall.constraints.len());
                for (i, constraint) in recall.constraints.iter().enumerate() {
                    println!("  Constraint {}: {:?}", i + 1, constraint);
                }
                println!();
            }
            _ => panic!("Expected Recall query"),
        }
    }

    // Example 6: With base rate prior
    {
        println!("Example 6: Recall with Bayesian prior");
        let query = "RECALL episode CONFIDENCE > 0.6 BASE_RATE 0.3";
        let ast = Parser::parse(query).expect("valid query");

        match ast {
            Query::Recall(recall) => {
                println!("  Query: {}", query);
                println!("  Has base rate: {}", recall.base_rate.is_some());
                if let Some(base_rate) = recall.base_rate {
                    println!("  Prior probability: {}", base_rate.raw());
                }
                println!();
            }
            _ => panic!("Expected Recall query"),
        }
    }
}

/// SPREAD query examples - activation spreading
fn spread_examples() {
    println!("SPREAD QUERIES");
    println!("--------------\n");

    // Example 1: Basic spreading
    {
        println!("Example 1: Basic activation spreading");
        let query = "SPREAD FROM episode_123";
        let ast = Parser::parse(query).expect("valid query");

        match ast {
            Query::Spread(spread) => {
                println!("  Query: {}", query);
                println!("  Source node: {}", spread.source.as_str());
                println!(
                    "  Max hops: {:?} (default: {})",
                    spread.max_hops,
                    SpreadQuery::DEFAULT_MAX_HOPS
                );
                println!(
                    "  Decay rate: {:?} (default: {})",
                    spread.decay_rate,
                    SpreadQuery::DEFAULT_DECAY_RATE
                );
                println!();
            }
            _ => panic!("Expected Spread query"),
        }
    }

    // Example 2: Controlled spreading
    {
        println!("Example 2: Controlled activation spreading");
        let query = "SPREAD FROM concept_ai MAX_HOPS 5 DECAY 0.15 THRESHOLD 0.05";
        let ast = Parser::parse(query).expect("valid query");

        match ast {
            Query::Spread(spread) => {
                println!("  Query: {}", query);
                println!("  Source node: {}", spread.source.as_str());
                println!("  Max hops: {:?}", spread.max_hops);
                println!("  Decay rate: {:?}", spread.decay_rate);
                println!("  Activation threshold: {:?}", spread.activation_threshold);
                println!("  Estimated cost: {} units", spread.estimated_cost());
                println!();
            }
            _ => panic!("Expected Spread query"),
        }
    }

    // Example 3: Wide exploration
    {
        println!("Example 3: Wide exploratory spreading");
        let query = "SPREAD FROM node_start MAX_HOPS 10 DECAY 0.05 THRESHOLD 0.001";
        let ast = Parser::parse(query).expect("valid query");

        match ast {
            Query::Spread(spread) => {
                println!("  Query: {}", query);
                println!("  Source node: {}", spread.source.as_str());
                println!("  Strategy: Wide exploration (low decay, low threshold, many hops)");
                println!("  Warning: High computational cost for deep spreading");
                println!("  Estimated cost: {} units", spread.estimated_cost());
                println!();
            }
            _ => panic!("Expected Spread query"),
        }
    }

    // Example 4: Focused activation
    {
        println!("Example 4: Focused activation spreading");
        let query = "SPREAD FROM memory_core MAX_HOPS 2 DECAY 0.3 THRESHOLD 0.1";
        let ast = Parser::parse(query).expect("valid query");

        match ast {
            Query::Spread(spread) => {
                println!("  Query: {}", query);
                println!("  Source node: {}", spread.source.as_str());
                println!("  Strategy: Focused activation (high decay, high threshold, few hops)");
                println!("  Use case: Finding only strongly related concepts");
                println!();
            }
            _ => panic!("Expected Spread query"),
        }
    }
}

/// PREDICT query examples - future state prediction
fn predict_examples() {
    println!("PREDICT QUERIES");
    println!("---------------\n");

    // Example 1: Basic prediction
    {
        println!("Example 1: Basic prediction from context");
        let query = "PREDICT next_state GIVEN current_episode";
        let ast = Parser::parse(query).expect("valid query");

        match ast {
            Query::Predict(predict) => {
                println!("  Query: {}", query);
                println!("  Context nodes: {}", predict.context.len());
                println!("  Time horizon: {:?}", predict.horizon);
                println!();
            }
            _ => panic!("Expected Predict query"),
        }
    }

    // Example 2: Multiple context nodes
    {
        println!("Example 2: Prediction with multiple context nodes");
        let query = "PREDICT outcome GIVEN context1, context2, context3";
        let ast = Parser::parse(query).expect("valid query");

        match ast {
            Query::Predict(predict) => {
                println!("  Query: {}", query);
                println!("  Context nodes: {}", predict.context.len());
                for (i, node) in predict.context.iter().enumerate() {
                    println!("    Context {}: {}", i + 1, node.as_str());
                }
                println!("  Estimated cost: {} units", predict.estimated_cost());
                println!();
            }
            _ => panic!("Expected Predict query"),
        }
    }

    // Example 3: With time horizon
    {
        println!("Example 3: Prediction with time horizon");
        let query = "PREDICT future_state GIVEN current HORIZON 3600";
        let ast = Parser::parse(query).expect("valid query");

        match ast {
            Query::Predict(predict) => {
                println!("  Query: {}", query);
                if let Some(horizon) = predict.horizon {
                    println!(
                        "  Time horizon: {} seconds ({} minutes)",
                        horizon.as_secs(),
                        horizon.as_secs() / 60
                    );
                }
                println!("  Use case: Predicting state 1 hour in the future");
                println!();
            }
            _ => panic!("Expected Predict query"),
        }
    }
}

/// IMAGINE query examples - creative pattern completion
fn imagine_examples() {
    println!("IMAGINE QUERIES");
    println!("---------------\n");

    // Example 1: Basic imagination
    {
        println!("Example 1: Basic pattern completion");
        let query = "IMAGINE new_concept";
        let ast = Parser::parse(query).expect("valid query");

        match ast {
            Query::Imagine(imagine) => {
                println!("  Query: {}", query);
                println!("  Seeds: {}", imagine.seeds.len());
                println!("  Novelty: {:?} (default: moderate)", imagine.novelty);
                println!();
            }
            _ => panic!("Expected Imagine query"),
        }
    }

    // Example 2: Seeded generation
    {
        println!("Example 2: Seeded creative generation");
        let query = "IMAGINE hybrid BASED ON concept1, concept2";
        let ast = Parser::parse(query).expect("valid query");

        match ast {
            Query::Imagine(imagine) => {
                println!("  Query: {}", query);
                println!("  Seed nodes: {}", imagine.seeds.len());
                for (i, seed) in imagine.seeds.iter().enumerate() {
                    println!("    Seed {}: {}", i + 1, seed.as_str());
                }
                println!("  Use case: Blending two concepts");
                println!();
            }
            _ => panic!("Expected Imagine query"),
        }
    }

    // Example 3: Controlled creativity
    {
        println!("Example 3: Conservative pattern completion");
        let query = "IMAGINE novel_idea BASED ON seed1, seed2 NOVELTY 0.3";
        let ast = Parser::parse(query).expect("valid query");

        match ast {
            Query::Imagine(imagine) => {
                println!("  Query: {}", query);
                if let Some(novelty) = imagine.novelty {
                    println!("  Novelty level: {} (conservative)", novelty);
                }
                println!("  Strategy: Stay close to known patterns");
                println!();
            }
            _ => panic!("Expected Imagine query"),
        }
    }

    // Example 4: Highly creative
    {
        println!("Example 4: Highly creative generation");
        let query = "IMAGINE wild_idea NOVELTY 0.9 CONFIDENCE > 0.4";
        let ast = Parser::parse(query).expect("valid query");

        match ast {
            Query::Imagine(imagine) => {
                println!("  Query: {}", query);
                if let Some(novelty) = imagine.novelty {
                    println!("  Novelty level: {} (highly creative)", novelty);
                }
                println!("  Strategy: Explore distant associations");
                println!("  Estimated cost: {} units", imagine.estimated_cost());
                println!();
            }
            _ => panic!("Expected Imagine query"),
        }
    }
}

/// CONSOLIDATE query examples - memory consolidation
fn consolidate_examples() {
    println!("CONSOLIDATE QUERIES");
    println!("-------------------\n");

    // Example 1: Basic consolidation
    {
        println!("Example 1: Simple memory consolidation");
        let query = "CONSOLIDATE episode_123 INTO concept_ml";
        let ast = Parser::parse(query).expect("valid query");

        match ast {
            Query::Consolidate(consolidate) => {
                println!("  Query: {}", query);
                println!("  Target node: {}", consolidate.target.as_str());
                println!("  Episode selector: {:?}", consolidate.episodes);
                println!();
            }
            _ => panic!("Expected Consolidate query"),
        }
    }

    // Example 2: Batch consolidation
    {
        println!("Example 2: Batch consolidation with constraints");
        let query = "CONSOLIDATE WHERE confidence > 0.7 INTO summary_node";
        let ast = Parser::parse(query).expect("valid query");

        match ast {
            Query::Consolidate(consolidate) => {
                println!("  Query: {}", query);
                println!("  Target node: {}", consolidate.target.as_str());
                if let EpisodeSelector::Where(constraints) = &consolidate.episodes {
                    println!("  Constraints: {}", constraints.len());
                }
                println!("  Use case: Consolidating high-confidence memories");
                println!();
            }
            _ => panic!("Expected Consolidate query"),
        }
    }

    // Example 3: Pattern-based consolidation
    {
        println!("Example 3: Pattern-based consolidation");
        let query = r#"CONSOLIDATE "learning experience" INTO expertise_node"#;
        let ast = Parser::parse(query).expect("valid query");

        match ast {
            Query::Consolidate(consolidate) => {
                println!("  Query: {}", query);
                println!("  Target node: {}", consolidate.target.as_str());
                println!("  Strategy: Consolidating all episodes matching pattern");
                println!(
                    "  Estimated cost: {} units",
                    ConsolidateQuery::estimated_cost()
                );
                println!();
            }
            _ => panic!("Expected Consolidate query"),
        }
    }
}

/// Builder pattern examples - programmatic query construction
fn builder_pattern_examples() {
    println!("BUILDER PATTERN");
    println!("---------------\n");

    // Example 1: Type-safe query building
    {
        println!("Example 1: Type-safe RecallQuery builder");

        let query = RecallQueryBuilder::new()
            .pattern(Pattern::NodeId(NodeIdentifier::from("episode_123")))
            .constraint(Constraint::ConfidenceAbove(Confidence::from_raw(0.7)))
            .confidence_threshold(ConfidenceThreshold::Above(Confidence::from_raw(0.8)))
            .limit(10)
            .build()
            .expect("valid query");

        println!("  Built query with:");
        println!("    - Pattern: NodeId");
        println!("    - Constraints: {}", query.constraints.len());
        println!("    - Confidence threshold: present");
        println!("    - Limit: {:?}", query.limit);
        println!();

        // Validate the query
        query.validate().expect("query should be valid");
        println!("  Validation: PASSED");
        println!();
    }

    // Example 2: Building embedding query
    {
        println!("Example 2: Building embedding similarity query");

        let embedding = vec![0.1; 768]; // 768-dimensional vector

        let query = RecallQueryBuilder::new()
            .pattern(Pattern::Embedding {
                vector: embedding,
                threshold: 0.85,
            })
            .constraint(Constraint::ContentContains(Cow::Borrowed(
                "machine learning",
            )))
            .build()
            .expect("valid query");

        println!("  Built query with:");
        if let Pattern::Embedding { vector, threshold } = &query.pattern {
            println!("    - Embedding dimensions: {}", vector.len());
            println!("    - Similarity threshold: {}", threshold);
        }
        println!("    - Content constraint: present");
        println!();

        // Validate the query
        query.validate().expect("query should be valid");
        println!("  Validation: PASSED");
        println!();
    }

    // Example 3: Manually constructing SpreadQuery
    {
        println!("Example 3: Manual SpreadQuery construction");

        let query = SpreadQuery {
            source: NodeIdentifier::from("node_123"),
            max_hops: Some(5),
            decay_rate: Some(0.15),
            activation_threshold: Some(0.05),
            refractory_period: Some(Duration::from_millis(100)),
        };

        println!("  Constructed query:");
        println!("    - Source: {}", query.source.as_str());
        println!("    - Max hops: {:?}", query.max_hops);
        println!("    - Decay rate: {:?}", query.decay_rate);
        println!("    - Threshold: {:?}", query.activation_threshold);
        println!("    - Refractory period: {:?}", query.refractory_period);
        println!();

        // Validate the query
        query.validate().expect("query should be valid");
        println!("  Validation: PASSED");
        println!();
    }
}

/// Error examples - demonstrating helpful error messages
fn error_examples() {
    println!("ERROR HANDLING");
    println!("--------------\n");

    // Example 1: Typo detection
    {
        println!("Example 1: Typo detection");
        let query = "RECAL episode"; // Typo: RECAL instead of RECALL
        let result = Parser::parse(query);

        match result {
            Err(err) => {
                println!("  Query: {}", query);
                println!("  Error detected: Typo in keyword");
                println!("  Error message:\n{}", err);
            }
            Ok(_) => panic!("Expected error for typo"),
        }
    }

    // Example 2: Missing required token
    {
        println!("Example 2: Missing required pattern");
        let query = "RECALL"; // Missing pattern
        let result = Parser::parse(query);

        match result {
            Err(err) => {
                println!("  Query: {}", query);
                println!("  Error detected: Missing pattern");
                println!("  Error message:\n{}", err);
            }
            Ok(_) => panic!("Expected error for missing pattern"),
        }
    }

    // Example 3: Invalid constraint
    {
        println!("Example 3: Empty embedding vector");
        let query = "RECALL []"; // Empty embedding
        let result = Parser::parse(query);

        match result {
            Err(err) => {
                println!("  Query: {}", query);
                println!("  Error detected: Empty embedding");
                println!("  Error message:\n{}", err);
            }
            Ok(_) => panic!("Expected error for empty embedding"),
        }
    }

    // Example 4: Validation error
    {
        println!("Example 4: Validation error - invalid embedding dimension");

        let query_result = RecallQueryBuilder::new()
            .pattern(Pattern::Embedding {
                vector: vec![0.1, 0.2, 0.3], // Wrong dimensions (should be 768)
                threshold: 0.8,
            })
            .build();

        match query_result {
            Err(err) => {
                println!("  Error detected during build: Invalid embedding dimensions");
                println!("  Error message: {}", err);
            }
            Ok(query) => {
                // If build succeeds, validate should catch it
                let result = query.validate();
                match result {
                    Err(err) => {
                        println!(
                            "  Error detected during validation: Invalid embedding dimensions"
                        );
                        println!("  Error message: {}", err);
                    }
                    Ok(_) => panic!("Expected validation error"),
                }
            }
        }
    }

    println!();
}
