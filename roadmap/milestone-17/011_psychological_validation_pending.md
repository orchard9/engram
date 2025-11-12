# Task 011: Psychological Validation

## Objective
Validate dual memory implementation against established psychological research on semantic memory, fan effects, category formation, and other core memory phenomena with rigorous empirical methodology.

## Background
Engram's dual memory architecture must demonstrate alignment with cognitive science findings to ensure biological plausibility and predictable behavior. This task creates a comprehensive psychological validation framework that tests memory system behavior against published empirical results from classic experiments.

## Rationale
While individual cognitive mechanisms have internal validation (fan effect detector, priming engines, concept formation), we need integrated tests that validate emergent behavior against the full psychology literature. This ensures that component interactions produce results matching human memory phenomena, not just isolated mechanisms.

## Requirements
1. Implement test scenarios replicating key psychology experiments
2. Measure correlation with published empirical data (r > 0.8 target)
3. Validate category formation against prototype theory (Rosch 1975)
4. Test semantic priming with proper SOA timings (Neely 1977)
5. Validate additional phenomena: spacing effect, levels of processing, context-dependent memory
6. Use statistical analysis with significance testing and effect sizes
7. Document deviations with biological/computational justification
8. Create automated test suite for CI/CD integration

## Technical Specification

### Files to Create
- `engram-core/tests/psychological_validation.rs` - Main test suite
- `engram-core/tests/psychological/fan_effect_replication.rs` - Anderson (1974) replication
- `engram-core/tests/psychological/category_formation_tests.rs` - Rosch (1975) validation
- `engram-core/tests/psychological/semantic_priming_tests.rs` - Neely (1977) validation
- `engram-core/tests/psychological/spacing_effect_tests.rs` - Bjork & Bjork (1992)
- `engram-core/tests/psychological/levels_of_processing_tests.rs` - Craik & Lockhart (1972)
- `engram-core/tests/psychological/context_dependent_memory_tests.rs` - Godden & Baddeley (1975)
- `engram-core/tests/psychological/test_datasets.rs` - Shared test data
- `engram-core/tests/psychological/statistical_analysis.rs` - Correlation and significance testing

### 1. Fan Effect Validation (Anderson 1974)

#### Background
Task 007 already implements comprehensive fan effect validation. This section integrates that validation into the broader psychological framework.

#### Test Design
```rust
/// Replicates Anderson (1974) person-location learning paradigm
///
/// Participants learned sentences like "The doctor is in the bank"
/// Then verified facts, with RT increasing linearly with fan (associations)
///
/// Key Findings:
/// - Fan 1 (baseline): 1159ms ± 22ms
/// - Fan 2: 1236ms ± 25ms (+77ms)
/// - Fan 3: 1305ms ± 28ms (+69ms average)
/// - Linear slope: ~70ms per additional association
/// - Correlation with fan: r = 0.98 (Anderson 1974, Table 1)
#[test]
fn test_anderson_1974_person_location_paradigm() {
    let graph = UnifiedMemoryGraph::concurrent();
    let detector = FanEffectDetector::default();

    // Create person-location pairs matching Anderson's materials
    // Low fan: "The doctor is in the park" (doctor→1 location, park→1 person)
    // High fan: "The lawyer is in the church/park/store" (lawyer→3 locations)

    let test_cases = create_anderson_test_cases(&graph);

    // Measure retrieval activation for each fan level
    let mut results = Vec::new();
    for (person_id, expected_fan, expected_rt) in test_cases {
        let fan_effect = detector.detect_fan_effect(&person_id, &graph);

        results.push(FanEffectData {
            fan: fan_effect.fan,
            predicted_rt: fan_effect.retrieval_time_ms,
            empirical_rt: expected_rt,
            activation_divisor: fan_effect.activation_divisor,
        });
    }

    // Statistical validation
    let correlation = calculate_pearson_correlation(
        &results.iter().map(|r| r.fan as f32).collect::<Vec<_>>(),
        &results.iter().map(|r| r.predicted_rt).collect::<Vec<_>>()
    );

    assert!(
        correlation > 0.8,
        "Fan effect correlation with Anderson (1974) must exceed r > 0.8, got {}",
        correlation
    );

    // Validate slope (70ms per association)
    let slope = calculate_linear_slope(&results);
    assert!(
        (slope - 70.0).abs() < 10.0,
        "Fan effect slope should be ~70ms/association, got {:.1}ms",
        slope
    );
}

fn create_anderson_test_cases(graph: &UnifiedMemoryGraph) -> Vec<(Uuid, usize, f32)> {
    // Person-location pairs from Anderson (1974) Experiment 1
    // Returns: (person_node_id, expected_fan, empirical_rt_ms)

    let persons = vec![
        ("doctor", 1, 1159.0),    // Low fan
        ("lawyer", 3, 1305.0),    // High fan
        ("fireman", 2, 1236.0),   // Medium fan
    ];

    let locations = vec![
        ("bank", 1),
        ("church", 3),
        ("park", 3),
        ("store", 2),
    ];

    // Create nodes and edges matching fan structure
    let mut test_cases = Vec::new();
    for (person, person_fan, empirical_rt) in persons {
        let person_id = graph.store_memory(Memory::new(
            person.to_string(),
            create_embedding(person),
            Confidence::HIGH,
        )).unwrap();

        // Add associations to create target fan
        let person_locations = select_locations_for_fan(&locations, person_fan);
        for (location, _) in person_locations {
            let loc_id = graph.store_memory(Memory::new(
                location.to_string(),
                create_embedding(location),
                Confidence::HIGH,
            )).unwrap();

            graph.add_edge(person_id, loc_id, 1.0).unwrap();
        }

        test_cases.push((person_id, person_fan, empirical_rt));
    }

    test_cases
}
```

#### Validation Criteria
- Pearson correlation r > 0.8 between predicted and empirical RT
- Slope within ±10ms of 70ms/association
- All individual predictions within ±20ms of empirical data (within 1 SD)

### 2. Category Formation Validation (Rosch 1975)

#### Background
Eleanor Rosch demonstrated that categories have graded membership structure:
- Prototypical exemplars (robin, sparrow) judged more "bird-like" than atypical ones (penguin, ostrich)
- Verification RT faster for prototypes: "A robin is a bird" < "A penguin is a bird"
- Category formation extracts central tendency (prototype) from exemplars

#### Test Design
```rust
/// Validates prototype effects in concept formation
///
/// Rosch (1975) Key Findings:
/// - Prototypicality ratings: robin (6.9/7.0) vs penguin (3.9/7.0)
/// - Verification RT: prototype ~620ms vs atypical ~750ms (+130ms)
/// - Production frequency: robins named first in 90% of category generation tasks
///
/// References:
/// - Rosch (1975). "Cognitive representations of semantic categories."
///   JEP: General 104(3): 192-233.
/// - Rosch & Mervis (1975). "Family resemblances." Cognitive Psychology 7(4): 573-605.
#[test]
fn test_rosch_1975_prototype_effects() {
    let graph = UnifiedMemoryGraph::concurrent();
    let concept_engine = ConceptFormationEngine::default();

    // Create bird exemplars with varying typicality
    let typical_features = vec![
        "flies", "sings", "builds_nests", "small", "perches_in_trees"
    ];

    let atypical_features = vec![
        "swims", "large", "flightless", "lives_in_cold", "eats_fish"
    ];

    let bird_episodes = vec![
        // Prototypical birds (high family resemblance)
        create_episode("robin", &typical_features, 0.95),
        create_episode("sparrow", &typical_features, 0.93),
        create_episode("bluebird", &typical_features, 0.92),

        // Atypical birds (low family resemblance)
        create_episode("penguin", &atypical_features, 0.45),
        create_episode("ostrich", &atypical_features, 0.42),
        create_episode("emu", &atypical_features, 0.40),
    ];

    // Form concept through consolidation
    let bird_concept = concept_engine.form_concepts(&bird_episodes, SleepStage::NREM2)
        .into_iter()
        .next()
        .expect("Should form one bird concept");

    // Test 1: Prototypes closer to centroid (higher binding strength)
    let robin_distance = euclidean_distance(
        &bird_episodes[0].embedding,
        &bird_concept.centroid
    );
    let penguin_distance = euclidean_distance(
        &bird_episodes[3].embedding,
        &bird_concept.centroid
    );

    assert!(
        robin_distance < penguin_distance,
        "Prototype (robin) should be closer to concept centroid than atypical (penguin). \
         Robin distance: {:.3}, Penguin distance: {:.3}",
        robin_distance,
        penguin_distance
    );

    // Test 2: Graded membership structure
    let typicality_scores = bird_episodes.iter().map(|ep| {
        1.0 - euclidean_distance(&ep.embedding, &bird_concept.centroid)
    }).collect::<Vec<_>>();

    // Typical birds should have higher typicality scores
    let avg_typical = (typicality_scores[0] + typicality_scores[1] + typicality_scores[2]) / 3.0;
    let avg_atypical = (typicality_scores[3] + typicality_scores[4] + typicality_scores[5]) / 3.0;

    assert!(
        avg_typical > avg_atypical + 0.1,
        "Average typicality for prototypes ({:.3}) should exceed atypical ({:.3}) by >0.1",
        avg_typical,
        avg_atypical
    );

    // Test 3: Coherence score reflects within-category structure
    assert!(
        bird_concept.coherence_score > 0.65,
        "Bird concept should have coherence >0.65 per CA3 pattern completion threshold, got {:.3}",
        bird_concept.coherence_score
    );

    // Test 4: Semantic distance reflects abstraction level
    // More abstract concepts have higher semantic distance (variability among exemplars)
    assert!(
        bird_concept.semantic_distance > 0.2,
        "Semantic distance should reflect exemplar variability, got {:.3}",
        bird_concept.semantic_distance
    );
}

/// Tests typicality effect on spreading activation
///
/// Rosch (1978) showed prototype advantage in:
/// - Faster verification RT
/// - Stronger priming effects
/// - Earlier production in generation tasks
#[test]
fn test_typicality_effect_on_activation() {
    let graph = UnifiedMemoryGraph::concurrent();

    // Setup: Bird concept with prototype and atypical exemplars
    let bird_concept_id = create_bird_concept(&graph);
    let robin_id = create_and_bind_episode(&graph, "robin", bird_concept_id, 0.95);
    let penguin_id = create_and_bind_episode(&graph, "penguin", bird_concept_id, 0.45);

    // Spreading activation from concept should favor prototypes
    let spreading_engine = SpreadingActivationEngine::new();
    spreading_engine.activate_node(bird_concept_id, 1.0);

    // Allow spreading to complete
    thread::sleep(Duration::from_millis(50));

    let robin_activation = spreading_engine.get_activation(robin_id);
    let penguin_activation = spreading_engine.get_activation(penguin_id);

    assert!(
        robin_activation > penguin_activation * 1.5,
        "Prototype should receive stronger activation than atypical exemplar. \
         Robin: {:.3}, Penguin: {:.3}",
        robin_activation,
        penguin_activation
    );
}
```

#### Validation Criteria
- Prototypes closer to concept centroid (distance ratio > 1.2)
- Average typicality score difference > 0.1 between typical and atypical
- Coherence score > 0.65 (CA3 pattern completion threshold)
- Activation ratio prototype:atypical > 1.5

### 3. Semantic Priming Validation (Neely 1977)

#### Background
Neely (1977) established temporal dynamics of semantic priming:
- Automatic spreading activation: <400ms (unconscious)
- Strategic expectancy effects: >500ms (conscious)
- Optimal SOA (stimulus onset asynchrony): 250-400ms
- Related word pairs show 50-80ms RT reduction

#### Test Design
```rust
/// Validates semantic priming with proper temporal dynamics
///
/// Neely (1977) Experiment 1 findings:
/// - SOA 250ms: Related prime advantage = 65ms (automatic spreading)
/// - SOA 700ms: Related prime advantage = 80ms (automatic + strategic)
/// - Optimal automatic priming: 200-400ms SOA window
/// - Priming magnitude: ~10-15% RT reduction for related pairs
///
/// References:
/// - Neely (1977). "Semantic priming and retrieval from lexical memory."
///   JEP: General 106(3): 226-254. [SOA effects: Table 2, p. 238]
#[test]
fn test_neely_1977_semantic_priming_soa() {
    let priming_engine = SemanticPrimingEngine::new();

    // Test word pairs from Neely (1977) materials
    let related_pairs = vec![
        ("doctor", "nurse"),
        ("bread", "butter"),
        ("lion", "tiger"),
        ("chair", "table"),
    ];

    let unrelated_pairs = vec![
        ("doctor", "butter"),
        ("bread", "tiger"),
        ("lion", "table"),
        ("chair", "nurse"),
    ];

    // Test SOA timings
    let soa_timings = vec![
        100,  // Too fast for priming
        250,  // Optimal automatic priming
        400,  // Peak priming
        700,  // Strategic priming onset
    ];

    for soa_ms in soa_timings {
        let mut related_activations = Vec::new();
        let mut unrelated_activations = Vec::new();

        for (prime, target) in &related_pairs {
            // Activate prime
            priming_engine.activate_priming(
                prime,
                &create_embedding(prime),
                || vec![(target.to_string(), create_embedding(target), 1)]
            );

            // Wait SOA duration
            thread::sleep(Duration::from_millis(soa_ms));

            // Measure target activation
            let activation = priming_engine.compute_priming_boost(target);
            related_activations.push(activation);
        }

        for (prime, target) in &unrelated_pairs {
            priming_engine.activate_priming(
                prime,
                &create_embedding(prime),
                || vec![] // No semantic relation
            );

            thread::sleep(Duration::from_millis(soa_ms));

            let activation = priming_engine.compute_priming_boost(target);
            unrelated_activations.push(activation);
        }

        // Calculate facilitation effect
        let avg_related = mean(&related_activations);
        let avg_unrelated = mean(&unrelated_activations);
        let facilitation = avg_related - avg_unrelated;

        // Validate temporal dynamics
        match soa_ms {
            100 => {
                // Too fast for significant priming
                assert!(
                    facilitation < 0.05,
                    "SOA 100ms: Priming should be minimal, got {:.3}",
                    facilitation
                );
            },
            250 => {
                // Optimal automatic priming
                assert!(
                    facilitation > 0.10 && facilitation < 0.20,
                    "SOA 250ms: Priming should be 10-20% (automatic), got {:.3}",
                    facilitation
                );
            },
            400 => {
                // Peak priming window
                assert!(
                    facilitation > 0.12 && facilitation < 0.22,
                    "SOA 400ms: Priming should be 12-22% (peak automatic), got {:.3}",
                    facilitation
                );
            },
            700 => {
                // Should show decay unless strategic processes engaged
                // (Engram doesn't model strategic expectancy, so should decay)
                assert!(
                    facilitation > 0.05 && facilitation < 0.15,
                    "SOA 700ms: Priming should decay without strategic processing, got {:.3}",
                    facilitation
                );
            },
            _ => {},
        }
    }
}

/// Tests distance-dependent priming attenuation
///
/// Collins & Loftus (1975) spreading activation model predicts:
/// - Direct associates: Full priming
/// - 2-hop neighbors: Attenuated priming (~50%)
/// - 3+ hops: Minimal/no priming
#[test]
fn test_semantic_distance_attenuation() {
    let graph = UnifiedMemoryGraph::concurrent();
    let priming_engine = SemanticPrimingEngine::new();

    // Create semantic network: doctor → nurse → hospital → ambulance
    let doctor_id = graph.store_memory(Memory::new(
        "doctor".to_string(),
        create_embedding("doctor"),
        Confidence::HIGH,
    )).unwrap();

    let nurse_id = graph.store_memory(Memory::new(
        "nurse".to_string(),
        create_embedding("nurse"),
        Confidence::HIGH,
    )).unwrap();

    let hospital_id = graph.store_memory(Memory::new(
        "hospital".to_string(),
        create_embedding("hospital"),
        Confidence::HIGH,
    )).unwrap();

    let ambulance_id = graph.store_memory(Memory::new(
        "ambulance".to_string(),
        create_embedding("ambulance"),
        Confidence::HIGH,
    )).unwrap();

    // Create edges: doctor → nurse → hospital → ambulance
    graph.add_edge(doctor_id, nurse_id, 0.9).unwrap();
    graph.add_edge(nurse_id, hospital_id, 0.85).unwrap();
    graph.add_edge(hospital_id, ambulance_id, 0.8).unwrap();

    // Activate "doctor" and measure spread
    priming_engine.activate_priming(
        "doctor",
        &create_embedding("doctor"),
        || vec![
            ("nurse".to_string(), create_embedding("nurse"), 1),
            ("hospital".to_string(), create_embedding("hospital"), 2), // 2-hop
        ]
    );

    thread::sleep(Duration::from_millis(250)); // Optimal SOA

    let nurse_activation = priming_engine.compute_priming_boost("nurse");
    let hospital_activation = priming_engine.compute_priming_boost("hospital");
    let ambulance_activation = priming_engine.compute_priming_boost("ambulance");

    // Validate distance-dependent attenuation
    assert!(
        nurse_activation > 0.10,
        "1-hop neighbor should receive strong priming (>0.10), got {:.3}",
        nurse_activation
    );

    assert!(
        hospital_activation > 0.04 && hospital_activation < nurse_activation * 0.6,
        "2-hop neighbor should have attenuated priming (~50% of 1-hop), got {:.3}",
        hospital_activation
    );

    assert!(
        ambulance_activation < 0.02,
        "3-hop neighbor should have minimal priming (<0.02), got {:.3}",
        ambulance_activation
    );
}
```

#### Validation Criteria
- SOA 250-400ms: Facilitation effect 10-20%
- SOA <100ms: Minimal facilitation (<5%)
- Related vs unrelated difference significant (p < 0.05)
- Distance attenuation: 2-hop ~50% of 1-hop, 3-hop <20% of 1-hop

### 4. Spacing Effect Validation (Bjork & Bjork 1992)

#### Background
The spacing effect is one of the most robust findings in memory research:
- Distributed practice superior to massed practice
- Optimal spacing increases with retention interval
- Effect size: d = 0.4-0.8 depending on spacing/retention ratio

#### Test Design
```rust
/// Validates spacing effect in memory consolidation
///
/// Bjork & Bjork (1992) Key Findings:
/// - Massed practice (immediate repetition): Weak long-term retention
/// - Spaced practice (hours/days between): Strong long-term retention
/// - Optimal spacing ≈ 10-20% of retention interval
/// - Effect size: Cohen's d = 0.46 for moderate spacing
///
/// References:
/// - Bjork & Bjork (1992). "A new theory of disuse and an old theory of
///   stimulus fluctuation." In Healy et al., From Learning Processes to
///   Cognitive Processes, p. 35-67.
/// - Cepeda et al. (2006). "Distributed practice in verbal recall tasks."
///   Psychological Bulletin 132(3): 354-380. [Meta-analysis: p. 367]
#[test]
fn test_bjork_1992_spacing_effect() {
    let graph = UnifiedMemoryGraph::concurrent();
    let consolidation_engine = ConsolidationEngine::new();

    // Create two learning conditions:
    // 1. Massed: 3 repetitions with 1-minute spacing
    // 2. Spaced: 3 repetitions with 1-hour spacing

    let test_memory = Memory::new(
        "capital_of_france_paris".to_string(),
        create_embedding("Paris is the capital of France"),
        Confidence::MEDIUM,
    );

    // Massed condition
    let massed_memories = vec![
        (test_memory.clone(), Instant::now()),
        (test_memory.clone(), Instant::now() + Duration::from_secs(60)),
        (test_memory.clone(), Instant::now() + Duration::from_secs(120)),
    ];

    // Spaced condition
    let spaced_memories = vec![
        (test_memory.clone(), Instant::now()),
        (test_memory.clone(), Instant::now() + Duration::from_secs(3600)),
        (test_memory.clone(), Instant::now() + Duration::from_secs(7200)),
    ];

    // Store and consolidate both conditions
    let massed_id = store_with_timestamps(&graph, &massed_memories);
    let spaced_id = store_with_timestamps(&graph, &spaced_memories);

    // Run consolidation cycles simulating 24 hours
    for _ in 0..5 {
        consolidation_engine.consolidate_with_concepts(SleepStage::NREM2);
    }

    // Test retention after 24 hours
    let massed_retention = measure_retention_strength(&graph, massed_id);
    let spaced_retention = measure_retention_strength(&graph, spaced_id);

    // Validate spacing advantage
    assert!(
        spaced_retention > massed_retention * 1.3,
        "Spaced practice should show >30% retention advantage. \
         Spaced: {:.3}, Massed: {:.3}",
        spaced_retention,
        massed_retention
    );

    // Calculate effect size (Cohen's d)
    let effect_size = (spaced_retention - massed_retention) /
        ((massed_retention + spaced_retention) / 2.0);

    assert!(
        effect_size > 0.4,
        "Spacing effect size should be >0.4 per Cepeda et al. (2006), got {:.2}",
        effect_size
    );
}
```

#### Validation Criteria
- Spaced retention > massed retention by 30%
- Effect size Cohen's d > 0.4
- Advantage increases with longer retention intervals

### 5. Levels of Processing Validation (Craik & Lockhart 1972)

#### Background
Craik & Lockhart demonstrated that encoding depth affects retention:
- Shallow (structural): "Is the word in capital letters?" - Poor retention
- Intermediate (phonemic): "Does the word rhyme with TRAIN?" - Moderate retention
- Deep (semantic): "Does the word fit: The ___ is delicious?" - Strong retention

#### Test Design
```rust
/// Validates levels of processing effect
///
/// Craik & Tulving (1975) Experiment 1:
/// - Shallow (case): 18% recall
/// - Phonemic (rhyme): 78% recall
/// - Semantic (sentence): 93% recall
/// - Linear trend: F(1,18) = 98.4, p < 0.001
///
/// References:
/// - Craik & Lockhart (1972). "Levels of processing." JVL&VB 11(6): 671-684.
/// - Craik & Tulving (1975). "Depth of processing and retention of words."
///   JEP: General 104(3): 268-294. [Recall data: Table 1, p. 274]
#[test]
fn test_craik_lockhart_1972_levels_of_processing() {
    let graph = UnifiedMemoryGraph::concurrent();

    // Create memories with different processing depths
    // Simulated through embedding quality and contextual richness

    // Shallow processing: Simple perceptual features
    let shallow_embedding = create_sparse_embedding("table", 0.3); // Low richness
    let shallow_memory = Memory::new(
        "table".to_string(),
        shallow_embedding,
        Confidence::LOW,
    );

    // Phonemic processing: Phonological features
    let phonemic_embedding = create_sparse_embedding("table", 0.6); // Medium richness
    let phonemic_memory = Memory::new(
        "table_rhymes_with_cable".to_string(),
        phonemic_embedding,
        Confidence::MEDIUM,
    );

    // Deep semantic processing: Rich semantic context
    let semantic_embedding = create_dense_embedding("table", 0.95); // High richness
    let semantic_memory = Memory::new(
        "table_for_eating_dinner_with_family".to_string(),
        semantic_embedding,
        Confidence::HIGH,
    );

    // Store all three
    let shallow_id = graph.store_memory(shallow_memory).unwrap();
    let phonemic_id = graph.store_memory(phonemic_memory).unwrap();
    let semantic_id = graph.store_memory(semantic_memory).unwrap();

    // Simulate retention interval (24 hours)
    apply_temporal_decay(&graph, Duration::from_secs(86400));

    // Measure retention strength
    let shallow_strength = measure_retention_strength(&graph, shallow_id);
    let phonemic_strength = measure_retention_strength(&graph, phonemic_id);
    let semantic_strength = measure_retention_strength(&graph, semantic_id);

    // Validate levels of processing effect
    assert!(
        semantic_strength > phonemic_strength &&
        phonemic_strength > shallow_strength,
        "Retention should follow semantic > phonemic > shallow. Got {:.3} > {:.3} > {:.3}",
        semantic_strength,
        phonemic_strength,
        shallow_strength
    );

    // Validate effect magnitude (semantic should be ~3-5x shallow)
    let depth_ratio = semantic_strength / shallow_strength;
    assert!(
        depth_ratio > 3.0,
        "Semantic/shallow retention ratio should be >3.0, got {:.2}",
        depth_ratio
    );
}
```

#### Validation Criteria
- Retention ordering: Semantic > Phonemic > Shallow
- Semantic/shallow ratio > 3.0
- Linear trend across processing levels significant

### 6. Context-Dependent Memory Validation (Godden & Baddeley 1975)

#### Background
Godden & Baddeley showed encoding-retrieval context match improves recall:
- Learn on land, test on land: 13.5 words recalled
- Learn on land, test underwater: 8.6 words recalled
- Context match advantage: ~36% improvement

#### Test Design
```rust
/// Validates encoding-retrieval context match effect
///
/// Godden & Baddeley (1975) Key Findings:
/// - Matching context: 13.5 words (land-land, underwater-underwater)
/// - Mismatching context: 8.6 words (land-underwater, underwater-land)
/// - Context reinstatement advantage: 36%
/// - Effect size: η² = 0.62 (large effect)
///
/// References:
/// - Godden & Baddeley (1975). "Context-dependent memory in two natural
///   environments." British Journal of Psychology 66(3): 325-331. [Data: Table 1]
#[test]
fn test_godden_baddeley_1975_context_dependent_memory() {
    let graph = UnifiedMemoryGraph::concurrent();

    // Create contextual embeddings
    let land_context = create_embedding("land_environment_features");
    let water_context = create_embedding("underwater_environment_features");

    // Create memories with different contexts
    let word_list = vec![
        "pencil", "clock", "mountain", "river", "book",
        "window", "ocean", "forest", "cloud", "garden"
    ];

    // Encode in land context
    let land_encoded_ids = word_list.iter().map(|word| {
        let embedding = blend_embeddings(
            &create_embedding(word),
            &land_context,
            0.3 // 30% context weight
        );

        graph.store_memory(Memory::new(
            format!("{}_{}", word, "land_context"),
            embedding,
            Confidence::MEDIUM,
        )).unwrap()
    }).collect::<Vec<_>>();

    // Encode in underwater context
    let water_encoded_ids = word_list.iter().map(|word| {
        let embedding = blend_embeddings(
            &create_embedding(word),
            &water_context,
            0.3 // 30% context weight
        );

        graph.store_memory(Memory::new(
            format!("{}_{}", word, "water_context"),
            embedding,
            Confidence::MEDIUM,
        )).unwrap()
    }).collect::<Vec<_>>();

    // Test recall in matching vs mismatching contexts

    // Matching: land-land
    let matching_land_recall = recall_with_context(
        &graph,
        &land_encoded_ids,
        &land_context
    );

    // Mismatching: land-water
    let mismatching_land_recall = recall_with_context(
        &graph,
        &land_encoded_ids,
        &water_context
    );

    // Matching: water-water
    let matching_water_recall = recall_with_context(
        &graph,
        &water_encoded_ids,
        &water_context
    );

    // Mismatching: water-land
    let mismatching_water_recall = recall_with_context(
        &graph,
        &water_encoded_ids,
        &land_context
    );

    // Calculate average matching vs mismatching
    let avg_matching = (matching_land_recall + matching_water_recall) / 2.0;
    let avg_mismatching = (mismatching_land_recall + mismatching_water_recall) / 2.0;

    // Validate context match advantage
    let context_advantage = (avg_matching - avg_mismatching) / avg_mismatching;

    assert!(
        context_advantage > 0.25,
        "Context match should improve recall by >25%, got {:.1}%",
        context_advantage * 100.0
    );

    // Calculate effect size (eta-squared)
    let effect_size = calculate_eta_squared(
        avg_matching,
        avg_mismatching,
        measure_variance(&[matching_land_recall, matching_water_recall,
                          mismatching_land_recall, mismatching_water_recall])
    );

    assert!(
        effect_size > 0.4,
        "Context effect size should be large (η² > 0.4), got {:.2}",
        effect_size
    );
}
```

#### Validation Criteria
- Context match advantage > 25%
- Effect size η² > 0.4 (large effect)
- Symmetric advantage for both contexts

### 7. Statistical Analysis Framework

```rust
/// Statistical analysis utilities for psychological validation
pub mod statistical_analysis {
    /// Calculate Pearson correlation coefficient
    ///
    /// Formula: r = Σ[(x - x̄)(y - ȳ)] / √[Σ(x - x̄)² Σ(y - ȳ)²]
    ///
    /// # Returns
    /// Correlation coefficient in [-1.0, 1.0]
    pub fn calculate_pearson_correlation(x: &[f32], y: &[f32]) -> f32 {
        assert_eq!(x.len(), y.len(), "Arrays must have equal length");

        let n = x.len() as f32;
        let x_mean = mean(x);
        let y_mean = mean(y);

        let numerator: f32 = x.iter()
            .zip(y.iter())
            .map(|(xi, yi)| (xi - x_mean) * (yi - y_mean))
            .sum();

        let x_variance: f32 = x.iter()
            .map(|xi| (xi - x_mean).powi(2))
            .sum();

        let y_variance: f32 = y.iter()
            .map(|yi| (yi - y_mean).powi(2))
            .sum();

        numerator / (x_variance * y_variance).sqrt()
    }

    /// Calculate Cohen's d effect size
    ///
    /// Formula: d = (M₁ - M₂) / SD_pooled
    ///
    /// Interpretation:
    /// - Small: d = 0.2
    /// - Medium: d = 0.5
    /// - Large: d = 0.8
    pub fn calculate_cohens_d(
        mean1: f32,
        mean2: f32,
        std1: f32,
        std2: f32,
        n1: usize,
        n2: usize,
    ) -> f32 {
        let pooled_variance = ((n1 - 1) as f32 * std1.powi(2) +
                              (n2 - 1) as f32 * std2.powi(2)) /
                             ((n1 + n2 - 2) as f32);

        let pooled_std = pooled_variance.sqrt();

        (mean1 - mean2) / pooled_std
    }

    /// Two-sample t-test for significance testing
    ///
    /// Returns: (t_statistic, degrees_of_freedom, two_tailed_p_value)
    pub fn t_test_independent(
        sample1: &[f32],
        sample2: &[f32],
    ) -> (f32, usize, f32) {
        let n1 = sample1.len();
        let n2 = sample2.len();

        let mean1 = mean(sample1);
        let mean2 = mean(sample2);

        let var1 = variance(sample1);
        let var2 = variance(sample2);

        // Welch's t-test (unequal variances)
        let t_stat = (mean1 - mean2) /
            ((var1 / n1 as f32) + (var2 / n2 as f32)).sqrt();

        // Welch-Satterthwaite degrees of freedom
        let df_numerator = ((var1 / n1 as f32) + (var2 / n2 as f32)).powi(2);
        let df_denominator =
            (var1 / n1 as f32).powi(2) / (n1 - 1) as f32 +
            (var2 / n2 as f32).powi(2) / (n2 - 1) as f32;

        let df = (df_numerator / df_denominator).floor() as usize;

        // Calculate p-value using t-distribution
        let p_value = t_distribution_two_tailed(t_stat.abs(), df);

        (t_stat, df, p_value)
    }

    /// Calculate eta-squared effect size for ANOVA
    ///
    /// Formula: η² = SS_between / SS_total
    ///
    /// Interpretation:
    /// - Small: η² = 0.01
    /// - Medium: η² = 0.06
    /// - Large: η² = 0.14
    pub fn calculate_eta_squared(
        group1_mean: f32,
        group2_mean: f32,
        total_variance: f32,
    ) -> f32 {
        let between_group_variance = (group1_mean - group2_mean).powi(2) / 2.0;
        between_group_variance / (between_group_variance + total_variance)
    }

    /// Linear regression slope calculation
    ///
    /// Returns slope (beta coefficient) for y = alpha + beta*x
    pub fn calculate_linear_slope(x: &[f32], y: &[f32]) -> f32 {
        assert_eq!(x.len(), y.len());

        let x_mean = mean(x);
        let y_mean = mean(y);

        let numerator: f32 = x.iter()
            .zip(y.iter())
            .map(|(xi, yi)| (xi - x_mean) * (yi - y_mean))
            .sum();

        let denominator: f32 = x.iter()
            .map(|xi| (xi - x_mean).powi(2))
            .sum();

        numerator / denominator
    }

    /// Helper: Calculate mean
    fn mean(data: &[f32]) -> f32 {
        data.iter().sum::<f32>() / data.len() as f32
    }

    /// Helper: Calculate variance
    fn variance(data: &[f32]) -> f32 {
        let mean = mean(data);
        data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / (data.len() - 1) as f32
    }

    /// Helper: Two-tailed p-value from t-distribution
    ///
    /// Uses approximation for t-distribution CDF
    fn t_distribution_two_tailed(t: f32, df: usize) -> f32 {
        // Simplified implementation - in production would use statistical library
        // This approximation works for df > 30
        if df > 30 {
            // Approximate with normal distribution
            2.0 * normal_cdf(-t.abs())
        } else {
            // Use lookup table or numerical integration
            t_distribution_cdf_approximation(t, df)
        }
    }

    /// Placeholder for normal CDF (would use statistical library)
    fn normal_cdf(z: f32) -> f32 {
        0.5 * (1.0 + erf(z / std::f32::consts::SQRT_2))
    }

    /// Error function approximation
    fn erf(x: f32) -> f32 {
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t *
                       (-x * x).exp();

        sign * y
    }

    /// Placeholder for t-distribution CDF
    fn t_distribution_cdf_approximation(_t: f32, _df: usize) -> f32 {
        // Would use numerical integration or lookup table
        // For now, return conservative p-value
        0.05
    }
}
```

### 8. Test Datasets

```rust
/// Shared test datasets for psychological validation
pub mod test_datasets {
    use super::*;

    /// Anderson (1974) person-location pairs
    pub fn anderson_1974_materials() -> Vec<(String, String, usize)> {
        // (person, location, person_fan)
        vec![
            // Fan 1
            ("doctor".to_string(), "park".to_string(), 1),
            ("lawyer".to_string(), "store".to_string(), 1),

            // Fan 2
            ("fireman".to_string(), "park".to_string(), 2),
            ("fireman".to_string(), "bank".to_string(), 2),

            // Fan 3
            ("teacher".to_string(), "church".to_string(), 3),
            ("teacher".to_string(), "bank".to_string(), 3),
            ("teacher".to_string(), "park".to_string(), 3),
        ]
    }

    /// Rosch (1975) bird category exemplars with typicality ratings
    pub fn rosch_1975_bird_exemplars() -> Vec<(String, Vec<String>, f32)> {
        // (exemplar, features, typicality_rating_0_to_1)
        vec![
            // Prototypical (ratings 6.5-7.0 on 7-point scale)
            ("robin".to_string(),
             vec!["flies", "small", "sings", "builds_nests"],
             0.98),
            ("sparrow".to_string(),
             vec!["flies", "small", "sings", "common"],
             0.95),
            ("bluebird".to_string(),
             vec!["flies", "small", "colorful", "sings"],
             0.93),

            // Atypical (ratings 3.5-4.5 on 7-point scale)
            ("penguin".to_string(),
             vec!["swims", "flightless", "cold_climate", "large"],
             0.56),
            ("ostrich".to_string(),
             vec!["runs", "flightless", "very_large", "desert"],
             0.50),
            ("emu".to_string(),
             vec!["runs", "flightless", "large", "australia"],
             0.53),
        ]
    }

    /// Neely (1977) semantic priming word pairs
    pub fn neely_1977_word_pairs() -> (Vec<(String, String)>, Vec<(String, String)>) {
        // (related_pairs, unrelated_pairs)
        let related = vec![
            ("doctor".to_string(), "nurse".to_string()),
            ("bread".to_string(), "butter".to_string()),
            ("lion".to_string(), "tiger".to_string()),
            ("chair".to_string(), "table".to_string()),
            ("king".to_string(), "queen".to_string()),
            ("salt".to_string(), "pepper".to_string()),
        ];

        let unrelated = vec![
            ("doctor".to_string(), "butter".to_string()),
            ("bread".to_string(), "tiger".to_string()),
            ("lion".to_string(), "table".to_string()),
            ("chair".to_string(), "nurse".to_string()),
            ("king".to_string(), "pepper".to_string()),
            ("salt".to_string(), "queen".to_string()),
        ];

        (related, unrelated)
    }

    /// Create embedding for test word
    ///
    /// In production tests, would use actual sentence transformer embeddings.
    /// For validation tests, creates synthetic embeddings with controlled similarity.
    pub fn create_embedding(word: &str) -> [f32; 768] {
        // Deterministic seeded random embedding based on word
        let seed = word.bytes().map(|b| b as u64).sum::<u64>();
        let mut embedding = [0.0f32; 768];

        for (i, elem) in embedding.iter_mut().enumerate() {
            let val = ((seed + i as u64) as f32).sin();
            *elem = val;
        }

        // Normalize
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        for elem in &mut embedding {
            *elem /= norm;
        }

        embedding
    }

    /// Create sparse embedding (for shallow processing)
    pub fn create_sparse_embedding(word: &str, density: f32) -> [f32; 768] {
        let mut embedding = create_embedding(word);

        // Zero out elements to create sparsity
        for (i, elem) in embedding.iter_mut().enumerate() {
            if (i as f32 / 768.0) > density {
                *elem = 0.0;
            }
        }

        // Re-normalize
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for elem in &mut embedding {
                *elem /= norm;
            }
        }

        embedding
    }

    /// Create dense embedding with semantic richness (for deep processing)
    pub fn create_dense_embedding(word: &str, richness: f32) -> [f32; 768] {
        let base = create_embedding(word);
        let context = create_embedding(&format!("{}_semantic_context", word));

        // Blend base and context weighted by richness
        let mut embedding = [0.0f32; 768];
        for i in 0..768 {
            embedding[i] = (1.0 - richness) * base[i] + richness * context[i];
        }

        // Normalize
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        for elem in &mut embedding {
            *elem /= norm;
        }

        embedding
    }
}
```

## Integration Testing Strategy

### Automated CI/CD Integration
```rust
/// Integration test suite that runs in CI/CD
///
/// Uses #[ignore] for tests requiring long runtime (>1min)
/// All other tests should complete in <10s for CI feedback loop
#[cfg(test)]
mod ci_integration {
    use super::*;

    #[test]
    fn psychological_validation_fast_suite() {
        // Run all validation tests that complete quickly
        test_anderson_1974_person_location_paradigm();
        test_rosch_1975_prototype_effects();
        test_neely_1977_semantic_priming_soa();
    }

    #[test]
    #[ignore] // Long-running test (>1min)
    fn psychological_validation_full_suite() {
        // Run comprehensive validation including slow tests
        test_bjork_1992_spacing_effect();
        test_craik_lockhart_1972_levels_of_processing();
        test_godden_baddeley_1975_context_dependent_memory();
    }
}
```

## Acceptance Criteria

### Empirical Correlation Targets
- [ ] Fan effect correlation with Anderson (1974): r > 0.8
- [ ] Fan effect slope: 70ms ± 10ms per association
- [ ] Rosch prototype advantage: typical > atypical by 30% in activation
- [ ] Neely priming facilitation at SOA 250-400ms: 10-20%
- [ ] Spacing effect advantage: spaced > massed by 30%
- [ ] Spacing effect size: Cohen's d > 0.4
- [ ] Levels of processing: semantic > phonemic > shallow
- [ ] Context-dependent memory advantage: matching > mismatching by 25%

### Statistical Validation
- [ ] All significant effects have p < 0.05
- [ ] Effect sizes match literature (Cohen's d, η²)
- [ ] Correlation coefficients computed with proper significance tests
- [ ] No Type I errors (false positives) in validation suite

### Documentation
- [ ] Each test documents empirical source (paper, table, page)
- [ ] Deviations from empirical data explained with justification
- [ ] Statistical methods documented with formulas
- [ ] Test dataset sources cited

### Performance
- [ ] Fast test suite completes in <10s (for CI/CD)
- [ ] Full test suite completes in <5min
- [ ] Statistical computations use efficient algorithms (O(n) or O(n log n))

## Dependencies
- Task 001 (Dual Memory Types) - Memory node infrastructure
- Task 004 (Concept Formation) - Category/prototype testing
- Task 007 (Fan Effect) - Anderson (1974) validation already implemented
- Task 009 (Blended Recall) - Context-dependent memory testing
- Existing priming infrastructure (engram-core/src/cognitive/priming/)
- Existing consolidation system (engram-core/src/consolidation/)

## Key References

### Fan Effect
1. **Anderson (1974)** - "Retrieval of propositional information from long-term memory." Cognitive Psychology 6(4): 451-474. [Person-location paradigm: Table 1]

### Category Formation
2. **Rosch (1975)** - "Cognitive representations of semantic categories." JEP: General 104(3): 192-233. [Typicality ratings: Table 3]
3. **Rosch & Mervis (1975)** - "Family resemblances: Studies in the internal structure of categories." Cognitive Psychology 7(4): 573-605.

### Semantic Priming
4. **Neely (1977)** - "Semantic priming and retrieval from lexical memory: Roles of inhibitionless spreading activation and limited-capacity attention." JEP: General 106(3): 226-254. [SOA effects: Table 2, p. 238]
5. **Collins & Loftus (1975)** - "A spreading-activation theory of semantic processing." Psychological Review 82(6): 407-428.

### Spacing Effect
6. **Bjork & Bjork (1992)** - "A new theory of disuse and an old theory of stimulus fluctuation." In Healy et al., From Learning Processes to Cognitive Processes, p. 35-67.
7. **Cepeda et al. (2006)** - "Distributed practice in verbal recall tasks: A review and quantitative synthesis." Psychological Bulletin 132(3): 354-380. [Meta-analysis: p. 367]

### Levels of Processing
8. **Craik & Lockhart (1972)** - "Levels of processing: A framework for memory research." JVL&VB 11(6): 671-684.
9. **Craik & Tulving (1975)** - "Depth of processing and the retention of words in episodic memory." JEP: General 104(3): 268-294. [Recall data: Table 1, p. 274]

### Context-Dependent Memory
10. **Godden & Baddeley (1975)** - "Context-dependent memory in two natural environments: On land and underwater." British Journal of Psychology 66(3): 325-331. [Data: Table 1]
11. **Smith & Vela (2001)** - "Environmental context-dependent memory: A review and meta-analysis." Psychonomic Bulletin & Review 8(2): 203-220. [Effect sizes]

## Implementation Notes

### Biological Plausibility
All tests validate emergent behavior arising from biologically-plausible mechanisms:
- Fan effect: Resource conservation in spreading activation (fixed neural firing rates)
- Prototype effects: Pattern completion in CA3 attractor networks
- Priming: Hebbian residual activation with temporal decay
- Spacing effect: Consolidation strengthening through repeated replay
- Levels of processing: Encoding richness determines consolidation priority
- Context-dependent memory: Hippocampal pattern separation/completion with context cues

### Statistical Rigor
Use proper statistical methods throughout:
- Pearson r for correlation (with significance testing)
- Cohen's d for effect sizes (standardized mean differences)
- Eta-squared (η²) for ANOVA effect sizes
- Two-tailed t-tests with Welch correction for unequal variances
- Conservative p-value thresholds (p < 0.05, Bonferroni correction for multiple comparisons)

### Determinism
All tests must be deterministic for CI/CD:
- Use seeded random number generators
- Sort operations before processing
- Use stable floating-point summation (Kahan summation for centroids)
- Document any unavoidable non-determinism with justification

### Deviations from Empirical Data
Expected deviations and their justifications:
1. **Embedding-based similarity** may not perfectly match human semantic judgments (acceptable if correlation >0.7)
2. **Temporal dynamics** compressed from hours/days to seconds/minutes for testing (scaling factor documented)
3. **Perfect recall** not achievable with lossy compression (acceptable if >80% of empirical recall rate)
4. **Context effects** may be stronger/weaker depending on embedding space geometry (document actual effect sizes)

## Estimated Time
4 days

## Notes
This task creates the empirical validation framework that demonstrates Engram's memory system produces human-like memory phenomena. By grounding tests in classic psychology experiments with specific empirical targets, we ensure the dual memory architecture is not just biologically plausible in mechanism but also psychologically valid in behavior.

The statistical analysis framework enables rigorous quantitative validation rather than qualitative assertions. Each test specifies exact correlation targets, effect size thresholds, and significance levels drawn from the published literature.

This validation framework will serve as ongoing regression testing as the memory system evolves, ensuring that architectural changes don't inadvertently break alignment with cognitive science findings.
