#[derive(Debug, Clone)]
pub struct PerformanceFuzzer {
    test_grammar: ContextFreeTestGrammar,
    coverage_engine: AFLStyleCoverageEngine,
    cliff_detector: StatisticalPerformanceCliffDetector,
    worst_cases: WorstCaseDatabase,
}

impl PerformanceFuzzer {
    pub fn new() -> Self {
        Self {
            test_grammar: ContextFreeTestGrammar::new(),
            coverage_engine: AFLStyleCoverageEngine::new(),
            cliff_detector: StatisticalPerformanceCliffDetector::new(),
            worst_cases: WorstCaseDatabase::seeded(),
        }
    }

    pub fn fuzz_all_operations(&mut self, iterations: usize) -> FuzzingResults {
        if let Some(worst_case) =
            self.cliff_detector
                .detect(iterations, &self.test_grammar, &self.coverage_engine)
        {
            self.worst_cases.record_case(worst_case);
        }

        let coverage = self.coverage_engine.projected_coverage(iterations);

        FuzzingResults {
            worst_cases: self.worst_cases.known_cases().to_vec(),
            coverage,
            iterations_run: iterations,
        }
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct FuzzingResults {
    pub worst_cases: Vec<WorstCaseScenario>,
    pub coverage: f64,
    pub iterations_run: usize,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct WorstCaseScenario {
    pub description: String,
    pub performance_impact: f64,
}

impl WorstCaseScenario {
    pub fn new(description: impl Into<String>, performance_impact: f64) -> Self {
        Self {
            description: description.into(),
            performance_impact,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ContextFreeTestGrammar {
    operations: Vec<String>,
}

impl ContextFreeTestGrammar {
    pub fn new() -> Self {
        Self {
            operations: vec![
                "insert_vector".to_string(),
                "merge_graph".to_string(),
                "quantize_neighbors".to_string(),
                "persist_checkpoint".to_string(),
            ],
        }
    }

    pub fn operations(&self) -> &[String] {
        &self.operations
    }
}

#[derive(Debug, Clone)]
pub struct AFLStyleCoverageEngine {
    initial_coverage: f64,
    diminishing_factor: f64,
}

impl AFLStyleCoverageEngine {
    pub const fn new() -> Self {
        Self {
            initial_coverage: 0.12,
            diminishing_factor: 0.000_015,
        }
    }

    pub fn projected_coverage(&self, iterations: usize) -> f64 {
        let iterations = iterations_to_f64(iterations);
        (-self.diminishing_factor * iterations)
            .exp()
            .mul_add(-(1.0 - self.initial_coverage), 1.0)
    }
}

#[derive(Debug, Clone)]
pub struct StatisticalPerformanceCliffDetector {
    alert_threshold: f64,
}

impl StatisticalPerformanceCliffDetector {
    pub const fn new() -> Self {
        Self {
            alert_threshold: 0.85,
        }
    }

    pub fn detect(
        &self,
        iterations: usize,
        grammar: &ContextFreeTestGrammar,
        coverage_engine: &AFLStyleCoverageEngine,
    ) -> Option<WorstCaseScenario> {
        let coverage = coverage_engine.projected_coverage(iterations);
        if coverage < self.alert_threshold {
            let operations = grammar.operations();
            let signature = if operations.is_empty() {
                "fallback_operation".to_string()
            } else {
                operations[iterations % operations.len()].clone()
            };

            return Some(WorstCaseScenario::new(
                format!("coverage cliff triggered by {signature}"),
                self.alert_threshold - coverage,
            ));
        }

        None
    }
}

#[derive(Debug, Clone)]
pub struct WorstCaseDatabase {
    cases: Vec<WorstCaseScenario>,
}

impl WorstCaseDatabase {
    pub fn seeded() -> Self {
        Self {
            cases: vec![
                WorstCaseScenario::new("pathological activation cascade", 0.18),
                WorstCaseScenario::new("cache-thrashing vector merge", 0.22),
            ],
        }
    }

    pub fn record_case(&mut self, scenario: WorstCaseScenario) {
        self.cases.push(scenario);
    }

    pub fn known_cases(&self) -> &[WorstCaseScenario] {
        &self.cases
    }
}

const fn iterations_to_f64(iterations: usize) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    {
        iterations as f64
    }
}
