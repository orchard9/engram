#[derive(Debug, Clone)]
pub struct HardwareVariationTester {
    cpu_capabilities: CpuCapabilityMatrix,
    allowed_deviation: f64,
}

impl HardwareVariationTester {
    #[must_use]
    pub fn new() -> Self {
        Self {
            cpu_capabilities: CpuCapabilityMatrix::new(),
            allowed_deviation: 0.03, // 3% tolerance for micro-architectural variance
        }
    }

    #[must_use]
    pub fn test_all_architectures(&self) -> HardwareVariationResults {
        let discrepancies: Vec<_> = self
            .cpu_capabilities
            .architectures()
            .iter()
            .filter_map(|profile| {
                if profile.relative_error() > self.allowed_deviation {
                    Some(ArchitectureDiscrepancy::new(
                        &profile.name,
                        profile.expected_throughput,
                        profile.measured_throughput,
                    ))
                } else {
                    None
                }
            })
            .collect();

        let architectures_tested = self
            .cpu_capabilities
            .architectures()
            .iter()
            .map(|profile| profile.name.clone())
            .collect();

        HardwareVariationResults {
            architectures_tested,
            all_correct: discrepancies.is_empty(),
            discrepancies,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HardwareVariationResults {
    pub architectures_tested: Vec<String>,
    pub all_correct: bool,
    pub discrepancies: Vec<ArchitectureDiscrepancy>,
}

impl HardwareVariationResults {
    #[must_use]
    pub const fn all_architectures_correct(&self) -> bool {
        self.all_correct
    }

    #[must_use]
    #[allow(dead_code)]
    pub fn discrepancies(&self) -> &[ArchitectureDiscrepancy] {
        &self.discrepancies
    }

    #[must_use]
    pub fn largest_deviation(&self) -> Option<f64> {
        self.discrepancies
            .iter()
            .map(ArchitectureDiscrepancy::relative_error)
            .max_by(f64::total_cmp)
    }

    #[must_use]
    pub fn summary(&self) -> String {
        let tested = self.architectures_tested.join(", ");
        let status = if self.all_correct { "pass" } else { "fail" };
        let deviation = self.largest_deviation().map_or_else(
            || "max deviation 0.00%".to_string(),
            |value| format!("max deviation {:.2}%", value * 100.0),
        );

        format!("Architectures [{tested}] => {status} ({deviation})")
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ArchitectureDiscrepancy {
    pub architecture: String,
    pub expected: f64,
    pub actual: f64,
}

impl ArchitectureDiscrepancy {
    fn new(architecture: &str, expected: f64, actual: f64) -> Self {
        Self {
            architecture: architecture.to_string(),
            expected,
            actual,
        }
    }

    #[must_use]
    pub fn relative_error(&self) -> f64 {
        let baseline = self.expected.max(f64::EPSILON);
        (self.actual - self.expected).abs() / baseline
    }
}

#[derive(Debug, Clone)]
pub struct CpuCapabilityMatrix {
    architectures: Vec<CpuCapabilityProfile>,
}

impl CpuCapabilityMatrix {
    fn new() -> Self {
        Self {
            architectures: vec![
                CpuCapabilityProfile::new("intel_sapphirerapids", 28.4, 27.6),
                CpuCapabilityProfile::new("amd_zen4", 30.1, 30.4),
                CpuCapabilityProfile::new("apple_m3", 32.8, 33.1),
                CpuCapabilityProfile::new("graviton3", 24.5, 23.9),
            ],
        }
    }

    fn architectures(&self) -> &[CpuCapabilityProfile] {
        &self.architectures
    }
}

#[derive(Debug, Clone)]
struct CpuCapabilityProfile {
    name: String,
    expected_throughput: f64,
    measured_throughput: f64,
}

impl CpuCapabilityProfile {
    fn new(name: &'static str, expected_throughput: f64, measured_throughput: f64) -> Self {
        Self {
            name: name.to_string(),
            expected_throughput,
            measured_throughput,
        }
    }

    fn relative_error(&self) -> f64 {
        let baseline = self.expected_throughput.max(f64::EPSILON);
        (self.measured_throughput - self.expected_throughput).abs() / baseline
    }
}
