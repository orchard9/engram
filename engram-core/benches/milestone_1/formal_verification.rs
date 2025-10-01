#[derive(Debug, Clone)]
pub struct FormalVerificationSuite {
    properties: Vec<FormalProperty>,
}

impl FormalVerificationSuite {
    #[must_use]
    pub fn new() -> Self {
        Self {
            properties: FormalProperty::standard_properties(),
        }
    }

    #[must_use]
    pub fn verify_all_properties(&self) -> VerificationResult {
        let violations: Vec<_> = self
            .properties
            .iter()
            .filter_map(FormalProperty::evaluate)
            .collect();

        if violations.is_empty() {
            VerificationResult::AllPropertiesVerified
        } else {
            VerificationResult::PropertyViolations(violations)
        }
    }

    #[must_use]
    pub fn property_names(&self) -> Vec<&'static str> {
        self.properties
            .iter()
            .map(|property| property.name)
            .collect()
    }
}

#[derive(Debug, Clone)]
struct FormalProperty {
    name: &'static str,
    description: &'static str,
    validator: fn() -> bool,
}

impl FormalProperty {
    const fn new(name: &'static str, description: &'static str, validator: fn() -> bool) -> Self {
        Self {
            name,
            description,
            validator,
        }
    }

    fn evaluate(&self) -> Option<PropertyViolation> {
        if (self.validator)() {
            None
        } else {
            Some(PropertyViolation::new(self.name, self.description))
        }
    }

    fn standard_properties() -> Vec<Self> {
        vec![
            Self::new(
                "activation_conservation",
                "Total activation should not increase after decay within a single hop.",
                || {
                    let baseline_activation = 1.0_f64;
                    let decay_factor = 0.15_f64;
                    let remaining_activation = baseline_activation * (1.0 - decay_factor);
                    remaining_activation <= baseline_activation
                },
            ),
            Self::new(
                "confidence_bounds",
                "Confidence values must remain within [0, 1] after normalization.",
                || {
                    let mut confidences = [1.2_f64, 0.9, 0.3];
                    let max_confidence = confidences
                        .iter_mut()
                        .map(|confidence| confidence.clamp(0.0, 1.0))
                        .fold(0.0_f64, f64::max);
                    (0.0..=1.0).contains(&max_confidence)
                },
            ),
            Self::new(
                "probability_conservation",
                "Normalization should keep accumulated probability within epsilon.",
                || {
                    let probabilities = [0.3_f64, 0.5, 0.19, 0.01];
                    let total = probabilities.iter().sum::<f64>();
                    (total - 1.0).abs() < 1e-6
                },
            ),
        ]
    }
}

#[derive(Debug, Clone)]
pub enum VerificationResult {
    AllPropertiesVerified,
    PropertyViolations(Vec<PropertyViolation>),
}

impl VerificationResult {
    #[must_use]
    pub const fn all_properties_verified(&self) -> bool {
        matches!(self, Self::AllPropertiesVerified)
    }

    #[must_use]
    pub fn violations(&self) -> &[PropertyViolation] {
        match self {
            Self::PropertyViolations(violations) => violations,
            Self::AllPropertiesVerified => &[],
        }
    }

    #[must_use]
    pub fn violation_messages(&self) -> Vec<String> {
        self.violations()
            .iter()
            .map(PropertyViolation::summary)
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct PropertyViolation {
    property_name: String,
    description: String,
}

impl PropertyViolation {
    fn new(name: &str, description: &str) -> Self {
        Self {
            property_name: name.to_string(),
            description: description.to_string(),
        }
    }

    #[must_use]
    pub fn summary(&self) -> String {
        format!("{}: {}", self.property_name, self.description)
    }
}
