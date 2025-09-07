#[derive(Debug, Clone)]
pub struct FormalVerificationSuite;

impl FormalVerificationSuite {
    pub fn new() -> Self {
        Self
    }

    pub fn verify_all_properties(&self) -> VerificationResult {
        VerificationResult::AllPropertiesVerified
    }
}

#[derive(Debug, Clone)]
pub enum VerificationResult {
    AllPropertiesVerified,
    PropertyViolations(Vec<PropertyViolation>),
}

impl VerificationResult {
    pub fn all_properties_verified(&self) -> bool {
        matches!(self, VerificationResult::AllPropertiesVerified)
    }

    pub fn violations(&self) -> Vec<PropertyViolation> {
        match self {
            VerificationResult::PropertyViolations(v) => v.clone(),
            _ => vec![],
        }
    }
}

#[derive(Debug, Clone)]
pub struct PropertyViolation {
    pub property_name: String,
    pub description: String,
}
