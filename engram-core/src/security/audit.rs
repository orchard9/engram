//! Security audit logging infrastructure.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use slog::{Drain, Logger};
use std::sync::Arc;

/// Audit event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditEventType {
    /// Authentication event
    Authentication,
    /// Authorization event
    Authorization,
    /// Data access event
    DataAccess,
    /// Data modification event
    DataModification,
    /// Configuration change
    Configuration,
    /// System access event
    SystemAccess,
}

/// Audit result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditResult {
    /// Operation succeeded
    Success,
    /// Operation failed
    Failure(String),
    /// Operation partially succeeded
    PartialSuccess(String),
}

/// Security audit event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    /// Event timestamp
    pub timestamp: DateTime<Utc>,

    /// Event type
    pub event_type: AuditEventType,

    /// Principal performing action
    pub principal: Option<String>,

    /// Target resource
    pub resource: Option<String>,

    /// Operation details
    pub operation: String,

    /// Result (success/failure)
    pub result: AuditResult,

    /// Additional metadata
    pub metadata: serde_json::Value,

    /// Request correlation ID
    pub correlation_id: String,

    /// Source IP address
    pub source_ip: Option<String>,
}

impl AuditEvent {
    /// Create a new audit event
    #[must_use]
    pub fn new(event_type: AuditEventType, operation: String) -> Self {
        Self {
            timestamp: Utc::now(),
            event_type,
            principal: None,
            resource: None,
            operation,
            result: AuditResult::Success,
            metadata: serde_json::Value::Object(serde_json::Map::new()),
            correlation_id: uuid::Uuid::new_v4().to_string(),
            source_ip: None,
        }
    }

    /// Set principal
    #[must_use]
    pub fn with_principal(mut self, principal: String) -> Self {
        self.principal = Some(principal);
        self
    }

    /// Set resource
    #[must_use]
    pub fn with_resource(mut self, resource: String) -> Self {
        self.resource = Some(resource);
        self
    }

    /// Set result
    #[must_use]
    pub fn with_result(mut self, result: AuditResult) -> Self {
        self.result = result;
        self
    }

    /// Set source IP
    #[must_use]
    pub fn with_source_ip(mut self, source_ip: String) -> Self {
        self.source_ip = Some(source_ip);
        self
    }

    /// Set correlation ID
    #[must_use]
    pub fn with_correlation_id(mut self, correlation_id: String) -> Self {
        self.correlation_id = correlation_id;
        self
    }
}

/// Audit logger for security events
pub struct AuditLogger {
    /// Structured logger
    logger: Arc<Logger>,
}

impl AuditLogger {
    /// Create a new audit logger with JSON output
    ///
    /// # Errors
    ///
    /// Returns error if logger cannot be initialized
    #[allow(clippy::unnecessary_wraps)]
    pub fn new_json() -> Result<Self, super::SecurityError> {
        let drain = slog_json::Json::default(std::io::stdout()).fuse();
        let drain = slog_async::Async::new(drain).build().fuse();
        let logger = Logger::root(drain, slog::o!("subsystem" => "audit"));

        Ok(Self {
            logger: Arc::new(logger),
        })
    }

    /// Create a new audit logger with custom drain
    #[must_use]
    pub fn new_with_logger(logger: Logger) -> Self {
        Self {
            logger: Arc::new(logger),
        }
    }

    /// Log security event
    pub fn log_event(&self, event: &AuditEvent) {
        let anonymous = String::from("anonymous");
        let empty = String::new();

        slog::info!(self.logger, "AUDIT";
            "timestamp" => event.timestamp.to_rfc3339(),
            "event_type" => format!("{:?}", event.event_type),
            "principal" => event.principal.as_ref().unwrap_or(&anonymous),
            "resource" => event.resource.as_ref().unwrap_or(&empty),
            "operation" => &event.operation,
            "result" => format!("{:?}", event.result),
            "correlation_id" => &event.correlation_id,
            "source_ip" => event.source_ip.as_ref().unwrap_or(&empty),
            "metadata" => event.metadata.to_string()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_event_builder() {
        let event = AuditEvent::new(AuditEventType::Authentication, "login".to_string())
            .with_principal("user123".to_string())
            .with_resource("memory_space_1".to_string())
            .with_result(AuditResult::Success)
            .with_source_ip("192.168.1.1".to_string());

        assert_eq!(event.principal, Some("user123".to_string()));
        assert_eq!(event.resource, Some("memory_space_1".to_string()));
        assert_eq!(event.source_ip, Some("192.168.1.1".to_string()));
    }
}
