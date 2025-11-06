# Task: Key Rotation Workflow

## Objective
Implement a complete API key rotation workflow that allows zero-downtime key updates with configurable grace periods and automated notifications.

## Context
Production systems need to rotate API keys regularly without service interruption. This requires supporting multiple valid keys during transition periods and notifying consumers.

## Requirements

### 1. Key Versioning System
Extend key storage to support versions:
```rust
#[derive(Clone, Debug)]
pub struct ApiKeyVersion {
    /// Version number (incrementing)
    pub version: u32,
    
    /// Version status
    pub status: KeyVersionStatus,
    
    /// When this version was created
    pub created_at: DateTime<Utc>,
    
    /// When this version becomes active
    pub active_from: DateTime<Utc>,
    
    /// When this version expires
    pub expires_at: Option<DateTime<Utc>>,
    
    /// Secret hash for this version
    pub secret_hash: String,
}

#[derive(Clone, Debug, PartialEq)]
pub enum KeyVersionStatus {
    /// Future version (not yet active)
    Pending,
    /// Currently active version
    Active,
    /// Grace period (still valid but superseded)
    Deprecated,
    /// No longer valid
    Expired,
    /// Manually revoked
    Revoked,
}
```

### 2. Rotation State Machine
Implement rotation workflow:
```rust
pub struct KeyRotationManager {
    store: Arc<dyn ApiKeyStore>,
    notifier: Arc<dyn RotationNotifier>,
}

impl KeyRotationManager {
    /// Initiate key rotation with schedule
    pub async fn schedule_rotation(
        &self,
        key_id: &str,
        rotation_schedule: RotationSchedule,
    ) -> Result<RotationPlan> {
        let current_key = self.store.get_key(key_id).await?;
        
        // Create rotation plan
        let plan = RotationPlan {
            key_id: key_id.to_string(),
            current_version: current_key.version,
            new_version: current_key.version + 1,
            announce_at: rotation_schedule.announce_at,
            activate_at: rotation_schedule.activate_at,
            deprecate_at: rotation_schedule.deprecate_at,
            expire_at: rotation_schedule.expire_at,
            notification_schedule: rotation_schedule.notifications,
        };
        
        // Store rotation plan
        self.store.save_rotation_plan(&plan).await?;
        
        // Schedule notifications
        self.schedule_notifications(&plan).await?;
        
        Ok(plan)
    }
    
    /// Execute rotation step
    pub async fn execute_rotation_step(
        &self,
        key_id: &str,
    ) -> Result<RotationStatus> {
        let plan = self.store.get_rotation_plan(key_id).await?;
        let now = Utc::now();
        
        match plan.current_phase(now) {
            RotationPhase::Announcement => {
                // Generate new key version
                let new_secret = generate_secure_secret(32);
                let new_version = self.create_key_version(
                    key_id,
                    &new_secret,
                    KeyVersionStatus::Pending,
                ).await?;
                
                // Notify consumers
                self.notifier.send_announcement(
                    &plan,
                    &new_version,
                ).await?;
                
                Ok(RotationStatus::Announced { new_version })
            }
            
            RotationPhase::Activation => {
                // Activate new version
                self.store.activate_key_version(
                    key_id,
                    plan.new_version,
                ).await?;
                
                // Mark old version as deprecated
                self.store.deprecate_key_version(
                    key_id,
                    plan.current_version,
                ).await?;
                
                // Notify activation
                self.notifier.send_activation(&plan).await?;
                
                Ok(RotationStatus::Activated)
            }
            
            RotationPhase::Deprecation => {
                // Send deprecation warnings
                self.notifier.send_deprecation_warning(&plan).await?;
                Ok(RotationStatus::Deprecated)
            }
            
            RotationPhase::Expiration => {
                // Expire old version
                self.store.expire_key_version(
                    key_id,
                    plan.current_version,
                ).await?;
                
                // Final notification
                self.notifier.send_expiration(&plan).await?;
                
                Ok(RotationStatus::Completed)
            }
        }
    }
}
```

### 3. Notification System
Implement rotation notifications:
```rust
#[async_trait]
pub trait RotationNotifier: Send + Sync {
    /// Send rotation announcement
    async fn send_announcement(
        &self,
        plan: &RotationPlan,
        new_version: &ApiKeyVersion,
    ) -> Result<()>;
    
    /// Send activation notice
    async fn send_activation(&self, plan: &RotationPlan) -> Result<()>;
    
    /// Send deprecation warning
    async fn send_deprecation_warning(&self, plan: &RotationPlan) -> Result<()>;
    
    /// Send expiration notice
    async fn send_expiration(&self, plan: &RotationPlan) -> Result<()>;
}

/// Email notification implementation
pub struct EmailNotifier {
    smtp_client: SmtpClient,
    template_engine: TemplateEngine,
}

impl EmailNotifier {
    async fn send_notification(
        &self,
        recipients: &[String],
        subject: &str,
        template: &str,
        context: &Context,
    ) -> Result<()> {
        let body = self.template_engine.render(template, context)?;
        
        for recipient in recipients {
            let message = Message::builder()
                .from("noreply@engram.io".parse()?)
                .to(recipient.parse()?)
                .subject(subject)
                .body(body.clone())?;
                
            self.smtp_client.send(message).await?;
        }
        
        Ok(())
    }
}
```

### 4. Validation Updates
Update key validation to handle versions:
```rust
impl ApiKeyValidator {
    pub async fn validate_with_rotation(
        &self,
        auth_header: &str,
    ) -> Result<(AuthContext, KeyVersion), AuthError> {
        let parsed = parse_api_key(auth_header)?;
        
        // Try to find valid version
        let versions = self.store
            .get_key_versions(&parsed.key_id)
            .await?;
        
        for version in versions {
            // Skip invalid statuses
            if !matches!(
                version.status,
                KeyVersionStatus::Active | KeyVersionStatus::Deprecated
            ) {
                continue;
            }
            
            // Try to validate against this version
            if let Ok(_) = verify_secret_hash(
                &parsed.secret,
                &version.secret_hash,
            ) {
                // Check if deprecated
                if version.status == KeyVersionStatus::Deprecated {
                    warn!(
                        "Deprecated API key version {} used for key {}",
                        version.version,
                        parsed.key_id
                    );
                }
                
                // Build context
                let context = self.build_auth_context(&parsed.key_id).await?;
                return Ok((context, version.version));
            }
        }
        
        Err(AuthError::InvalidApiKey)
    }
}
```

### 5. CLI Rotation Commands
Extend CLI with rotation commands:
```rust
#[derive(Parser)]
pub enum RotationAction {
    /// Schedule a key rotation
    Schedule {
        /// Key ID to rotate
        key_id: String,
        
        /// When to announce (e.g., "7d" from now)
        #[arg(long)]
        announce_in: String,
        
        /// When to activate new key
        #[arg(long)]
        activate_in: String,
        
        /// Grace period for old key
        #[arg(long)]
        grace_period: String,
        
        /// Email addresses to notify
        #[arg(long, value_delimiter = ',')]
        notify: Vec<String>,
    },
    
    /// Show rotation status
    Status {
        /// Key ID
        key_id: String,
    },
    
    /// Cancel scheduled rotation
    Cancel {
        /// Key ID
        key_id: String,
        
        /// Force cancellation
        #[arg(long)]
        force: bool,
    },
    
    /// Complete rotation immediately
    Complete {
        /// Key ID
        key_id: String,
    },
}
```

### 6. Rotation Monitoring
Add metrics for rotation:
```rust
pub struct RotationMetrics {
    /// Active rotations by phase
    pub active_rotations: GaugeVec,
    
    /// Rotation duration histogram
    pub rotation_duration: HistogramVec,
    
    /// Failed rotations
    pub rotation_failures: CounterVec,
    
    /// Keys using deprecated versions
    pub deprecated_key_usage: GaugeVec,
}
```

### 7. Automated Rotation Policy
Implement policy-based rotation:
```rust
pub struct RotationPolicy {
    /// Maximum key age before rotation
    pub max_age: Duration,
    
    /// Rotation schedule template
    pub schedule_template: RotationScheduleTemplate,
    
    /// Auto-rotate enabled
    pub auto_rotate: bool,
    
    /// Notification settings
    pub notifications: NotificationSettings,
}

pub async fn enforce_rotation_policy(
    policy: &RotationPolicy,
    store: &dyn ApiKeyStore,
    manager: &KeyRotationManager,
) -> Result<Vec<RotationPlan>> {
    let keys = store.list_keys().await?;
    let mut plans = Vec::new();
    
    for key in keys {
        if should_rotate(&key, policy) {
            let schedule = policy.schedule_template.create_schedule(
                &key,
                Utc::now(),
            );
            
            let plan = manager.schedule_rotation(
                &key.key_id,
                schedule,
            ).await?;
            
            plans.push(plan);
        }
    }
    
    Ok(plans)
}
```

## Testing Requirements
1. Test rotation state transitions
2. Test multi-version validation
3. Test notification delivery
4. Test grace period handling
5. Test automated rotation
6. Test rollback scenarios
7. Integration test full rotation

## Documentation
- Rotation best practices guide
- Email template examples
- Timeline visualization
- Troubleshooting guide

## Acceptance Criteria
1. Zero-downtime rotation works
2. Multiple versions validated correctly
3. Notifications sent on schedule
4. Grace periods enforced
5. Automated rotation by policy
6. Metrics track rotation progress
7. CLI provides full rotation control