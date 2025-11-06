# Task: CLI Key Management Commands

## Objective
Add CLI commands for creating, listing, rotating, and revoking API keys, providing a complete key lifecycle management interface.

## Context
System administrators need CLI tools to manage API keys without direct database access. These commands will use the `ApiKeyValidator` and storage backend.

## Requirements

### 1. Add Auth Subcommand
Update `engram-cli/src/cli.rs`:
```rust
#[derive(Parser)]
pub enum Commands {
    // ... existing commands ...
    
    /// API key management commands
    Auth {
        #[command(subcommand)]
        action: AuthAction,
    },
}

#[derive(Parser)]
pub enum AuthAction {
    /// Create a new API key
    CreateKey {
        /// Human-readable name for the key
        #[arg(long)]
        name: String,
        
        /// Comma-separated list of allowed memory spaces
        #[arg(long, value_delimiter = ',')]
        spaces: Vec<String>,
        
        /// Comma-separated list of permissions
        #[arg(long, value_delimiter = ',')]
        permissions: Vec<String>,
        
        /// Expiration time (e.g., "90d", "6m", "1y")
        #[arg(long)]
        expires_in: Option<String>,
        
        /// Rate limit (requests per second)
        #[arg(long, default_value = "100")]
        rate_limit: u32,
        
        /// Output format
        #[arg(long, value_enum, default_value = "table")]
        format: OutputFormat,
    },
    
    /// List all API keys
    ListKeys {
        /// Show revoked keys
        #[arg(long)]
        include_revoked: bool,
        
        /// Show expired keys
        #[arg(long)]
        include_expired: bool,
        
        /// Output format
        #[arg(long, value_enum, default_value = "table")]
        format: OutputFormat,
    },
    
    /// Rotate an API key
    RotateKey {
        /// Key ID to rotate
        key_id: String,
        
        /// Grace period for old key (e.g., "7d")
        #[arg(long, default_value = "7d")]
        grace_period: String,
        
        /// Force immediate rotation without grace period
        #[arg(long)]
        force: bool,
    },
    
    /// Revoke an API key
    RevokeKey {
        /// Key ID to revoke
        key_id: String,
        
        /// Reason for revocation
        #[arg(long)]
        reason: Option<String>,
        
        /// Skip confirmation prompt
        #[arg(long)]
        yes: bool,
    },
    
    /// Check API key status
    CheckKey {
        /// Key ID or full key to check
        key: String,
        
        /// Show detailed information
        #[arg(long)]
        detailed: bool,
    },
    
    /// Show keys expiring soon
    ExpiringKeys {
        /// Days to look ahead
        #[arg(long, default_value = "30")]
        days: u32,
        
        /// Output format
        #[arg(long, value_enum, default_value = "table")]
        format: OutputFormat,
    },
}
```

### 2. Implement Create Key Command
```rust
pub async fn create_api_key(
    name: String,
    spaces: Vec<String>,
    permissions: Vec<String>,
    expires_in: Option<String>,
    rate_limit: u32,
    format: OutputFormat,
) -> Result<()> {
    // Parse permissions
    let permissions = permissions
        .into_iter()
        .map(|p| Permission::from_str(&p))
        .collect::<Result<Vec<_>, _>>()?;
    
    // Parse spaces
    let allowed_spaces = spaces
        .into_iter()
        .map(|s| MemorySpaceId::try_from(s.as_str()))
        .collect::<Result<Vec<_>, _>>()?;
    
    // Parse expiration
    let expires_at = expires_in
        .map(|exp| parse_duration(&exp))
        .transpose()?
        .map(|dur| Utc::now() + dur);
    
    // Create request
    let request = GenerateKeyRequest {
        name,
        allowed_spaces,
        rate_limit: RateLimit {
            requests_per_second: rate_limit,
            burst_size: rate_limit * 2,
        },
        expires_at,
        permissions,
    };
    
    // Generate key
    let validator = get_key_validator().await?;
    let response = validator.generate_key(request).await?;
    
    // Output based on format
    match format {
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&json!({
                "key_id": response.key_id,
                "api_key": response.full_key,
                "created": true,
                "expires_at": expires_at,
            }))?);
        }
        OutputFormat::Table => {
            println!("✅ API key created successfully!");
            println!();
            println!("Key ID: {}", response.key_id);
            println!("API Key: {}", response.full_key);
            println!();
            println!("⚠️  Save this key securely - it won't be shown again!");
            if let Some(exp) = expires_at {
                println!("Expires: {}", exp.format("%Y-%m-%d %H:%M:%S UTC"));
            }
        }
    }
    
    Ok(())
}
```

### 3. Implement List Keys Command
```rust
pub async fn list_api_keys(
    include_revoked: bool,
    include_expired: bool,
    format: OutputFormat,
) -> Result<()> {
    let store = get_key_store().await?;
    let keys = store.list_keys().await?;
    
    // Filter keys
    let keys: Vec<_> = keys
        .into_iter()
        .filter(|k| include_revoked || k.revoked_at.is_none())
        .filter(|k| include_expired || k.expires_at.map_or(true, |e| e > Utc::now()))
        .collect();
    
    match format {
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&keys)?);
        }
        OutputFormat::Table => {
            let mut table = Table::new();
            table.set_titles(row![
                "Key ID", "Name", "Spaces", "Created", "Expires", "Last Used", "Status"
            ]);
            
            for key in keys {
                let status = if key.revoked_at.is_some() {
                    "Revoked"
                } else if key.expires_at.map_or(false, |e| e < Utc::now()) {
                    "Expired"
                } else {
                    "Active"
                };
                
                table.add_row(row![
                    key.key_id,
                    key.name,
                    key.allowed_spaces.join(", "),
                    key.created_at.format("%Y-%m-%d"),
                    key.expires_at.map_or("Never".to_string(), |e| e.format("%Y-%m-%d").to_string()),
                    key.last_used.map_or("Never".to_string(), |l| format_relative_time(l)),
                    status,
                ]);
            }
            
            table.printstd();
        }
    }
    
    Ok(())
}
```

### 4. Implement Rotate Key Command
```rust
pub async fn rotate_api_key(
    key_id: String,
    grace_period: String,
    force: bool,
) -> Result<()> {
    let store = get_key_store().await?;
    
    // Get existing key
    let old_key = store.get_key(&key_id).await?;
    
    // Create new key with same permissions
    let new_request = GenerateKeyRequest {
        name: format!("{} (rotated)", old_key.metadata.name),
        allowed_spaces: old_key.allowed_spaces.clone(),
        rate_limit: old_key.rate_limit.clone(),
        expires_at: old_key.metadata.expires_at,
        permissions: old_key.metadata.permissions.clone(),
    };
    
    let validator = get_key_validator().await?;
    let new_key = validator.generate_key(new_request).await?;
    
    // Set expiration on old key (with grace period)
    if !force {
        let grace_duration = parse_duration(&grace_period)?;
        store.update_expiration(&key_id, Some(Utc::now() + grace_duration)).await?;
        
        println!("✅ Key rotation initiated");
        println!("Old key {} will expire in {}", key_id, grace_period);
        println!("New key ID: {}", new_key.key_id);
        println!("New API key: {}", new_key.full_key);
    } else {
        store.revoke_key(&key_id, "Force rotated").await?;
        println!("✅ Key rotated immediately");
        println!("Old key {} has been revoked", key_id);
        println!("New key ID: {}", new_key.key_id);
        println!("New API key: {}", new_key.full_key);
    }
    
    Ok(())
}
```

### 5. Add Utility Functions
```rust
/// Parse duration strings like "7d", "30m", "1y"
fn parse_duration(s: &str) -> Result<Duration> {
    let (num, unit) = s.split_at(s.len() - 1);
    let num: i64 = num.parse()?;
    
    match unit {
        "d" => Ok(Duration::days(num)),
        "w" => Ok(Duration::weeks(num)),
        "m" => Ok(Duration::days(num * 30)), // Approximate
        "y" => Ok(Duration::days(num * 365)), // Approximate
        _ => Err(anyhow!("Invalid duration unit: {}", unit)),
    }
}

/// Format relative time for display
fn format_relative_time(time: DateTime<Utc>) -> String {
    let now = Utc::now();
    let duration = now - time;
    
    if duration.num_seconds() < 60 {
        "Just now".to_string()
    } else if duration.num_minutes() < 60 {
        format!("{} min ago", duration.num_minutes())
    } else if duration.num_hours() < 24 {
        format!("{} hours ago", duration.num_hours())
    } else {
        format!("{} days ago", duration.num_days())
    }
}
```

## Testing Requirements
1. Test key creation with various parameters
2. Test listing with filters
3. Test rotation workflow
4. Test revocation
5. Test permission parsing
6. Test duration parsing
7. Integration tests with real storage

## Documentation
- Add examples to CLI help text
- Create man page entries
- Add to operations guide
- Include in quickstart

## Acceptance Criteria
1. All key management commands work correctly
2. Clear success/error messages
3. Proper validation of inputs
4. Support for JSON and table output
5. Confirmation prompts for destructive actions
6. Integration with existing CLI structure