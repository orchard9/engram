# Task: Security Documentation

## Objective
Create comprehensive security documentation covering authentication setup, best practices, operational procedures, and troubleshooting guides.

## Context
Users and operators need clear documentation to properly configure and manage Engram's security features. This includes quickstart guides, detailed references, and operational runbooks.

## Requirements

### 1. Security Overview Document
Create `docs/explanation/security-architecture.md`:
```markdown
# Engram Security Architecture

## Overview
Engram provides enterprise-grade security through API key authentication, 
fine-grained permissions, and rate limiting.

## Security Principles
- **Defense in depth**: Multiple security layers
- **Least privilege**: Minimal permissions by default
- **Secure by default**: Production configs require auth
- **Zero trust**: Verify every request
- **Audit everything**: Complete security event logging

## Architecture Components

### Authentication Layer
- API key validation with Argon2id hashing
- Bearer token format: `engram_key_{id}_{secret}`
- Cached validation for performance
- Support for multiple auth modes

### Authorization Layer
- Per-space access control
- Fine-grained permissions
- Role-based access patterns
- Dynamic permission checks

### Rate Limiting
- Per-key rate limits
- Global system limits
- Adaptive throttling
- Burst handling

### Security Monitoring
- Real-time metrics
- Anomaly detection
- Security alerts
- Audit logging
```

### 2. Quickstart Guide
Create `docs/tutorials/security-quickstart.md`:
```markdown
# Security Quickstart

Get started with Engram authentication in 5 minutes.

## 1. Enable Authentication

Edit your config file:
```toml
[security]
auth_mode = "api_key"
rate_limiting = true
```

## 2. Create Your First API Key

```bash
engram auth create-key \
  --name "my-app" \
  --spaces "default" \
  --permissions "MemoryRead,MemoryWrite"
```

Save the API key securely - it won't be shown again!

## 3. Test Authentication

```bash
# Set your API key
export ENGRAM_API_KEY="engram_key_abc123_secret456"

# Test with curl
curl -H "Authorization: Bearer $ENGRAM_API_KEY" \
  http://localhost:9090/api/v1/system/health
```

## 4. Configure Your Application

Python example:
```python
from engram import Client

client = Client(
    url="http://localhost:9090",
    api_key="engram_key_abc123_secret456"
)
```

## Next Steps
- Set up key rotation
- Configure rate limits
- Enable monitoring
```

### 3. API Key Management Guide
Create `docs/howto/manage-api-keys.md`:
```markdown
# Managing API Keys

## Creating Keys

### Basic Key Creation
```bash
engram auth create-key --name "production-app"
```

### With Specific Permissions
```bash
engram auth create-key \
  --name "read-only-app" \
  --permissions "MemoryRead,SystemHealth" \
  --spaces "analytics,metrics"
```

### Time-Limited Keys
```bash
engram auth create-key \
  --name "temp-access" \
  --expires-in "30d"
```

## Listing Keys

```bash
# Show all active keys
engram auth list-keys

# Include expired/revoked
engram auth list-keys --include-expired --include-revoked

# JSON output for automation
engram auth list-keys --format json
```

## Rotating Keys

### Standard Rotation (7-day grace period)
```bash
engram auth rotate-key abc123
```

### Custom Grace Period
```bash
engram auth rotate-key abc123 --grace-period "30d"
```

### Immediate Rotation (no grace period)
```bash
engram auth rotate-key abc123 --force
```

## Revoking Keys

```bash
# With confirmation prompt
engram auth revoke-key abc123 --reason "Compromised"

# Skip confirmation
engram auth revoke-key abc123 --yes
```

## Key Security Best Practices

1. **Never commit keys to version control**
2. **Use environment variables or secret management**
3. **Rotate keys regularly (every 90 days)**
4. **Use least privilege permissions**
5. **Monitor key usage patterns**
```

### 4. Operations Runbook
Create `docs/operations/security-runbook.md`:
```markdown
# Security Operations Runbook

## Incident Response

### Suspected Key Compromise

1. **Immediate Actions**
   ```bash
   # Revoke compromised key immediately
   engram auth revoke-key <key-id> --reason "Suspected compromise"
   
   # Check recent usage
   engram auth check-key <key-id> --detailed
   ```

2. **Investigation**
   - Review audit logs for unusual activity
   - Check rate limit violations
   - Identify affected memory spaces

3. **Remediation**
   - Create new key for legitimate user
   - Update application configurations
   - Monitor for continued suspicious activity

### High Authentication Failure Rate

1. **Identify Source**
   ```bash
   # Check metrics
   curl http://localhost:9090/metrics | grep auth_failures
   ```

2. **Mitigate**
   - Enable stricter rate limiting
   - Block suspicious IPs at firewall
   - Review recent configuration changes

### Key Rotation Procedures

#### Scheduled Rotation
1. Schedule rotation 30 days in advance
2. Notify all key users via email
3. Monitor deprecated key usage
4. Complete rotation after grace period

#### Emergency Rotation
1. Use force rotation for immediate effect
2. Notify users through multiple channels
3. Provide temporary elevated support

## Monitoring Setup

### Prometheus Queries
```yaml
# Auth success rate (target: >99.9%)
rate(engram_auth_attempts_total{result="success"}[5m]) / 
rate(engram_auth_attempts_total[5m])

# Deprecated key usage (target: 0)
sum(engram_deprecated_key_usage)

# Rate limit violations (investigate if >10/min)
rate(engram_rate_limit_hits_total[1m])
```

### Alert Configuration
See `config/prometheus/alerts/security.yml`

## Backup Procedures

### API Key Database Backup
```bash
# Backup key database
cp /var/lib/engram/api_keys.db \
   /backup/api_keys_$(date +%Y%m%d).db

# Verify backup
sqlite3 /backup/api_keys_*.db "SELECT COUNT(*) FROM api_keys;"
```

### Restore from Backup
```bash
# Stop Engram
systemctl stop engram

# Restore database
cp /backup/api_keys_20240115.db /var/lib/engram/api_keys.db

# Restart
systemctl start engram
```
```

### 5. Reference Documentation
Create `docs/reference/security-api.md`:
```markdown
# Security API Reference

## Configuration Schema

### Security Configuration
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| auth_mode | string | "none" | Authentication mode: "none", "api_key" |
| rate_limiting | bool | false | Enable rate limiting |
| api_keys.backend | string | "file" | Storage backend |
| api_keys.storage_path | string | "./data/api_keys.db" | Database path |

### Permission Types
| Permission | Description | Default |
|------------|-------------|---------|
| MemoryRead | Read memories | ❌ |
| MemoryWrite | Create/update memories | ❌ |
| MemoryDelete | Delete memories | ❌ |
| SpaceCreate | Create memory spaces | ❌ |
| SpaceDelete | Delete memory spaces | ❌ |
| SystemHealth | Access health endpoints | ✅ |
| SystemMetrics | Access metrics | ✅ |
| SystemShutdown | Shutdown server | ❌ |
| AdminKeyManagement | Manage API keys | ❌ |

## HTTP Headers

### Request Headers
- `Authorization: Bearer engram_key_{id}_{secret}` - API key auth
- `X-Memory-Space-Id: {space}` - Target memory space

### Response Headers
- `X-RateLimit-Limit` - Request limit per second
- `X-RateLimit-Remaining` - Remaining requests
- `X-RateLimit-Reset` - Reset timestamp

## Error Responses

### 401 Unauthorized
```json
{
  "error": {
    "code": 401,
    "message": "Invalid API key",
    "type": "authentication_error"
  }
}
```

### 403 Forbidden
```json
{
  "error": {
    "code": 403,
    "message": "Missing required permission: MemoryWrite",
    "type": "authorization_error"
  }
}
```

### 429 Too Many Requests
```json
{
  "error": {
    "code": 429,
    "message": "Rate limit exceeded",
    "type": "rate_limit_error",
    "retry_after": 30
  }
}
```
```

### 6. Migration Guide
Create `docs/howto/migrate-to-auth.md`:
```markdown
# Migrating to Authenticated Engram

## Overview
This guide helps you enable authentication in an existing Engram deployment.

## Pre-Migration Checklist
- [ ] Inventory all applications using Engram
- [ ] Document current access patterns
- [ ] Plan key distribution strategy
- [ ] Schedule migration window
- [ ] Prepare rollback plan

## Migration Steps

### 1. Enable Auth in Test Environment
```toml
# test.toml
[security]
auth_mode = "api_key"
```

### 2. Create API Keys for Applications
```bash
# Create keys for each application
engram auth create-key --name "webapp" --spaces "default"
engram auth create-key --name "analytics" --spaces "metrics"
```

### 3. Update Application Configurations
Gradually update each application with its API key.

### 4. Monitor Unauthenticated Access
Watch logs for requests without auth headers.

### 5. Enable Authentication in Production
Once all applications are updated, enable auth.

## Rollback Procedure
```toml
# Disable auth immediately if issues
[security]
auth_mode = "none"
```

## Common Issues

### "Missing Authorization header"
Application not sending API key. Check:
- Environment variables loaded
- Header format correct
- Network proxies not stripping headers

### "Invalid API key"
- Verify key not expired
- Check key copied completely
- Ensure proper Bearer format
```

## Testing
1. Test all documentation examples
2. Verify code snippets compile/run
3. Test procedures in staging
4. Review with security team
5. Get feedback from beta users

## Documentation Structure
```
docs/
├── explanation/
│   └── security-architecture.md
├── tutorials/
│   └── security-quickstart.md
├── howto/
│   ├── manage-api-keys.md
│   └── migrate-to-auth.md
├── operations/
│   └── security-runbook.md
└── reference/
    └── security-api.md
```

## Acceptance Criteria
1. All security features documented
2. Examples for common languages
3. Operational procedures tested
4. Migration guide validated
5. Troubleshooting covers common issues
6. Documentation follows Diátaxis framework