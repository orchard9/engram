# Security Operations Guide

This guide covers security configuration, operations, and best practices for running Engram in production environments.

## Table of Contents

1. [Overview](#overview)
2. [TLS/mTLS Configuration](#tlsmtls-configuration)
3. [Authentication](#authentication)
4. [Authorization](#authorization)
5. [Secrets Management](#secrets-management)
6. [Audit Logging](#audit-logging)
7. [Security Hardening](#security-hardening)
8. [Monitoring](#monitoring)
9. [Incident Response](#incident-response)
10. [Compliance](#compliance)

## Overview

Engram implements defense-in-depth security with multiple layers:

- **Transport Security**: TLS 1.3+ for all communication
- **Authentication**: Multi-factor (API keys, JWT, OAuth2/OIDC)
- **Authorization**: Fine-grained access control per memory space
- **Secrets Management**: Integration with Vault, AWS Secrets Manager, K8s Secrets
- **Audit Logging**: Complete security event trail
- **Hardening**: System and application-level security measures

## TLS/mTLS Configuration

### Generating Certificates

For development and testing, use the provided script:

```bash
cd deployments/tls
./generate_certs.sh
```

This generates:
- `ca.crt`, `ca.key` - Certificate Authority
- `server.crt`, `server.key` - Server certificate
- `client.crt`, `client.key` - Client certificate (for mTLS)

### Production Certificates

For production, obtain certificates from a trusted CA:

1. **Let's Encrypt** (free, automated):
   ```bash
   certbot certonly --standalone -d engram.example.com
   ```

2. **Commercial CA** (DigiCert, GlobalSign, etc.):
   - Generate CSR
   - Submit to CA
   - Install signed certificate

### TLS Configuration

Configure TLS in `engram.toml`:

```toml
[security.tls]
enabled = true
cert_chain_path = "/path/to/server.crt"
private_key_path = "/path/to/server.key"
min_protocol_version = "TLS13"

# Optional: Enable mTLS for gRPC
[security.mtls]
enabled = true
ca_bundle_path = "/path/to/ca.crt"
client_cert_required = true
```

### Certificate Rotation

Rotate certificates without downtime:

1. Generate new certificates
2. Update configuration files
3. Send SIGHUP to Engram process:
   ```bash
   kill -HUP $(pgrep engram)
   ```

## Authentication

### API Key Authentication

#### Generating API Keys

```bash
engram api-key generate \
  --name "production-client" \
  --spaces "space1,space2" \
  --permissions "memory:read,memory:write" \
  --expires-in "365d"
```

Output:
```
API Key ID: engram_key_abc123def456
Secret: xyz789...
Full Key: engram_key_abc123def456_xyz789...

IMPORTANT: Store this key securely. It will not be shown again.
```

#### Using API Keys

Include in Authorization header:

```bash
curl -H "Authorization: Bearer engram_key_abc123def456_xyz789..." \
  https://engram.example.com/api/recall
```

#### Key Management

List keys:
```bash
engram api-key list
```

Revoke key:
```bash
engram api-key revoke engram_key_abc123def456
```

### JWT Token Authentication

#### Configuration

Configure JWT validation in `engram.toml`:

```toml
[security.jwt]
enabled = true
issuer = "https://auth.example.com"
audience = "engram-api"
jwks_uri = "https://auth.example.com/.well-known/jwks.json"
jwks_refresh_interval = "1h"
```

#### Token Format

JWT claims required:
```json
{
  "sub": "user@example.com",
  "iat": 1234567890,
  "exp": 1234571490,
  "spaces": ["space1", "space2"],
  "perms": ["memory:read", "memory:write"]
}
```

#### Using JWT Tokens

```bash
curl -H "Authorization: Bearer eyJhbGc..." \
  https://engram.example.com/api/recall
```

### OAuth2/OIDC Integration

#### Configuration

```toml
[security.oauth2]
enabled = true
issuer = "https://auth.example.com"
client_id = "engram-prod"
client_secret_key = "oauth2/client-secret"  # Reference to secrets manager
redirect_uri = "https://engram.example.com/oauth/callback"
scopes = ["openid", "profile", "engram"]

# Claim mappings
[security.oauth2.claims]
spaces_claim = "engram_spaces"
permissions_claim = "engram_perms"
```

## Authorization

### Memory Space Access Control

Access control is enforced per memory space:

```toml
[security.authorization]
# Default deny - only explicitly allowed operations permitted
default_policy = "deny"

# Wildcard space access (admin only)
allow_wildcard = false
```

### Permissions

Available permissions:

- `memory:read` - Read memory operations
- `memory:write` - Write memory operations
- `memory:delete` - Delete memory operations
- `space:create` - Create memory spaces
- `space:delete` - Delete memory spaces
- `space:list` - List memory spaces
- `consolidation:trigger` - Trigger consolidation
- `consolidation:monitor` - Monitor consolidation
- `system:introspect` - System introspection
- `system:metrics` - View system metrics
- `system:health` - Check system health
- `admin:all` - Full administrative access

### Rate Limiting

Configure per-principal rate limits:

```toml
[security.rate_limiting]
enabled = true
default_requests_per_second = 100
default_burst_size = 200

# Per-operation limits
[security.rate_limiting.operations]
recall = 1000
remember = 500
consolidate = 10
```

## Secrets Management

### HashiCorp Vault

#### Configuration

```toml
[security.secrets.vault]
enabled = true
address = "https://vault.example.com:8200"
role_id = "engram-role"
secret_id_path = "/etc/engram/vault-secret-id"
mount_path = "secret"
```

#### Storing Secrets

```bash
vault kv put secret/engram/db-password value="supersecret"
vault kv put secret/engram/oauth-client-secret value="oauth-secret"
```

#### Using Secrets in Configuration

Reference secrets by key:

```toml
database_password = "vault:db-password"
oauth_client_secret = "vault:oauth-client-secret"
```

### AWS Secrets Manager

#### Configuration

```toml
[security.secrets.aws]
enabled = true
region = "us-west-2"
# AWS credentials from IAM role or environment
```

#### Storing Secrets

```bash
aws secretsmanager create-secret \
  --name engram/prod/db-password \
  --secret-string "supersecret"
```

#### Using Secrets

```toml
database_password = "aws:engram/prod/db-password"
```

### Kubernetes Secrets

#### Creating Secrets

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: engram-auth-secrets
  namespace: engram
type: Opaque
data:
  api-key-salt: <base64-encoded>
  jwt-public-key: <base64-encoded>
```

#### Mounting in Pods

```yaml
volumeMounts:
  - name: auth-secrets
    mountPath: /etc/engram/secrets
    readOnly: true
volumes:
  - name: auth-secrets
    secret:
      secretName: engram-auth-secrets
```

## Audit Logging

### Configuration

```toml
[security.audit]
enabled = true
format = "json"
output = "/var/log/engram/audit.log"
buffer_size = 1000
flush_interval = "10s"

# Log levels
[security.audit.events]
authentication = "info"
authorization = "info"
data_access = "info"
data_modification = "warn"
configuration = "warn"
system_access = "info"
```

### Audit Event Format

```json
{
  "timestamp": "2025-10-27T12:34:56Z",
  "event_type": "Authentication",
  "principal": "user@example.com",
  "resource": "memory_space_1",
  "operation": "recall",
  "result": "Success",
  "correlation_id": "abc-123-def-456",
  "source_ip": "192.168.1.100",
  "metadata": {}
}
```

### Forwarding to SIEM

Forward audit logs to centralized logging:

```bash
# Filebeat
filebeat -c filebeat.yml

# Fluentd
fluentd -c fluentd.conf

# Logstash
logstash -f logstash.conf
```

## Security Hardening

### Run Hardening Script

```bash
sudo ./scripts/security_hardening.sh
```

This applies:
- OS-level hardening (kernel parameters, disabled services)
- Application-level hardening (file permissions, user isolation)
- Container security recommendations
- Security scanning

### Manual Hardening Checklist

- [ ] Non-root user for Engram process
- [ ] Minimal file system permissions (750 for dirs, 640 for configs)
- [ ] SELinux/AppArmor policies enforced
- [ ] Unnecessary services disabled
- [ ] Kernel hardening parameters applied
- [ ] Firewall rules configured
- [ ] Resource limits set
- [ ] Secure defaults in configuration
- [ ] Regular security updates
- [ ] Vulnerability scanning enabled

## Monitoring

### Security Metrics

Monitor these security metrics:

- Authentication success/failure rate
- Authorization denials
- Rate limit hits
- Certificate expiration (30-day warning)
- Unusual access patterns
- Failed login attempts
- API key usage

### Alerting

Configure alerts for:

```yaml
alerts:
  - name: AuthenticationFailures
    condition: auth_failures > 100 per 5m
    severity: warning

  - name: AuthorizationDenials
    condition: authz_denials > 50 per 5m
    severity: warning

  - name: CertificateExpiring
    condition: cert_days_remaining < 30
    severity: critical

  - name: RateLimitExceeded
    condition: rate_limit_hits > 1000 per 1m
    severity: warning
```

## Incident Response

### Detection

1. Monitor security alerts
2. Review audit logs regularly
3. Track anomalous behavior
4. Investigate suspicious patterns

### Response Procedures

#### Compromised API Key

1. Revoke key immediately:
   ```bash
   engram api-key revoke <key-id>
   ```

2. Review audit logs for unauthorized access:
   ```bash
   grep <key-id> /var/log/engram/audit.log
   ```

3. Assess impact (data accessed/modified)
4. Generate new key if needed
5. Update client configuration
6. Document incident

#### Suspicious Access Patterns

1. Block source IP temporarily:
   ```bash
   iptables -A INPUT -s <ip-address> -j DROP
   ```

2. Review access logs
3. Verify legitimate usage
4. Escalate if confirmed breach
5. Update security rules

#### Certificate Compromise

1. Revoke compromised certificate
2. Generate new certificates
3. Rotate immediately
4. Update certificate revocation list
5. Monitor for misuse
6. Document incident

## Compliance

### SOC 2

Requirements addressed:
- Access control (authentication + authorization)
- Audit logging (complete event trail)
- Encryption in transit (TLS 1.3+)
- Change management (audit logs)
- Monitoring and alerting

### GDPR

Requirements addressed:
- Data access control (authorization engine)
- Audit trail (who accessed what, when)
- Right to be forgotten (deletion operations)
- Data minimization (permission-based access)
- Security by design (defense in depth)

### HIPAA

Requirements addressed:
- Access control (164.308(a)(4))
- Audit controls (164.312(b))
- Transmission security (164.312(e))
- Authentication (164.312(d))
- Authorization (164.308(a)(4))

### Compliance Validation

Generate compliance reports:

```bash
engram compliance report --standard soc2
engram compliance report --standard gdpr
engram compliance report --standard hipaa
```

## Best Practices

1. **Principle of Least Privilege**: Grant minimal permissions required
2. **Defense in Depth**: Multiple security layers
3. **Zero Trust**: Verify every request
4. **Audit Everything**: Complete logging
5. **Regular Updates**: Keep dependencies current
6. **Incident Response**: Have procedures ready
7. **Security Training**: Educate team members
8. **Regular Testing**: Penetration tests, audits
9. **Backup & Recovery**: Test disaster recovery
10. **Documentation**: Keep security docs current

## Support

For security issues:
- Email: security@engram.io
- PGP Key: [key fingerprint]
- Responsible disclosure policy: docs/SECURITY.md

For general questions:
- Documentation: https://docs.engram.io
- Community: https://community.engram.io
- Support: support@engram.io
