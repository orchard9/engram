# Example 06: Authentication

**Learning Goal**: Secure API access with API keys, JWT tokens, and memory space scoping.

**Difficulty**: Intermediate
**Time**: 15 minutes
**Prerequisites**: Completed Example 01

## Cognitive Concept

Authentication in Engram provides:
1. **Identity**: Who is accessing memories?
2. **Authorization**: What memory spaces can they access?
3. **Isolation**: Prevent cross-tenant memory leakage

## What You'll Learn

- Generate API keys with `engram auth create-key`
- Use JWT tokens for multi-tenant isolation
- Scope operations to specific memory spaces
- Handle authentication errors (ERR-4004)
- Implement token refresh logic

## Authentication Methods

### Method 1: API Keys (Simple)
Best for: Single-tenant, development, testing

```bash
engram auth create-key --name "my-app"
# Returns: ek_live_1234567890abcdef
```

### Method 2: JWT Tokens (Multi-tenant)
Best for: Production, multi-tenant SaaS, fine-grained permissions

```python
token = jwt.encode({
    "sub": "user_123",
    "memory_space_id": "tenant_acme",
    "permissions": ["read", "write"]
}, secret)
```

## Code Examples

See language-specific implementations in this directory:

- `rust.rs` - Rust with Bearer token headers
- `python.py` - Python with JWT creation/verification
- `typescript.ts` - TypeScript with token refresh
- `go.go` - Go with middleware integration
- `java.java` - Java with Spring Security

## Security Best Practices

1. **Never commit API keys** - use environment variables
2. **Rotate keys regularly** - 90-day maximum
3. **Scope to minimum permissions** - read-only when possible
4. **Use HTTPS in production** - never send keys over plain HTTP
5. **Implement token refresh** - before expiration

## Multi-Tenant Isolation

Each API key/token scoped to memory space:

```python
# Explicit space
client.remember(memory, memory_space_id="tenant_acme")

# Implicit from token
token_claims = {"memory_space_id": "tenant_acme"}
# All operations automatically scoped
```

## Next Steps

- [Security Guide](/operations/security.md) - Production security
- [08-multi-tenant](../08-multi-tenant/) - Multi-tenant patterns
