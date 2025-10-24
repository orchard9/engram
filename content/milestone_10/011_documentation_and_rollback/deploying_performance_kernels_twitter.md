# Documentation and Rollback - Twitter Thread

**Tweet 1/6:**

Documentation is insurance. You write it hoping you'll never need it.

But when Zig kernels break production at 3 AM, that documented rollback procedure is the difference between 5-minute recovery and 2-hour debugging.

We learned this the hard way.

**Tweet 2/6:**

Deployment checklist we wish we had on day 1:

- [ ] Zig installed on all nodes
- [ ] Tests pass
- [ ] Arena size configured
- [ ] Monitoring added
- [ ] Rollback tested
- [ ] Gradual rollout plan

Each checkbox prevents a class of incidents.

**Tweet 3/6:**

Emergency rollback (RTO: 5 minutes):

```bash
systemctl stop engram
cargo build --release  # Omit --features zig-kernels
cp target/release/engram /usr/local/bin/
systemctl start engram
```

No database migrations. No state cleanup. Just rebuild and restart.

**Tweet 4/6:**

When to rollback?

Critical (correctness): Immediate
High (performance): Within 1 hour
Medium (errors <1%): Investigate
Low (warnings): Monitor

Document the decision matrix before production.

**Tweet 5/6:**

Test your rollback procedures monthly:

Deploy to staging → Wait 5 minutes → Rollback → Verify health

Time each step. Update docs if RTO exceeds target.

Don't wait for production to test emergency procedures.

**Tweet 6/6:**

Three docs we wrote:

1. Operations: How to deploy
2. Rollback: How to recover
3. Architecture: How it works

Writing forces you to think through edge cases.

Reviewing catches mistakes.

Deploy fast, rollback faster.
