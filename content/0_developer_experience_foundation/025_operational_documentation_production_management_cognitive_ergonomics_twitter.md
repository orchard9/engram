# Operational Documentation and Production Management Cognitive Ergonomics Twitter Thread

**Tweet 1/16**
It's 2am. The system is down. You open the ops docs and find... 47 pages of architecture theory.

Research: 90% of operators don't read docs until something breaks (Rettig 1991)

Yet most ops docs are written like textbooks, not emergency procedures 🧵

**Tweet 2/16**
The cognitive collapse during incidents:

Normal state:
- Working memory: 7±2 items
- Analytical reasoning: 100%
- Complex problem solving: Available

Stress state:
- Working memory: 3-4 items  
- Analytical reasoning: 30%
- Complex problem solving: Impossible

**Tweet 3/16**
What fails under stress:
❌ Complex architecture explanations
❌ Multi-step procedures requiring analysis
❌ Abstract troubleshooting guidance
❌ Reference material assuming calm reading

What survives:
✅ Copy-paste command checklists
✅ Visual flowcharts
✅ Pattern-matching procedures

**Tweet 4/16**
Traditional ops doc (requires high cognitive function):

"The spreading activation algorithm performance depends on several factors including graph connectivity, threshold settings, and system load. Consider investigating..."

**Tweet 5/16**
Stress-optimized ops doc (works at 2am):

"🚨 HIGH LATENCY - IMMEDIATE ACTIONS
1. Check threshold: engram config get confidence_threshold  
2. If <0.4: engram config set confidence_threshold 0.5
3. Verify: engram metrics latency --watch 60s
SUCCESS: P95 <200ms"

**Tweet 6/16**
The Context-Action-Verification framework improves task completion by 67% under stress (Carroll 1990):

CONTEXT: When is this the right action?
ACTION: Exact commands to execute
VERIFICATION: How do you know it worked?

Maps to human emergency response patterns

**Tweet 7/16**
Context prevents the most common ops error: right action, wrong time

"CONTEXT: Memory usage >90%, consolidation backlog >1000
DO NOT RUN IF: Active spreading activation in progress"

Situational awareness before decisive action

**Tweet 8/16**
Actions must be literally copy-pasteable with no interpretation:

```bash
# Emergency consolidation - execute in sequence
engram maintenance pause --services "background"
engram consolidation emergency --batch-size 100 --verbose  
watch -n 30 'engram consolidation status'
```

Zero cognitive translation required

**Tweet 9/16**
Verification needs self-assessment criteria operators can evaluate under pressure:

✓ Memory usage dropped >20%
✓ Queue <100 pending operations  
✓ No errors in consolidation logs

Clear success/failure indicators, no expert interpretation needed

**Tweet 10/16**
Progressive disclosure reduces cognitive load by 34% (Krug 2000):

Level 1: 🚨 EMERGENCIES (2-5 min fixes)
Level 2: 📋 COMMON OPS (5-15 min procedures)  
Level 3: 🔍 ADVANCED (30+ min diagnostics)

Find what you need without information overload

**Tweet 11/16**
Memory systems need probabilistic decision trees:

"Spreading activation timeouts"
→ "All queries affected?" 85% confidence threshold issue
→ "Specific patterns?" 70% query optimization needed
→ "Random timeouts?" 60% resource constraints

Multiple valid explanations

**Tweet 12/16**
Executable documentation reduces config errors by 71% (Spinellis 2003):

Write scripts that ARE the documentation
- Self-validating procedures
- Built-in success checks  
- Automatic error handling
- Cannot become outdated

**Tweet 13/16**
Visual mental models for dynamic systems:

Text can't capture spreading activation flow or consolidation progress

Diagrams enable pattern recognition under stress when text comprehension fails

External cognitive aids for complex state tracking

**Tweet 14/16**
Traditional database playbooks don't work for memory systems:

- Probabilistic behaviors vs deterministic failures
- Confidence drift vs data corruption  
- Spreading timeouts vs query optimization
- Consolidation effects vs maintenance windows

Need specialized incident patterns

**Tweet 15/16**
Post-incident documentation improvement loop:

After every incident:
✓ What worked in current docs?
✗ Where did docs fail to help?  
📝 What new knowledge was discovered?
🔄 How should procedures change?

Continuous evolution based on reality

**Tweet 16/16**
The 2am Documentation Test:

Every operational procedure must be executable by tired operators under production stress

If it requires calm study and deep thinking, it will fail when you need it most

Design for stress, not comfort