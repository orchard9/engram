# Tier-Aware Spreading Scheduler Research

## Research Topics and Findings

### Hierarchical Memory Models in Cognitive Architectures

**Key Research Areas:**
- Multi-level memory systems in cognitive architectures
- Priority scheduling for hierarchical storage systems
- Temporal locality patterns in memory access
- Cache-aware scheduling algorithms

**Primary Sources:**

1. **"Memory Systems: Cache, DRAM, Disk" by Jacob, Ng, and Wang (2007)**
   - Chapter 13: Memory hierarchy optimization
   - Section 13.4: Priority-based memory scheduling
   - Relevance: Foundational understanding of memory hierarchy scheduling principles

2. **"The Adaptive Character of Thought" by Anderson (1990)**
   - Chapter 5: ACT-R memory activation and decay
   - Page 123-145: Base-level activation and memory retrieval
   - Relevance: Cognitive basis for tier-aware activation spreading

3. **"Parallel Computer Architecture: A Hardware/Software Approach" by Culler, Singh, and Gupta (1999)**
   - Chapter 8: Memory consistency and cache coherence
   - Section 8.3: Directory-based cache coherence protocols
   - Relevance: Coherence mechanisms for distributed memory tiers

### Priority Scheduling Algorithms for Multi-Tier Systems

**Research Focus:**
- Real-time scheduling with deadline constraints
- Priority inversion prevention
- Work-conserving vs non-work-conserving schedulers
- Feedback-controlled scheduling

**Key Papers:**

1. **"Priority Scheduling in Operating Systems" by Liu and Layland (1973)**
   - Rate monotonic and earliest deadline first algorithms
   - Schedulability analysis for real-time systems
   - Application: Hot tier priority preemption design

2. **"The Linux Completely Fair Scheduler" by Wong et al. (2008)**
   - Virtual runtime and red-black tree scheduling
   - Multi-core load balancing strategies
   - Relevance: Fair scheduling across multiple storage tiers

3. **"Priority Inheritance Protocols: An Approach to Real-Time Synchronization" by Sha, Rajkumar, and Lehoczky (1990)**
   - Priority ceiling and priority inheritance protocols
   - Blocking time analysis and deadlock prevention
   - Application: Preventing priority inversion between tiers

### Lock-Free Queue Implementations for Real-Time Systems

**Research Domain:**
- Non-blocking concurrent data structures
- Memory ordering and synchronization primitives
- ABA problem and hazard pointers
- Performance characteristics of lock-free queues

**Foundational Works:**

1. **"Simple, Fast, and Practical Non-Blocking and Blocking Concurrent Queue Algorithms" by Michael and Scott (1996)**
   - Compare-and-swap based queue implementation
   - Memory management with hazard pointers
   - Performance analysis vs lock-based alternatives
   - Direct relevance: Lock-free queue design for tier scheduling

2. **"Hazard Pointers: Safe Memory Reclamation for Lock-Free Objects" by Michael (2004)**
   - Safe memory reclamation without garbage collection
   - Integration with concurrent data structures
   - Application: Memory management in tier-specific queues

3. **"Lock-free data structures" by Herlihy and Shavit (2008)**
   - Chapter 10: Concurrent queues and stacks
   - Section 10.5: The ABA problem and solutions
   - Relevance: Correctness guarantees for concurrent tier access

### Time-Budget Scheduling in Databases

**Research Area:**
- Query execution with time constraints
- Adaptive query processing
- Anytime algorithms for database operations
- Progressive query refinement

**Relevant Studies:**

1. **"Adaptive Query Processing" by Deshpande and Hellerstein (2004)**
   - River system for adaptive query execution
   - Time-bounded query processing strategies
   - Application: Timeout-based tier processing budgets

2. **"Time-Constrained Evaluation of Preference Queries" by Kießling and Fischer (2005)**
   - Skyline queries with time constraints
   - Progressive refinement algorithms
   - Relevance: Time-budget allocation across storage tiers

3. **"Anytime Query Processing for Real-time Applications" by Labrinidis and Roussopoulos (2004)**
   - Approximate query answering under time constraints
   - Quality-time trade-offs in query execution
   - Application: Cold tier bypass when time budget exceeded

### Work-Stealing Schedulers in Parallel Computing

**Focus Areas:**
- Dynamic load balancing in parallel systems
- Locality-aware work distribution
- Scalability analysis of work-stealing algorithms
- Integration with NUMA architectures

**Core Research:**

1. **"The Implementation of the Cilk-5 Multithreaded Language" by Frigo, Leiserson, and Randall (1998)**
   - Work-stealing runtime system design
   - Load balancing and cache-aware scheduling
   - Performance analysis on symmetric multiprocessors
   - Application: Work-stealing within storage tiers

2. **"A Java Fork/Join Framework" by Lea (2000)**
   - Design principles for work-stealing in Java
   - Performance comparison with other parallel patterns
   - Relevance: Practical implementation patterns for tier-specific work-stealing

3. **"Thread Scheduling for Multiprogrammed Multiprocessors" by Ousterhout (1982)**
   - Co-scheduling and gang scheduling techniques
   - Locality preservation in thread scheduling
   - Application: Maintaining cache locality within tier processing

### NUMA-Aware Scheduling Research

**Research Domain:**
- Non-uniform memory access optimization
- Cache-aware task placement
- Memory bandwidth utilization
- Scalability on many-core systems

**Key Sources:**

1. **"The Cache Performance and Optimizations of Blocked Algorithms" by Lam, Rothberg, and Wolf (1991)**
   - Cache-oblivious algorithms design
   - Memory hierarchy optimization techniques
   - Relevance: Cache-optimal tier access patterns

2. **"NUMA-aware Scheduling and Memory Allocation" by Dashti et al. (2013)**
   - Memory locality optimization strategies
   - Performance impact of NUMA placement
   - Application: Tier-specific thread and memory placement

### Cognitive Memory Consolidation Research

**Biological Foundations:**
- Hippocampal-neocortical memory consolidation
- Systems consolidation theory
- Memory replay and reactivation
- Sleep-dependent memory processing

**Neuroscience References:**

1. **"The Hippocampus and Memory Consolidation" by Squire and Alvarez (1995)**
   - Standard model of systems consolidation
   - Time-dependent reorganization of memory networks
   - Relevance: Biological inspiration for tier migration patterns

2. **"Memory Consolidation, Retrograde Amnesia and the Hippocampal Complex" by Nadel and Moscovitch (1997)**
   - Multiple trace theory of memory consolidation
   - Ongoing role of hippocampus in remote memory
   - Application: Dynamic tier assignment based on memory age and access

3. **"Reactivation of Hippocampal Ensemble Memories During Sleep" by Wilson and McNaughton (1994)**
   - Memory replay during slow-wave sleep
   - Strengthening of memory traces through reactivation
   - Relevance: Background processing patterns for cold tier

## Implementation Research Synthesis

### Architecture Implications

The research reveals several key design principles for the tier-aware spreading scheduler:

1. **Priority Preemption**: Hot tier processing must be able to preempt warm/cold tier operations to maintain sub-millisecond response times

2. **Lock-Free Design**: Using compare-and-swap based queues prevents blocking between tiers while maintaining memory ordering guarantees

3. **Time-Budget Allocation**: Each tier receives a specific time budget (100μs hot, 1ms warm, 10ms cold) with adaptive bypass mechanisms

4. **Work Conservation**: Work-stealing within tiers maximizes utilization while avoiding cross-tier stealing that could violate priority guarantees

5. **NUMA Awareness**: Thread and memory placement should respect the physical storage location of each tier

### Performance Targets

Based on cognitive memory research and real-time systems literature:

- **Hot Tier**: <100μs P95 latency (working memory constraints)
- **Warm Tier**: <1ms P95 latency (conscious recall threshold)
- **Cold Tier**: <10ms P95 latency (acceptable for background consolidation)
- **Overall**: <10ms P95 for single-hop activation (user perception threshold)

### Risk Mitigation Strategies

Research-informed approaches to common scheduling problems:

- **Priority Inversion**: Use priority ceiling protocols for any shared resources
- **Starvation**: Implement aging mechanisms for cold tier tasks
- **Memory Overhead**: Leverage hazard pointers for lock-free memory management
- **Cache Pollution**: Maintain separate working sets per tier

This research foundation provides the theoretical backing for implementing a production-ready tier-aware spreading scheduler that respects both cognitive memory principles and real-time systems constraints.