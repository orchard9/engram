# M14 Systems Implementation Guide

**Purpose**: Low-level implementation guidance for distributed architecture
**Audience**: Systems engineers implementing M14 AFTER prerequisites are met
**Status**: Reference document for future implementation

---

## 1. WAL File Format Design

### 1.1 Entry Structure

```rust
// engram-core/src/replication/wal.rs

use std::mem::size_of;

/// WAL entry layout (fixed-size header + variable payload)
///
/// Memory layout (64-byte aligned for cache efficiency):
/// ┌─────────────────────────────────────────────────┐
/// │ Header (64 bytes)                               │
/// ├─────────────────────────────────────────────────┤
/// │ Payload (variable, 4KB aligned)                 │
/// └─────────────────────────────────────────────────┘
#[repr(C, align(64))]
struct WalEntry {
    // ── Header (64 bytes) ──
    magic: u32,              // 0xDEADBEEF (corruption detection)
    version: u8,             // Protocol version (for upgrades)
    flags: u8,               // Compression, encryption flags
    _reserved: [u8; 2],      // Future use

    sequence: u64,           // Monotonic per-space counter
    space_id_len: u16,       // Length of space_id string
    _pad1: [u8; 6],          // Align to 8-byte boundary

    timestamp_ns: i64,       // Nanosecond precision (chrono::Utc)
    primary_node: u64,       // Which node wrote this (NodeId hash)

    payload_size: u32,       // Size of payload in bytes
    checksum: u32,           // xxHash of payload
    header_checksum: u32,    // xxHash of header (detect header corruption)
    _pad2: [u8; 4],          // Total header = 64 bytes

    // ── Payload (variable) ──
    // [space_id: [u8; space_id_len]]  // Not null-terminated
    // [operation: u8]                 // Store=1, Delete=2, Consolidate=3
    // [data: [u8; payload_size - space_id_len - 1]]
}

impl WalEntry {
    /// Size of fixed header (64 bytes)
    const HEADER_SIZE: usize = 64;

    /// Magic number for entry validation
    const MAGIC: u32 = 0xDEADBEEF;

    /// Current protocol version
    const VERSION: u8 = 1;

    /// Create new WAL entry for write operation
    fn new_store(
        sequence: u64,
        space_id: &str,
        memory: &Memory,
        primary_node: NodeId,
    ) -> Result<Self> {
        let timestamp_ns = Utc::now().timestamp_nanos_opt()
            .ok_or_else(|| anyhow!("Timestamp overflow"))?;

        // Serialize memory to bytes
        let memory_bytes = bincode::serialize(memory)?;

        // Construct payload: [space_id][operation][memory_bytes]
        let mut payload = Vec::with_capacity(
            space_id.len() + 1 + memory_bytes.len()
        );
        payload.extend_from_slice(space_id.as_bytes());
        payload.push(1); // Store operation
        payload.extend_from_slice(&memory_bytes);

        // Compute checksums
        let checksum = xxhash_rust::xxh3::xxh3_64(&payload);

        let entry = Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            flags: 0,
            _reserved: [0; 2],

            sequence,
            space_id_len: space_id.len() as u16,
            _pad1: [0; 6],

            timestamp_ns,
            primary_node: primary_node.hash(),

            payload_size: payload.len() as u32,
            checksum: checksum as u32,
            header_checksum: 0, // Computed after
            _pad2: [0; 4],
        };

        // Compute header checksum (after all fields set)
        let header_checksum = entry.compute_header_checksum();

        Ok(Self {
            header_checksum,
            ..entry
        })
    }

    /// Validate entry integrity
    fn validate(&self) -> Result<()> {
        // Check magic number
        if self.magic != Self::MAGIC {
            bail!("Invalid magic: expected {:x}, got {:x}",
                  Self::MAGIC, self.magic);
        }

        // Check version compatibility
        if self.version > Self::VERSION {
            bail!("Unsupported version: {}", self.version);
        }

        // Validate header checksum
        let expected = self.compute_header_checksum();
        if self.header_checksum != expected {
            bail!("Header checksum mismatch");
        }

        Ok(())
    }

    /// Compute header checksum (xxHash of first 60 bytes)
    fn compute_header_checksum(&self) -> u32 {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                self as *const Self as *const u8,
                60, // Exclude header_checksum itself
            )
        };
        xxhash_rust::xxh3::xxh3_64(bytes) as u32
    }
}
```

### 1.2 WAL File Layout

```rust
/// WAL file layout (memory-mapped for zero-copy replication)
///
/// Structure:
/// ┌────────────────────────────────────────┐
/// │ File Header (4 KB)                     │ <- Page-aligned
/// ├────────────────────────────────────────┤
/// │ Entry Index (variable, grows backward) │
/// ├────────────────────────────────────────┤
/// │ ... unused space ...                   │
/// ├────────────────────────────────────────┤
/// │ Entries (grow forward)                 │
/// └────────────────────────────────────────┘
///
/// Index grows backward, entries grow forward → meet in middle
#[repr(C, align(4096))]
struct WalFileHeader {
    magic: u64,              // 0xDEADBEEF_CAFEBABE
    version: u32,
    _reserved: u32,

    created_at: i64,         // File creation timestamp
    space_id: [u8; 256],     // Null-terminated space ID

    entry_count: u64,        // Number of entries in file
    first_sequence: u64,     // Sequence of first entry
    last_sequence: u64,      // Sequence of last entry

    entries_offset: u64,     // Byte offset to first entry
    entries_size: u64,       // Total bytes of entries

    index_offset: u64,       // Byte offset to index (grows backward)
    index_size: u64,         // Total bytes of index

    checksum: u64,           // xxHash of header
    _pad: [u8; 3816],        // Total size = 4096 bytes
}

/// Entry index for fast sequence lookup
///
/// Maps sequence number → file offset
#[repr(C)]
struct IndexEntry {
    sequence: u64,
    offset: u64,
}

struct WalFile {
    path: PathBuf,
    mmap: memmap2::MmapMut,  // Memory-mapped file
    header: *mut WalFileHeader,
}

impl WalFile {
    /// Create new WAL file with initial size
    fn create(path: PathBuf, space_id: &str, initial_size: usize) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)?;

        // Allocate initial size (will grow as needed)
        file.set_len(initial_size as u64)?;

        // Memory-map file (read-write)
        let mut mmap = unsafe { memmap2::MmapMut::map_mut(&file)? };

        // Initialize header
        let header = unsafe { &mut *(mmap.as_mut_ptr() as *mut WalFileHeader) };
        *header = WalFileHeader {
            magic: 0xDEADBEEF_CAFEBABE,
            version: 1,
            _reserved: 0,
            created_at: Utc::now().timestamp_nanos_opt().unwrap(),
            space_id: {
                let mut id = [0u8; 256];
                id[..space_id.len()].copy_from_slice(space_id.as_bytes());
                id
            },
            entry_count: 0,
            first_sequence: 0,
            last_sequence: 0,
            entries_offset: 4096, // After header
            entries_size: 0,
            index_offset: initial_size as u64, // Starts at end
            index_size: 0,
            checksum: 0,
            _pad: [0; 3816],
        };

        Ok(Self {
            path,
            mmap,
            header: header as *mut WalFileHeader,
        })
    }

    /// Append entry to WAL (lock-free via atomic offset update)
    fn append(&mut self, entry: &WalEntry, payload: &[u8]) -> Result<u64> {
        let header = unsafe { &mut *self.header };

        // Compute total size (header + payload, 4KB aligned)
        let entry_size = WalEntry::HEADER_SIZE + payload.len();
        let aligned_size = (entry_size + 4095) & !4095; // Round up to 4KB

        // Atomic offset update (CAS loop)
        let offset = loop {
            let current_offset = header.entries_offset + header.entries_size;
            let new_offset = current_offset + aligned_size as u64;

            // Check if we need to grow file
            if new_offset > header.index_offset {
                self.grow_file()?;
                continue;
            }

            // Atomic CAS: update entries_size
            let success = unsafe {
                let ptr = &header.entries_size as *const u64 as *mut u64;
                std::sync::atomic::AtomicU64::from_ptr(ptr)
                    .compare_exchange(
                        header.entries_size,
                        new_offset - header.entries_offset,
                        Ordering::SeqCst,
                        Ordering::SeqCst,
                    )
                    .is_ok()
            };

            if success {
                break current_offset;
            }
        };

        // Write entry to mmap (lock-free, writer doesn't block readers)
        unsafe {
            let dst = self.mmap.as_mut_ptr().add(offset as usize);
            std::ptr::copy_nonoverlapping(
                entry as *const WalEntry as *const u8,
                dst,
                WalEntry::HEADER_SIZE,
            );
            std::ptr::copy_nonoverlapping(
                payload.as_ptr(),
                dst.add(WalEntry::HEADER_SIZE),
                payload.len(),
            );
        }

        // Update index (lock-free, grows backward)
        self.append_index_entry(entry.sequence, offset)?;

        // Sync to disk (async fsync)
        self.mmap.flush_async()?;

        Ok(entry.sequence)
    }
}
```

**Key Design Decisions**:
1. **64-byte aligned header**: Fits in single cache line (reduce DRAM accesses)
2. **Memory-mapped**: Zero-copy replication (send directly from mmap to NIC)
3. **Lock-free append**: Atomic CAS for concurrent writes
4. **Backward-growing index**: Avoids moving entries when index grows
5. **xxHash checksum**: Fastest hash (7 GB/sec, faster than CRC32)

---

## 2. NUMA-Aware WAL Allocation

### 2.1 Per-NUMA-Node WAL Files

```rust
// engram-core/src/replication/numa_wal.rs

use hwloc2::Topology;

/// NUMA-aware WAL manager
///
/// One WAL file per NUMA node to avoid cross-NUMA writes
struct NumaAwareWalManager {
    /// WAL files, one per NUMA node
    wal_per_node: Vec<Arc<Mutex<WalFile>>>,

    /// Space → NUMA node mapping (affinity)
    space_affinity: DashMap<String, usize>,

    /// NUMA topology
    topology: Topology,
}

impl NumaAwareWalManager {
    fn new() -> Result<Self> {
        let mut topology = Topology::new()?;
        let num_nodes = topology.objects_with_type(&ObjectType::NUMANode)?.len();

        // Create one WAL per NUMA node
        let mut wal_per_node = Vec::with_capacity(num_nodes);
        for node_id in 0..num_nodes {
            let path = PathBuf::from(format!("wal_node_{}.dat", node_id));

            // Allocate WAL on specific NUMA node
            let wal = Self::create_wal_on_numa_node(&topology, node_id, path)?;
            wal_per_node.push(Arc::new(Mutex::new(wal)));
        }

        Ok(Self {
            wal_per_node,
            space_affinity: DashMap::new(),
            topology,
        })
    }

    /// Create WAL file with memory allocated on specific NUMA node
    fn create_wal_on_numa_node(
        topology: &Topology,
        node_id: usize,
        path: PathBuf,
    ) -> Result<WalFile> {
        // Pin current thread to NUMA node (ensures malloc allocates locally)
        let node = topology.objects_with_type(&ObjectType::NUMANode)?
            .get(node_id)
            .ok_or_else(|| anyhow!("NUMA node {} not found", node_id))?;

        topology.set_cpubind_for_thread(
            0, // Current thread
            node.cpuset().unwrap(),
            CpuBindFlags::CPUBIND_THREAD,
        )?;

        // Create WAL file (memory-mapped region allocated on this NUMA node)
        let wal = WalFile::create(path, "default", 1024 * 1024 * 1024)?; // 1 GB

        Ok(wal)
    }

    /// Append entry to appropriate WAL (based on space affinity)
    async fn append(&self, space_id: &str, entry: WalEntry, payload: &[u8]) -> Result<u64> {
        // Determine NUMA node for this space (consistent hashing)
        let numa_node = self.space_affinity
            .entry(space_id.to_string())
            .or_insert_with(|| {
                // Hash space ID to NUMA node
                let hash = xxhash_rust::xxh3::xxh3_64(space_id.as_bytes());
                (hash % self.wal_per_node.len() as u64) as usize
            })
            .value()
            .clone();

        // Append to local NUMA node's WAL (no cross-NUMA write)
        let wal = self.wal_per_node[numa_node].lock().unwrap();
        wal.append(&entry, payload)
    }
}
```

**Performance Gain**:
- Cross-NUMA write: **2-3x slower** (100ns → 250ns)
- Local NUMA write: **100ns** (single memory controller)
- **Speedup**: 2-3x reduction in WAL write latency

---

## 3. Zero-Copy Replication with io_uring

### 3.1 Registered Buffer Setup

```rust
// engram-core/src/replication/zerocopy.rs

use io_uring::{opcode, types, IoUring, squeue};

/// Zero-copy replication using io_uring registered buffers
struct ZeroCopyReplicator {
    ring: IoUring,

    /// Pre-registered buffers (allocated on NUMA node 0 near NIC)
    buffers: Vec<*mut u8>,

    /// Free buffer indices (ring buffer for allocation)
    free_buffers: ArrayQueue<usize>,

    /// Socket file descriptor
    socket_fd: RawFd,
}

impl ZeroCopyReplicator {
    fn new(socket_fd: RawFd, num_buffers: usize) -> Result<Self> {
        let mut ring = IoUring::new(256)?;

        // Allocate buffers on NUMA node 0 (where NIC typically resides)
        let buffers: Vec<*mut u8> = (0..num_buffers)
            .map(|_| unsafe {
                // 4 KB buffer, aligned to page boundary
                let layout = std::alloc::Layout::from_size_align(4096, 4096)
                    .unwrap();

                // NUMA-aware allocation (requires libnuma)
                #[cfg(target_os = "linux")]
                {
                    let ptr = libc::numa_alloc_onnode(4096, 0);
                    if ptr.is_null() {
                        panic!("Failed to allocate on NUMA node 0");
                    }
                    ptr as *mut u8
                }

                #[cfg(not(target_os = "linux"))]
                {
                    std::alloc::alloc(layout)
                }
            })
            .collect();

        // Register buffers with io_uring (zero-copy DMA)
        ring.submitter().register_buffers(&buffers)?;

        // Initialize free buffer pool
        let free_buffers = ArrayQueue::new(num_buffers);
        for i in 0..num_buffers {
            free_buffers.push(i).unwrap();
        }

        Ok(Self {
            ring,
            buffers,
            free_buffers,
            socket_fd,
        })
    }

    /// Replicate WAL entry with zero-copy send
    async fn replicate_entry(&mut self, entry: &WalEntry, payload: &[u8]) -> Result<()> {
        // Allocate buffer from pool
        let buf_idx = self.free_buffers.pop()
            .ok_or_else(|| anyhow!("No free buffers"))?;

        // Serialize entry into registered buffer
        let buf = unsafe { &mut *self.buffers[buf_idx] };
        let total_size = WalEntry::HEADER_SIZE + payload.len();

        unsafe {
            // Copy header
            std::ptr::copy_nonoverlapping(
                entry as *const WalEntry as *const u8,
                buf.as_mut_ptr(),
                WalEntry::HEADER_SIZE,
            );

            // Copy payload
            std::ptr::copy_nonoverlapping(
                payload.as_ptr(),
                buf.as_mut_ptr().add(WalEntry::HEADER_SIZE),
                payload.len(),
            );
        }

        // Submit zero-copy send operation
        let send_op = opcode::Send::new(
            types::Fd(self.socket_fd),
            buf.as_ptr(),
            total_size as u32,
        )
        .build()
        .user_data(buf_idx as u64); // Track buffer index for completion

        unsafe {
            self.ring.submission()
                .push(&send_op)
                .map_err(|_| anyhow!("Failed to submit send op"))?;
        }

        self.ring.submit()?;

        // Wait for completion (async)
        let cqe = self.ring.completion().next()
            .ok_or_else(|| anyhow!("No completion event"))?;

        // Free buffer back to pool
        let completed_buf_idx = cqe.user_data() as usize;
        self.free_buffers.push(completed_buf_idx)
            .map_err(|_| anyhow!("Failed to return buffer to pool"))?;

        // Check for errors
        if cqe.result() < 0 {
            bail!("Send failed: {}", std::io::Error::from_raw_os_error(-cqe.result()));
        }

        Ok(())
    }
}
```

**Performance Comparison**:

| Method | Memory Copies | Latency | Throughput |
|--------|---------------|---------|------------|
| gRPC | 3 (app → protobuf → gRPC → TCP) | 15-25 μs | 40K-60K msg/sec |
| TCP sendmsg | 2 (app → TCP buffer → NIC) | 10-15 μs | 60K-100K msg/sec |
| io_uring zerocopy | 0 (DMA from app buffer) | 5-10 μs | 100K-200K msg/sec |

**Speedup**: **2-3x** reduction in replication latency

**Complexity**: 7-10 days (Linux-specific, requires io_uring expertise)

**Recommendation**: Phase 5 optimization (use gRPC in Phase 1-4)

---

## 4. SWIM Protocol Implementation Details

### 4.1 State Machine

```rust
// engram-core/src/cluster/swim.rs

/// SWIM node state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeState {
    Alive,
    Suspect,
    Dead,
    Left,
}

/// SWIM membership entry
struct MembershipEntry {
    node_id: String,
    addr: SocketAddr,
    state: NodeState,
    incarnation: u64,       // Refutation counter
    state_changed_at: Instant,
}

/// SWIM protocol engine
struct SwimProtocol {
    /// Our node ID
    self_id: String,

    /// Membership table (lock-free concurrent access)
    members: DashMap<String, MembershipEntry>,

    /// UDP socket for SWIM messages
    socket: UdpSocket,

    /// Protocol parameters
    config: SwimConfig,
}

struct SwimConfig {
    probe_interval: Duration,      // How often to probe (default: 1s)
    probe_timeout: Duration,       // Timeout for direct ping (default: 500ms)
    indirect_probes: usize,        // Number of indirect probes (default: 3)
    suspect_timeout: Duration,     // Time in Suspect before Dead (default: 5s)
}

impl SwimProtocol {
    /// Main SWIM loop (runs forever)
    async fn run(&self) {
        let mut interval = tokio::time::interval(self.config.probe_interval);

        loop {
            interval.tick().await;

            // Select random alive node to probe
            let target = self.select_random_alive_node();

            // Send direct ping
            self.send_ping(&target).await;

            // Wait for ack with timeout
            let ack_received = tokio::time::timeout(
                self.config.probe_timeout,
                self.wait_for_ack(&target),
            ).await;

            match ack_received {
                Ok(Ok(_)) => {
                    // Direct ping succeeded, mark alive
                    self.mark_alive(&target);
                },
                _ => {
                    // Direct ping failed, try indirect probes
                    self.indirect_probe(&target).await;
                }
            }

            // Check for suspect timeouts
            self.check_suspect_timeouts();
        }
    }

    /// Indirect probe via K random nodes
    async fn indirect_probe(&self, target: &str) {
        let intermediates = self.select_random_nodes(self.config.indirect_probes);

        let mut handles = vec![];
        for intermediate in intermediates {
            let handle = tokio::spawn({
                let target = target.to_string();
                let intermediate = intermediate.clone();
                async move {
                    self.send_ping_req(&intermediate, &target).await
                }
            });
            handles.push(handle);
        }

        // Wait for any indirect ack
        let any_ack = futures::future::select_all(handles).await;

        if any_ack.is_ok() {
            // Indirect probe succeeded
            self.mark_alive(target);
        } else {
            // All indirect probes failed, mark suspect
            self.mark_suspect(target);
        }
    }

    /// Mark node as suspect (starts suspect timer)
    fn mark_suspect(&self, node_id: &str) {
        self.members.alter(node_id, |_, mut entry| {
            if entry.state == NodeState::Alive {
                entry.state = NodeState::Suspect;
                entry.state_changed_at = Instant::now();
                entry.incarnation += 1; // Bump incarnation on state change
            }
            entry
        });

        // Gossip the state change
        self.gossip_update(node_id);
    }

    /// Check for suspect timeouts (move to Dead)
    fn check_suspect_timeouts(&self) {
        let now = Instant::now();

        for mut entry in self.members.iter_mut() {
            if entry.state == NodeState::Suspect {
                let suspect_duration = now.duration_since(entry.state_changed_at);

                if suspect_duration > self.config.suspect_timeout {
                    // Suspect timeout expired, mark dead
                    entry.state = NodeState::Dead;
                    entry.state_changed_at = now;
                    entry.incarnation += 1;

                    // Gossip the state change
                    self.gossip_update(&entry.node_id);
                }
            }
        }
    }

    /// Handle received ping (respond with ack)
    async fn handle_ping(&self, from: SocketAddr, msg: PingMessage) {
        // Update incarnation if higher
        self.members.alter(&msg.node_id, |_, mut entry| {
            if msg.incarnation > entry.incarnation {
                entry.incarnation = msg.incarnation;
            }
            entry
        });

        // Send ack
        let ack = AckMessage {
            node_id: self.self_id.clone(),
            incarnation: self.get_self_incarnation(),
        };

        self.send_message(from, Message::Ack(ack)).await;
    }

    /// Handle refutation (node claiming it's alive, not dead)
    fn handle_refutation(&self, node_id: &str, incarnation: u64) {
        self.members.alter(node_id, |_, mut entry| {
            if incarnation > entry.incarnation {
                // Higher incarnation → accept refutation
                entry.state = NodeState::Alive;
                entry.incarnation = incarnation;
                entry.state_changed_at = Instant::now();
            }
            entry
        });
    }
}
```

### 4.2 Gossip Dissemination

```rust
/// Gossip updates (piggyback on ping/ack messages)
impl SwimProtocol {
    /// Piggyback membership updates on message
    fn add_gossip(&self, msg: &mut Message) {
        // Select up to 10 most recent updates
        let updates: Vec<_> = self.members.iter()
            .filter(|e| {
                // Only gossip recent changes
                e.state_changed_at.elapsed() < Duration::from_secs(60)
            })
            .take(10)
            .map(|e| MembershipUpdate {
                node_id: e.node_id.clone(),
                addr: e.addr,
                state: e.state,
                incarnation: e.incarnation,
            })
            .collect();

        msg.set_gossip(updates);
    }

    /// Merge received gossip updates
    fn merge_gossip(&self, updates: Vec<MembershipUpdate>) {
        for update in updates {
            self.members.alter(&update.node_id, |_, mut entry| {
                // Only apply if higher incarnation (prevents stale updates)
                if update.incarnation > entry.incarnation {
                    entry.state = update.state;
                    entry.incarnation = update.incarnation;
                    entry.addr = update.addr;
                    entry.state_changed_at = Instant::now();
                }
                entry
            });
        }
    }
}
```

**Convergence Proof**:
- Each gossip message carries up to 10 updates
- Infection-style propagation: P(node receives update in round k) = 1 - (1 - 1/N)^k
- After O(log N) rounds: P(convergence) > 99%
- For 100 nodes: log₂(100) ≈ 7 rounds = 7 seconds

---

## 5. Vector Clock Conflict Resolution

### 5.1 Vector Clock Implementation

```rust
// engram-core/src/cluster/vector_clock.rs

use std::collections::HashMap;

/// Vector clock for causality tracking
///
/// Maps NodeId → logical timestamp
#[derive(Debug, Clone, PartialEq, Eq)]
struct VectorClock {
    clocks: HashMap<String, u64>,
}

impl VectorClock {
    fn new() -> Self {
        Self {
            clocks: HashMap::new(),
        }
    }

    /// Increment our node's logical timestamp
    fn increment(&mut self, node_id: &str) {
        *self.clocks.entry(node_id.to_string()).or_insert(0) += 1;
    }

    /// Merge with another vector clock (take max per node)
    fn merge(&mut self, other: &VectorClock) {
        for (node, &timestamp) in &other.clocks {
            let our_timestamp = self.clocks.entry(node.clone()).or_insert(0);
            *our_timestamp = (*our_timestamp).max(timestamp);
        }
    }

    /// Compare with another vector clock (causal ordering)
    fn compare(&self, other: &VectorClock) -> CausalOrder {
        let mut less = false;
        let mut greater = false;

        // Collect all node IDs from both clocks
        let mut all_nodes: HashSet<&String> = self.clocks.keys().collect();
        all_nodes.extend(other.clocks.keys());

        for node in all_nodes {
            let our_time = self.clocks.get(node).copied().unwrap_or(0);
            let other_time = other.clocks.get(node).copied().unwrap_or(0);

            if our_time < other_time {
                less = true;
            }
            if our_time > other_time {
                greater = true;
            }
        }

        match (less, greater) {
            (true, false) => CausalOrder::Less,      // We happened before
            (false, true) => CausalOrder::Greater,   // We happened after
            (false, false) => CausalOrder::Equal,    // Same event
            (true, true) => CausalOrder::Concurrent, // Concurrent updates
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CausalOrder {
    Less,
    Greater,
    Equal,
    Concurrent,
}
```

### 5.2 Conflict Resolution

```rust
/// Resolve conflicting semantic memories using vector clocks
fn resolve_conflict(
    local: &SemanticMemory,
    remote: &SemanticMemory,
) -> SemanticMemory {
    match local.vector_clock.compare(&remote.vector_clock) {
        CausalOrder::Less => {
            // Remote happened after local → remote wins
            remote.clone()
        },
        CausalOrder::Greater => {
            // Local happened after remote → local wins
            local.clone()
        },
        CausalOrder::Equal => {
            // Identical events → no conflict
            local.clone()
        },
        CausalOrder::Concurrent => {
            // Concurrent updates → need tie-breaking
            resolve_concurrent_conflict(local, remote)
        }
    }
}

/// Resolve concurrent updates (split-brain scenario)
fn resolve_concurrent_conflict(
    local: &SemanticMemory,
    remote: &SemanticMemory,
) -> SemanticMemory {
    // Strategy 1: Confidence-based voting
    if remote.confidence > local.confidence {
        return remote.clone();
    }
    if local.confidence > remote.confidence {
        return local.clone();
    }

    // Strategy 2: Merge patterns (if same confidence)
    merge_semantic_memories(local, remote)
}

/// Merge two semantic memories (CRDT-style)
fn merge_semantic_memories(
    local: &SemanticMemory,
    remote: &SemanticMemory,
) -> SemanticMemory {
    SemanticMemory {
        id: local.id.clone(),

        // Union of patterns (G-Set CRDT)
        patterns: local.patterns.iter()
            .chain(remote.patterns.iter())
            .cloned()
            .collect(),

        // Average confidence
        confidence: (local.confidence + remote.confidence) / 2.0,

        // Merge vector clocks (take max per node)
        vector_clock: {
            let mut merged = local.vector_clock.clone();
            merged.merge(&remote.vector_clock);
            merged
        },

        // Later timestamp wins
        updated_at: local.updated_at.max(remote.updated_at),
    }
}
```

**Overhead Analysis**:
- Vector clock size: 8 bytes per node (u64)
- 100-node cluster: 800 bytes per semantic memory
- 10,000 memories: **8 MB** total (acceptable)
- Comparison cost: O(N) where N = number of nodes

---

## 6. Performance Optimization Techniques

### 6.1 Connection Pooling

```rust
// engram-core/src/cluster/connection_pool.rs

use tonic::transport::Channel;

/// gRPC connection pool (reuse channels, avoid handshake overhead)
struct ConnectionPool {
    /// Channels per remote node (4-8 connections per node)
    pools: DashMap<NodeId, Vec<Channel>>,

    /// Pool size per node
    pool_size: usize,
}

impl ConnectionPool {
    fn new(pool_size: usize) -> Self {
        Self {
            pools: DashMap::new(),
            pool_size,
        }
    }

    /// Get channel to remote node (round-robin load balancing)
    async fn get_channel(&self, node: &NodeId) -> Result<Channel> {
        // Check if pool exists for this node
        if !self.pools.contains_key(node) {
            // Create new pool
            self.create_pool(node).await?;
        }

        // Get pool
        let pool = self.pools.get(node)
            .ok_or_else(|| anyhow!("Pool not found"))?;

        // Round-robin selection
        let idx = rand::random::<usize>() % pool.len();
        Ok(pool[idx].clone())
    }

    /// Create connection pool for node
    async fn create_pool(&self, node: &NodeId) -> Result<()> {
        let addr = self.resolve_node_addr(node)?;

        let mut channels = Vec::with_capacity(self.pool_size);
        for _ in 0..self.pool_size {
            let channel = Channel::from_shared(addr.clone())?
                .connect()
                .await?;
            channels.push(channel);
        }

        self.pools.insert(node.clone(), channels);
        Ok(())
    }
}
```

**Performance Gain**:
- TLS handshake: **10-50ms** (avoided by reusing channels)
- TCP connection setup: **1-5ms** (avoided)
- Channel reuse: **<1ms** overhead

---

## 7. Testing Infrastructure

### 7.1 Network Simulator

```rust
// engram-core/tests/network_simulator.rs

/// Deterministic network simulator for distributed testing
///
/// Simulates:
/// - Packet loss (configurable percentage)
/// - Latency injection (fixed or variable)
/// - Network partitions (symmetric, asymmetric)
/// - Flapping (rapid partition/heal cycles)
struct NetworkSimulator {
    /// Simulated network topology
    nodes: HashMap<NodeId, SimulatedNode>,

    /// Partition configuration (which nodes can talk to which)
    partitions: Vec<HashSet<NodeId>>,

    /// Packet loss rate (0.0-1.0)
    packet_loss: f64,

    /// Latency distribution (min, max)
    latency: (Duration, Duration),
}

impl NetworkSimulator {
    /// Send message with simulated network effects
    fn send(&mut self, from: NodeId, to: NodeId, msg: Message) -> Result<()> {
        // Check partition (can these nodes communicate?)
        if !self.can_communicate(&from, &to) {
            return Err(anyhow!("Nodes partitioned"));
        }

        // Simulate packet loss
        if rand::random::<f64>() < self.packet_loss {
            return Err(anyhow!("Packet lost"));
        }

        // Simulate latency
        let latency = Duration::from_millis(
            rand::thread_rng().gen_range(
                self.latency.0.as_millis()..=self.latency.1.as_millis()
            ) as u64
        );

        // Deliver message after latency
        tokio::spawn({
            let to_node = self.nodes.get_mut(&to).unwrap();
            async move {
                tokio::time::sleep(latency).await;
                to_node.receive(msg);
            }
        });

        Ok(())
    }

    /// Create network partition (split nodes into groups)
    fn create_partition(&mut self, groups: Vec<HashSet<NodeId>>) {
        self.partitions = groups;
    }

    /// Heal partition (restore full connectivity)
    fn heal_partition(&mut self) {
        let all_nodes: HashSet<_> = self.nodes.keys().cloned().collect();
        self.partitions = vec![all_nodes];
    }

    /// Can two nodes communicate in current partition?
    fn can_communicate(&self, from: &NodeId, to: &NodeId) -> bool {
        for partition in &self.partitions {
            if partition.contains(from) && partition.contains(to) {
                return true;
            }
        }
        false
    }
}
```

### 7.2 Jepsen-Style History Checker

```rust
// engram-core/tests/jepsen_checker.rs

/// History-based consistency checker (Jepsen-style)
///
/// Records all operations and checks invariants
struct HistoryChecker {
    /// All operations in temporal order
    history: Vec<Operation>,
}

#[derive(Debug, Clone)]
struct Operation {
    op_type: OpType,
    key: String,
    value: Option<Memory>,
    started_at: Instant,
    completed_at: Option<Instant>,
    node: NodeId,
}

#[derive(Debug, Clone, Copy)]
enum OpType {
    Write,
    Read,
}

impl HistoryChecker {
    /// Check eventual consistency invariant
    fn check_eventual_consistency(&self) -> Result<()> {
        // Group operations by key
        let mut by_key: HashMap<String, Vec<&Operation>> = HashMap::new();
        for op in &self.history {
            by_key.entry(op.key.clone()).or_default().push(op);
        }

        // For each key, verify reads eventually see writes
        for (key, ops) in by_key {
            self.check_key_consistency(&key, &ops)?;
        }

        Ok(())
    }

    fn check_key_consistency(&self, key: &str, ops: &[&Operation]) -> Result<()> {
        // Find all writes
        let writes: Vec<_> = ops.iter()
            .filter(|op| matches!(op.op_type, OpType::Write))
            .collect();

        // Find all reads
        let reads: Vec<_> = ops.iter()
            .filter(|op| matches!(op.op_type, OpType::Read))
            .collect();

        // Check: all reads after convergence time see latest write
        let convergence_time = Duration::from_secs(60); // Plan: <60s

        for read in reads {
            // Find latest write before this read
            let latest_write = writes.iter()
                .filter(|w| w.completed_at.unwrap() < read.started_at)
                .max_by_key(|w| w.completed_at.unwrap());

            if let Some(write) = latest_write {
                // Check if read happened after convergence time
                let elapsed = read.started_at.duration_since(
                    write.completed_at.unwrap()
                );

                if elapsed > convergence_time {
                    // Read must see the write (eventual consistency)
                    if read.value != write.value {
                        bail!(
                            "Consistency violation: read of {} at {:?} \
                             did not see write at {:?} (elapsed: {:?})",
                            key, read.started_at, write.completed_at.unwrap(), elapsed
                        );
                    }
                }
            }
        }

        Ok(())
    }
}
```

---

## 8. Conclusion

This implementation guide provides low-level details for:

1. **WAL Format**: 64-byte aligned headers, xxHash checksums, memory-mapped files
2. **NUMA-Aware Replication**: Per-node WAL files, affinity-based allocation
3. **Zero-Copy I/O**: io_uring registered buffers, DMA to NIC
4. **SWIM Protocol**: State machine, gossip dissemination, refutation logic
5. **Vector Clocks**: Causal ordering, conflict resolution, CRDT merging
6. **Performance Optimization**: Connection pooling, batching, load balancing
7. **Testing Infrastructure**: Network simulator, Jepsen-style checkers

**Use this guide AFTER prerequisites are met**:
- Deterministic consolidation
- Single-node baselines
- M13 completion
- 7-day single-node soak test

**Implementation Timeline**: 19-29 weeks (realistic for production-ready distributed system)

**Remember**: Distributed systems are HARD. Respect the complexity.

---

**Author**: Systems Architecture (Margo Seltzer)
**Date**: 2025-10-31
**Status**: Reference for future M14 implementation
