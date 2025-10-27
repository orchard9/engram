//! Lock-free SPSC ring buffer for event storage
//!
//! Single producer (worker thread), single consumer (collector thread).
//! When full, drops oldest events to maintain bounded memory.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Lock-free ring buffer with bounded memory usage
///
/// Implements SPSC (single-producer, single-consumer) pattern with
/// lock-free operations. When the buffer is full, it overwrites the
/// oldest events (ring behavior).
pub struct RingBuffer<T> {
    /// Pre-allocated event storage
    buffer: Box<[Option<T>]>,
    /// Write position (producer)
    write_pos: AtomicUsize,
    /// Read position (consumer)
    read_pos: AtomicUsize,
    /// Buffer capacity (power of 2 for efficient modulo)
    capacity: usize,
}

impl<T: Copy> RingBuffer<T> {
    /// Create ring buffer with specified capacity
    ///
    /// Capacity is rounded up to next power of 2 for efficient modulo.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.next_power_of_two();
        let mut buffer = Vec::with_capacity(capacity);
        buffer.resize_with(capacity, || None);

        Self {
            buffer: buffer.into_boxed_slice(),
            write_pos: AtomicUsize::new(0),
            read_pos: AtomicUsize::new(0),
            capacity,
        }
    }

    /// Push event (producer side)
    ///
    /// If buffer is full, overwrites oldest event (ring behavior).
    /// Returns true if event was recorded, false if dropped.
    #[inline]
    #[allow(unsafe_code)] // Required for SPSC lock-free ring buffer
    pub fn push(&self, event: T) -> bool {
        let write_pos = self.write_pos.load(Ordering::Relaxed);
        let read_pos = self.read_pos.load(Ordering::Acquire);

        let next_write = (write_pos + 1) % self.capacity;

        // Check if buffer is full
        if next_write == read_pos {
            // Buffer full - drop oldest event by advancing read position
            self.read_pos
                .store((read_pos + 1) % self.capacity, Ordering::Release);
        }

        // Write event
        // SAFETY: write_pos is exclusively owned by producer thread in SPSC pattern
        // We use get_unchecked_mut through a const cast to write to the pre-allocated buffer
        unsafe {
            let slot = self.buffer.as_ptr().add(write_pos).cast_mut();
            slot.write(Some(event));
        }

        // Advance write position
        self.write_pos.store(next_write, Ordering::Release);

        true
    }

    /// Pop event (consumer side)
    ///
    /// Returns None if buffer is empty.
    #[inline]
    #[allow(unsafe_code)] // Required for SPSC lock-free ring buffer
    pub fn pop(&self) -> Option<T> {
        let read_pos = self.read_pos.load(Ordering::Relaxed);
        let write_pos = self.write_pos.load(Ordering::Acquire);

        // Check if buffer is empty
        if read_pos == write_pos {
            return None;
        }

        // Read event
        // SAFETY: read_pos is exclusively owned by consumer thread
        let event = unsafe {
            let slot = &raw const self.buffer[read_pos];
            *slot
        };

        // Advance read position
        self.read_pos
            .store((read_pos + 1) % self.capacity, Ordering::Release);

        event
    }

    /// Get number of events currently in buffer
    #[must_use]
    #[allow(dead_code)] // Utility method for diagnostics
    pub fn len(&self) -> usize {
        let write_pos = self.write_pos.load(Ordering::Acquire);
        let read_pos = self.read_pos.load(Ordering::Acquire);

        if write_pos >= read_pos {
            write_pos - read_pos
        } else {
            self.capacity - read_pos + write_pos
        }
    }

    /// Check if buffer is empty
    #[must_use]
    #[allow(dead_code)] // Utility method for diagnostics
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get buffer capacity
    #[must_use]
    #[allow(dead_code)] // Utility method for diagnostics
    pub const fn capacity(&self) -> usize {
        self.capacity
    }
}

// SAFETY: RingBuffer is thread-safe for SPSC pattern
#[allow(unsafe_code)] // Required for Send/Sync impl on SPSC ring buffer
unsafe impl<T: Send> Send for RingBuffer<T> {}
#[allow(unsafe_code)] // Required for Send/Sync impl on SPSC ring buffer
unsafe impl<T: Send> Sync for RingBuffer<T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_ring_buffer_creation() {
        let buffer: RingBuffer<u64> = RingBuffer::new(100);
        assert_eq!(buffer.capacity(), 128); // Rounded to next power of 2
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_ring_buffer_push_pop() {
        let buffer: RingBuffer<u64> = RingBuffer::new(100);

        buffer.push(42);
        buffer.push(84);
        buffer.push(126);

        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.pop(), Some(42));
        assert_eq!(buffer.pop(), Some(84));
        assert_eq!(buffer.pop(), Some(126));
        assert_eq!(buffer.pop(), None);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_ring_buffer_overflow() {
        let buffer: RingBuffer<u64> = RingBuffer::new(4); // Will round to 4

        // Fill the buffer
        buffer.push(1);
        buffer.push(2);
        buffer.push(3);

        // This should overwrite the oldest event (1)
        buffer.push(4);

        // Verify that oldest event was dropped
        assert_eq!(buffer.pop(), Some(2));
        assert_eq!(buffer.pop(), Some(3));
        assert_eq!(buffer.pop(), Some(4));
        assert_eq!(buffer.pop(), None);
    }

    #[test]
    #[allow(clippy::unwrap_used)] // Test code: panic on failure is acceptable
    fn test_ring_buffer_concurrent_spsc() {
        let buffer = Arc::new(RingBuffer::new(1000));
        let producer_buffer = Arc::clone(&buffer);
        let consumer_buffer = Arc::clone(&buffer);

        // Producer thread
        let producer = thread::spawn(move || {
            for i in 0..10_000 {
                producer_buffer.push(i);
            }
        });

        // Consumer thread
        let consumer = thread::spawn(move || {
            let mut consumed = 0;
            while consumed < 10_000 {
                if consumer_buffer.pop().is_some() {
                    consumed += 1;
                }
            }
            consumed
        });

        producer.join().unwrap();
        let count = consumer.join().unwrap();
        assert_eq!(count, 10_000);
    }

    #[test]
    fn test_ring_buffer_power_of_two_capacity() {
        assert_eq!(RingBuffer::<u64>::new(100).capacity(), 128);
        assert_eq!(RingBuffer::<u64>::new(256).capacity(), 256);
        assert_eq!(RingBuffer::<u64>::new(1000).capacity(), 1024);
        assert_eq!(RingBuffer::<u64>::new(10_000).capacity(), 16384);
    }
}
