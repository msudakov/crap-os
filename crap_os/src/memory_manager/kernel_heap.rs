//! Kernel Heap Allocator
//!
//! This implementation is a linked free-list allocator backed by the kernel
//! VMM. It implements `GlobalAlloc`, so Rust's `alloc` crate (Box, Vec, String,
//! etc.) works transparently once registered with `#[global_allocator]`.
//!
//! The heap grows on demand from a fixed virtual address range
//! `[base, base + max_size)`. Physical pages are mapped into that range in
//! chunks via "grow" calls. Free memory is tracked as a singly-linked list of
//! `FreeBlock` nodes, kept sorted by ascending address. Keeping the list sorted
//! allows adjacent freed blocks to be merged (coalesced) in a single pass,
//! preventing fragmentation over time.
//!
//! Each allocation stores an `AllocHeader` immediately before the returned
//! pointer, plus a copy of the `data_offset` field in the 8 bytes directly
//! before the returned data pointer. This redundant copy allows `deallocate` to
//! recover the block start without reading the full header, which is crucial
//! when the pointer is the only information the caller provides on `free`.
//!
//! > This is the memory layout of an allocated block:
//!   [ AllocHeader | <alignment padding> | data_offset | <data> ]
//!     ^                                                  ^^^^
//!     block_ptr                                     pointer returned to caller
//!
//! > This is another view of the same layout of an allocated block:
//!
//!  +----------------------------------+  <- block_ptr (raw page-aligned base)
//!  |  AllocHeader                     |
//!  |    total_size: usize  (8 bytes)  |
//!  |    data_offset: usize (8 bytes)  |
//!  |----------------------------------|
//!  |  alignment padding (0+ bytes)    |
//!  |----------------------------------|
//!  |  data_offset copy  (8 bytes)     |  <- data_ptr - 8
//!  |----------------------------------|  <- data_ptr (returned to caller)
//!  |  data  (layout.size bytes)       |
//!  +----------------------------------+  <- block_ptr + total_size
//!
//! > This is the memory layout of a free block:
//!   [ FreeBlock { size, *next } | ... unused space ... ]
//!     ^
//!     block_ptr
//!
//! > This is another view of the same layout of a free block:
//!
//!  +----------------------------------+  <- FreeBlock pointer
//!  |  size: usize   (8 bytes)         |  Total bytes available in this block
//!  |  next: *mut FreeBlock (8 bytes)  |  Pointer to next free block
//!  +----------------------------------+
//!
//! FreeBlock::size is the total size of the free region, including the
//! FreeBlock header itself. FreeBlock::next is the next free block in
//! address-sorted order, or null if this is the tail of the list.
//!
//! > Free list ordering:
//!     The free list is kept sorted by ascending block address. This makes
//!     coalescing O(1) per `free` (just check immediate neighbors) at the cost
//!     of O(n) sorted insertion, which is acceptable since `free` is not on a
//!     hot path in a kernel heap.
//!
//! > Lock ordering (this must never be inverted!):
//!     KERNEL_HEAP -> MEMORY_MANAGER
//!
//! This way, `grow()` acquires the MEMORY_MANAGER lock while the caller holds
//! the KERNEL_HEAP lock. No code path may hold MEMORY_MANAGER and then trigger
//! a heap allocation. Otherwise, a deadlock would be imminent.
//!
//! Two IRQ-safe spinlocks are necessary here, and this is what happens during a
//! heap allocation:
//!
//! caller
//! |_ GlobalHeapAllocator::alloc()
//!    |_ KERNEL_HEAP.heap.lock()      -> this acquires heap lock
//!    .  |_ KernelHeap::allocate()
//!    .     |_ if free list has space: no second lock needed, returns directly
//!    .     |_ if free list exhausted:
//!    .        |_ KernelHeap::grow()
//!    .        .  |_ MEMORY_MANAGER.lock()  -> this acquires MM lock
//!    .        .  .  |_ mm.alloc_page()
//!    .        .  .  |_ mm.map_page()
//!    .        .  |_ MEMORY_MANAGER unlocked
//!    .        |_ retries free list
//!    |_ KERNEL_HEAP unlocked

use crate::memory_manager::{PRESENT, WRITABLE, NX};
use crate::globals::{PAGE_SIZE, MEMORY_MANAGER, KERNEL_HEAP};
use core::ptr;
use core::alloc::{GlobalAlloc, Layout};

/// A node in the sorted free-block linked list, stored inline inside free
/// memory.
///
/// This struct is written directly into the first bytes of the free memory it
/// describes, so no separate allocation is needed. That means a free block must
/// be at least `FreeBlock::SIZE` bytes (16 bytes on x64) to hold this header.
#[repr(C)]
struct FreeBlock {
    /// Total number of bytes in this free region, including the bytes
    /// occupied by this `FreeBlock` header itself.
    size: usize,

    /// Pointer to the next `FreeBlock` in the sorted free list, or null
    /// if this is the last node.
    next: *mut FreeBlock,
}

impl FreeBlock {
    /// The size of the `FreeBlock` header in bytes (16 bytes on x64).
    ///
    /// This is used as the minimum block size; any region smaller than this
    /// cannot store the free-list node and therefore cannot be tracked.
    const SIZE: usize = core::mem::size_of::<FreeBlock>();
}

/// Header written at the very start of each allocated block at `block_ptr`,
/// before the alignment padding and data. It contains the metadata stored
/// before each live allocation.
///
/// When the caller calls `deallocate(data_ptr)`, we use the `data_offset` copy
/// stored just before `data_ptr` to walk back to `block_ptr`, then read this
/// header to get the block's total size, so we can return it to the free list.
/// The `data_offset` is always the last field, so it sits at exactly
/// `size_of::<usize>()` bytes before the `data_ptr`, regardless of how much
/// padding was inserted for alignment. This lets `deallocate` recover
/// `block_ptr` with a single pointer subtraction without needing to recompute
/// alignment arithmetic.
#[repr(C)]
struct AllocHeader {
    /// Total size of the allocated block in bytes, from `block_ptr` to the end
    /// of the data region. Used by `deallocate` to know how many bytes to
    /// return to the free list.
    total_size:  usize,
    
    /// Byte offset from `block_ptr` to the start of the callee-visible data.
    /// Equals `AllocHeader::SIZE` rounded up to the allocation's alignment.
    /// It is stored here and, again, at 8 bytes before `data_ptr`, so 
    /// `deallocate` can recover `block_ptr = data_ptr - data_offset`.
    data_offset: usize,
}

impl AllocHeader {
    /// The size of the `AllocHeader` in bytes (16 bytes on x64).
    ///
    /// Used as the base for computing `data_offset`: the data starts at least
    /// `AllocHeader::SIZE` bytes into the block, then rounded up further to
    /// satisfy the allocation's alignment requirement.
    const SIZE: usize = core::mem::size_of::<AllocHeader>();
}

// =============================================================================
// KernelHeap - the heap allocator state
// =============================================================================

/// The kernel heap allocator.
///
/// Manages a virtual address range, committing physical pages into it on
/// demand and tracking free memory with a sorted linked free list.
///
/// # Thread Safety
/// 
/// `KernelHeap` itself is NOT thread-safe, and all methods take `&mut self`.
/// Thread safety is provided by the `LockedHeap` wrapper, which protects the
/// heap behind an IRQ-safe spinlock.
pub struct KernelHeap {
    /// The virtual base address of the heap region. All heap memory lives at
    /// addresses >= `base`.
    base: u64,

    /// The maximum number of bytes the heap is allowed to grow to.
    /// `base + max_size` is the exclusive upper bound of the virtual range.
    max_size: u64,

    /// How many bytes have been committed (backed by physical pages) so far.
    /// New pages are always appended at `base + committed`.
    committed: u64,

    /// Head of the sorted free-block linked list. It is `null` when the heap
    /// has no free space (e.g., when it is freshly created, and before `init`
    /// is called).
    free_list: *mut FreeBlock,
}

// SAFETY: The heap is always accessed through the global IRQ spinlock,
// which disables interrupts for the duration of every heap operation. This
// implements unsafe Send for spinlock management.
unsafe impl Send for KernelHeap {}

impl KernelHeap {
    /// Creates a new, empty `KernelHeap` covering the virtual range
    /// `[base, base + max_size)`, with `base` inclusive and `base + max_size`
    /// exclusive.
    ///
    /// No physical memory is mapped and no free blocks exist yet. Call
    /// `KernelHeap::init` before making any allocations. The `const fn`
    /// allows this to be used in `static` initializers.
    /// 
    /// #Arguments
    /// 
    /// * `base` - The virtual base address of the heap region.
    /// * `max_size` - The maximum size of the heap in bytes.
    pub const fn new(base: u64, max_size: u64) -> Self {
        Self {
            base,
            max_size,
            committed: 0,
            free_list: ptr::null_mut(),  // No free blocks yet.
        }
    }

    /// Commits `initial_pages` pages from the PMM into the heap's virtual
    /// region and inserts them into the free list, making the heap ready for
    /// allocations.
    ///
    /// Must be called once after the MemoryManager has been placed in the
    /// global and is initialized.
    pub fn init(&mut self, initial_pages: u64) {
        self.grow(initial_pages);
    }

    /// Maps `num_pages` new physical pages into the next available virtual
    /// address range in the heap and adds them as a single free block.
    ///
    /// Pages are appended starting at `base + committed`, then `committed`
    /// is advanced by `pages * PAGE_SIZE`.
    ///
    /// Panics if the grow would exceed `max_size`, or if the physical memory
    /// manager is out of pages.
    /// 
    /// # Arguments
    /// 
    /// * `num_pages` - The number of physical pages to insert into the heap.
    fn grow(&mut self, num_pages: u64) {
        let num_bytes = num_pages * PAGE_SIZE;  // Convert page count to bytes
        
        // Enforce the hard upper bound on the heap's virtual range
        assert!(
            self.committed + num_bytes <= self.max_size,
            "[KERNEL_HEAP] Out of heap memory: grow() would exceed max size"
        );

        // The virtual address at which the new region starts. New regions are
        // always appended immediately after previously committed memory.
        let region_virtual_addr = self.base + self.committed;

        {
            // Lock the global memory manager, which also disables interrupts
            // so an interrupt handler cannot try to allocate concurrently
            // while we're in mid-grow.
            let mut mm_guard = MEMORY_MANAGER.lock();
            let mm = mm_guard.as_mut()
                .expect("[KERNEL_HEAP] MemoryManager is not initialized");

            // Allocate and map one physical page at a time
            for i in 0..num_pages {
                // Compute the virtual address for this page
                let virt_addr = region_virtual_addr + i * PAGE_SIZE;

                // Allocate a free physical page from the PMM
                let phys_addr = mm.alloc_page()
                    .expect("[KERNEL_HEAP] PMM out of memory during heap grow");

                // Map virtual to physical in the kernel's page tables.
                unsafe {
                    mm.map_page(virt_addr, phys_addr, PRESENT | WRITABLE | NX)
                };
            }
        } // MEMORY_MANAGER lock dropped here, and interrupts are re-enabled

        // Record that these bytes are now committed
        self.committed += num_bytes;

        // Add the entire newly committed region to the free list as one big
        // block. This will merge it with adjacent blocks if possible.
        unsafe {
            self.insert_free_block_sorted(region_virtual_addr as *mut u8,
                num_bytes as usize);
        }
    }

    /// Inserts a raw memory region into the sorted free list and immediately
    /// attempts to coalesce it with its neighbors. The free list is kept in
    /// ascending order of pointer value. This makes coalescing of adjacent
    /// blocks possible in O(1) after insertion.
    ///
    /// # Coalescing Algorithm
    /// 
    /// After inserting the block between `prev` and `next`:
    ///   1. If the end of the new block exactly meets the start of `next`,
    ///      merge the new block and `next` into one larger block.
    ///   2. If the end of `prev` exactly meets the start of the new block,
    ///      merge `prev` and the new block into one larger block.
    ///
    /// # Arguments
    /// * `ptr`        - Start of the memory region to add to the free list.
    /// * `total_size` - Total number of bytes in the region.
    /// 
    /// # Safety
    ///
    /// `ptr` must point to at least `FreeBlock::SIZE` bytes of valid,
    /// exclusively-owned, writable memory. `total_size` must be the true byte
    /// count of the entire region starting at `ptr`.
    unsafe fn insert_free_block_sorted(&mut self, ptr: *mut u8,
        total_size: usize
    ) {
        assert!(
            total_size >= FreeBlock::SIZE,
            "[KERNEL_HEAP] block is too small to hold FreeBlock header"
        );

        // Find the correct insertion position by walking the list until we find
        // the first node whose address is >= `ptr`.
        // After the loop:
        //   `prev` = the node that should precede our new block
        //      (if null, it means insert at head);
        //   `next` = the node that should follow our new block
        //      (if null, it means insert at tail).
        let mut prev: *mut FreeBlock = ptr::null_mut();
        let mut next = self.free_list;

        while !next.is_null() && (next as usize) < (ptr as usize) {
            prev = next;
            // Advance to the next node by reading the `next` pointer stored
            // inside the current free block.
            next = unsafe { (*next).next };
        }

        // Write the FreeBlock header into the region. Reuse the first
        // `FreeBlock::SIZE` bytes of the freed region to store the list node.
        // No separate allocation is needed.
        let block = ptr as *mut FreeBlock;
        unsafe { (*block).size = total_size };  // Block covers total_size bytes
        unsafe { (*block).next = next };  // Link to the following free block

        // Splice the new node into the list
        if prev.is_null() {
            // New block is before all existing blocks, it becomes the new head
            self.free_list = block;
        }
        else {
            // Link the previous node's `next` pointer to our new block
            unsafe { (*prev).next = block };
        }

        // Coalesce with the next block (forward merge). If the byte immediately
        // after our new block is where `next` starts, they are physically
        // adjacent and can be merged into one larger block.
        if !next.is_null() {
            // The address of the byte just past our new block
            let block_end = (block as usize) + unsafe { (*block).size };
            
            if block_end == (next as usize) {
                // Merge by absorbing `next` into `block`. Add `next`'s size to
                // our size so the combined block covers both regions.
                unsafe { (*block).size += (*next).size };

                // Skip over `next` in the list as it no longer exists as a
                // separate node.
                unsafe { (*block).next  = (*next).next };
            }
        }

        // Coalesce with the previous block (backward merge). If `prev` ends
        // exactly where our new block begins, merge `prev` and our block. At
        // this point, `block` may already be the result of a forward merge
        // from above, so the backward merge can produce a 3-way merge.
        if !prev.is_null() {
            // The address of the byte just past `prev`.
            let prev_end = (prev as usize) + unsafe { (*prev).size };

            if prev_end == (block as usize) {
                // Merge by absorbing `block` into `prev`. `prev` grows to cover
                // our block and any forward-merged region.
                unsafe { (*prev).size += (*block).size };

                // Remove `block` from the list by making `prev` point to
                // `block`'s `next`.
                unsafe { (*prev).next  = (*block).next };
            }
        }
    }

    /// Computes the `total_size` and `data_offset` for an allocation described
    /// by `layout`.
    /// 
    /// # Arguments
    /// 
    /// * `layout`  -  Layout of a memory block.
    ///
    /// # Alignment Computation
    /// 
    /// The standard formula for rounding `x` up to the next multiple of `align`
    /// (when `align` is a power of two) is:
    ///   `(x + align - 1) & !(align - 1)`
    /// where:
    ///   `x` = `AllocHeader::SIZE`  (how many header bytes precede the data)
    ///   `align` = `layout.align()` (the alignment required by Rust allocator)
    ///
    /// This ensures `data_ptr = block_ptr + data_offset` is correctly aligned
    /// for the requested type, satisfying the contract of the `GlobalAlloc`
    /// trait.
    /// 
    /// # Returns
    /// 
    /// `(total_size, data_offset)`, where:
    ///   - `data_offset` is the number of bytes from `block_ptr` to the data
    ///     start, calculated by rounding `AllocHeader::SIZE` up to
    ///     `layout.align()`.
    ///   - `total_size` is the total bytes consumed from the free list for
    ///     this allocation.
    #[inline]
    fn allocation_sizes(layout: &Layout) -> (usize, usize) {
        let align = layout.align();

        // Round the header size up to the nearest multiple of `align`. This is
        // the byte offset from `block_ptr` to where data begins.
        let data_offset = (AllocHeader::SIZE + align - 1) & !(align - 1);

        // Total bytes needed: header + padding region + data.
        let total_size  = data_offset + layout.size();

        (total_size, data_offset)
    }

    /// Allocates a block of memory, satisfying a given `layout`.
    ///
    /// Performs a first-fit search through the sorted free list. We return
    /// the first free block that is large enough, rather than searching for
    /// the best-fitting or smallest-fitting block. This runs in the order O(n)
    /// (in the number of free blocks) and tends to leave larger blocks
    /// intact at the end of the list. If no block is large enough, `grow` is
    /// called to commit more physical pages, which adds a new large free block
    /// and retries from the list head.
    /// 
    /// # Arguments
    /// 
    /// * `layout`  -  Memory block layout to satisfy with this allocation.
    /// 
    /// # Returns
    ///
    /// Returns a pointer to the start of the accessible data region,
    /// aligned as required by `layout`, or a null pointer on failure
    /// (only possible if `max_size` is exhausted and the PMM is also full).
    pub unsafe fn allocate(&mut self, layout: Layout) -> *mut u8 {
        // Compute how much raw memory we need and where the data will start
        let (total_size, data_offset) = Self::allocation_sizes(&layout);

        // `previous` and `current` are the two-pointer hand-over-hand walk used
        // to splice a node out of the singly-linked list when we find a fit.
        let mut previous: *mut FreeBlock = ptr::null_mut();
        let mut current  = self.free_list;  // Start at the head of free list

        loop {
            if current.is_null() {
                // Reached the end of the free list without finding a fit.
                // Grow the heap by at least `total_size` bytes, rounded up to
                // whole pages. We request a minimum of 4 pages (16 KB) to
                // amortize the cost of growing, as it avoids growing 1 page at
                // a time for many small allocations.
                let pages_needed =
                    ((total_size as u64 + PAGE_SIZE - 1) / PAGE_SIZE).max(4);
                self.grow(pages_needed);

                // After growing, restart the search from the beginning. The
                // new free block may have been coalesced anywhere in the list.
                current  = self.free_list;
                previous = ptr::null_mut();
                continue;
            }

            // Read the size of the current free block
            let available = unsafe { (*current).size };

            if available >= total_size {
                // This block is big enough, and we can use it

                // Read `next` before we overwrite the block's memory with the
                // header
                let next = unsafe { (*current).next };

                // Splice `current` out of the free list
                if previous.is_null() {
                    // `current` is the head of the list
                    self.free_list = next;
                }
                else {
                    // `current` is in the middle or tail, so bypass it
                    unsafe { (*previous).next = next };
                }

                // Raw pointer to the start of the block we just removed
                let block_ptr = current as *mut u8;

                // Check if there is enough space left over after the allocation
                // to create a new free block from the remainder
                let remaining = available - total_size;

                if remaining >= FreeBlock::SIZE {
                    // There is enough left over to form a valid free block.
                    // The leftover region starts immediately after this
                    // allocation.
                    let leftover_ptr = unsafe { block_ptr.add(total_size) };

                    // Re-insert the leftover into the sorted free list.
                    // This may also coalesce it with the next free block.
                    unsafe {
                        self.insert_free_block_sorted(leftover_ptr, remaining);
                    }
                }
                // If remaining < FreeBlock::SIZE, the leftover is too small to
                // track, and it becomes internal fragmentation (wasted bytes at
                // the end of this allocation). The `total_size` already
                // accounts for the data so the caller sees no side effect.

                // Write the AllocHeader at the start of the block
                let header = block_ptr as *mut AllocHeader;
                // Block size for dealloc
                unsafe { (*header).total_size  = total_size };
                // Header span for dealloc
                unsafe { (*header).data_offset = data_offset };

                // Compute and return the data pointer. The data starts
                // `data_offset` bytes into the block.
                let data_ptr = unsafe { block_ptr.add(data_offset) };

                // Write a copy of `data_offset` in the 8 bytes immediately
                // before `data_ptr`. This is the "back-pointer" that
                // `deallocate` uses to recover `block_ptr` from `data_ptr`
                // without a global lookup. This overlaps with the alignment
                // padding region, which is otherwise unused.
                // Since data_offset >= AllocHeader::SIZE + any padding, there
                // is always room for this usize before `data_ptr`.
                unsafe {
                    *(data_ptr.sub(core::mem::size_of::<usize>()) as *mut usize)
                        = data_offset
                };

                return data_ptr;
            }

            // This block is too small, and we need to advance to the next one
            previous = current;
            current  = unsafe { (*current).next };
        }
    }

    /// Frees a previously allocated block and returns it to the free list.
    ///
    /// The caller only provides `ptr` (the `data_ptr` returned by `allocate`).
    /// We need to find `block_ptr` (the raw start of the block) and its
    /// `total_size`.
    /// 
    /// This is the algorithm to recover the block boundaries from just `ptr`:
    ///  1. Read `data_offset` from `ptr - sizeof(usize)`, which was written
    ///     there by `allocate` as a back-pointer;
    ///  2. Set `block_ptr = ptr - data_offset`;
    ///  3. Cast `block_ptr` to `*const AllocHeader` and read `total_size`;
    ///  4. Call `insert_free_block_sorted(block_ptr, total_size)` to return
    ///     the memory to the free list and coalesce with any neighbors.
    ///
    /// # Arguments
    /// 
    /// * `ptr`  -  Previously-allocated pointer to be freed.
    /// 
    /// # Safety
    /// 
    /// `ptr` must be a pointer previously returned by `allocate` on this heap
    /// and not yet freed. Double-free or freeing an invalid pointer causes
    /// undefined behaviour.
    pub unsafe fn deallocate(&mut self, ptr: *mut u8) {
        // 1. Recover `data_offset` from the 8 bytes just before the
        // pointer
        let data_offset = unsafe {
            *(ptr.sub(core::mem::size_of::<usize>()) as *const usize)
        };

        // 2. Walk back from the pointer to the raw block start
        let block_ptr = unsafe { ptr.sub(data_offset) };

        // 3: Read `total_size` from the `AllocHeader` at the block start
        let header = block_ptr as *const AllocHeader;
        let total_size = unsafe { (*header).total_size };

        // 4: Return the entire block to the free list
        unsafe {
            self.insert_free_block_sorted(block_ptr, total_size)
        }
    }
}

// =============================================================================
// LockedHeap - thread-safe wrapper around KernelHeap
// =============================================================================

/// A `KernelHeap` wrapped in an `IrqSpinLock` for safe concurrent access.
///
/// All kernel heap operations go through this type. The `IrqSpinLock`
/// ensures that only one thread (or CPU core) accesses the heap at a time, and
/// that interrupts are disabled while the lock is held, preventing a deadlock
/// if an interrupt handler tries to allocate memory while an allocation is
/// already in progress on the same core.
pub struct LockedHeap {
    /// The protected heap is public so the global allocator shim can access it,
    /// but any access still requires locking.
    pub heap: crate::spinlock::StaticIrqSpinLock<KernelHeap>
}

impl LockedHeap {
    /// Constructs a `LockedHeap` wrapping a new, empty `KernelHeap`.
    ///
    /// `const fn` so this can initialize a `static` variable at compile time.
    pub const fn new(base: u64, max_size: u64) -> Self {
        Self {
            heap: crate::spinlock::StaticIrqSpinLock::new(
                KernelHeap::new(base, max_size)
            )
        }
    }
}

// =============================================================================
// GlobalHeapAllocator - Rust GlobalAlloc integration
// =============================================================================

/// A zero-sized type that implements Rust's `GlobalAlloc` trait.
///
/// By registering this as the `#[global_allocator]`, all Rust allocations in
/// the kernel (Box, Vec, String, Arc, etc.) are routed through `KERNEL_HEAP`.
///
/// The struct itself holds no state; all state lives in the `KERNEL_HEAP`
/// static, accessed via `globals::KERNEL_HEAP`.
pub struct GlobalHeapAllocator;

unsafe impl GlobalAlloc for GlobalHeapAllocator {
    /// Called by Rust's runtime whenever memory is allocated.
    ///
    /// It locks the heap, runs the first-fit allocator, and returns the
    /// received pointer. If the heap needs to grow to satisfy the request,
    /// `grow` is called automatically inside `allocate`.
    /// 
    /// # Arguments
    /// 
    /// * `layout`  -  Memory block layout to satisfy with this allocation.
    ///
    /// # Returns
    /// 
    /// Returns the pointer to allocated dynamic memory.
    /// 
    /// # Safety
    /// 
    /// The caller (Rust's allocator infrastructure) guarantees that `layout`
    /// has non-zero size and a power-of-two alignment.
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        unsafe { KERNEL_HEAP.heap.lock().allocate(layout) }
    }

    /// Called by Rust's runtime whenever memory is freed.
    ///
    /// It locks the heap and returns the block to the free list. The passed
    /// `layout` is actually ignored because Rust's `GlobalAlloc` contract
    /// requires the caller to pass the same `layout` that was used to allocate
    /// the pointer. But, our implementation stores `total_size` and
    /// `data_offset` in the block header, so we can recover all the information
    /// we need from `ptr` alone. The `layout` argument is therefore unused, but
    /// it has to be accepted to satisfy the trait.
    /// 
    /// # Arguments
    /// 
    /// * `ptr`     -  Pointer to the memory allocation to be freed.
    /// * `layout`  -  Layout that was used during allocation, but is ignored.
    ///
    /// # Safety
    /// 
    /// `ptr` must have been returned by `alloc` with a compatible `layout`
    /// and must not have been freed already.
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // Explicitly ignore layout, as all info is in the block header
        let _ = layout;
        unsafe { KERNEL_HEAP.heap.lock().deallocate(ptr) }
    }
}
