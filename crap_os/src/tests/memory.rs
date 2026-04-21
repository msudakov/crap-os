use crate::globals;
//use crate::sprintln;
//use crate::fbprintln;
use crate::memory_manager::{PRESENT, WRITABLE};

macro_rules! mm {
    () => {
        globals::MEMORY_MANAGER.lock().as_mut().unwrap()
    };
}

#[allow(dead_code)]
pub fn test_memory_manager() {
    unsafe {
        let virt1: u64 = 0xFFFF_C000_0000_0000;
        let virt2: u64 = 0xFFFF_C000_0001_0000;
        let virt3: u64 = 0xFFFF_A000_0002_0000;
        let virt4: u64 = 0xFFFF_D000_0000_0000;
        // =====================================================================
        // Test 1: Basic map and resolve
        // Maps a physical page to a virtual address and verifies that
        // get_physical_addr resolves it correctly.
        // =====================================================================
        let phys1 = mm!().alloc_page().expect("Test 1: alloc failed");
        mm!().map_page(virt1, phys1, PRESENT | WRITABLE);

        match mm!().get_physical_addr(virt1) {
            Some(resolved) => assert!(resolved == phys1,
                "Test 1 FAILED: expected {:#x}, got {:#x}", phys1, resolved),
            None => panic!("Test 1 FAILED: address not mapped"),
        }
        //sprintln!("Test 1 passed: map_page + get_physical_addr");
        //fbprintln!("Test 1 passed: map_page + get_physical_addr");

        // =====================================================================
        // Test 2: unmap_and_free_page
        // Verifies that after unmapping, the virtual address resolves to None,
        // and that the freed physical page can be re-allocated.
        // =====================================================================
        mm!().unmap_and_free_page(virt1);

        assert!(mm!().get_physical_addr(virt1).is_none(),
            "Test 2 FAILED: address should be unmapped after unmap_and_free_page");

        // The freed page should be at the top of the free list and come back on
        // the next alloc
        let reallocated = mm!().alloc_page().expect("Test 2: realloc failed");
        assert!(reallocated == phys1,
            "Test 2 FAILED: expected freed page {:#x} to be reallocated, got {:#x}",
            phys1, reallocated);
        mm!().free_page(reallocated);  // Clean up
        //sprintln!("Test 2 passed: unmap_and_free_page recycles physical frame");
        //fbprintln!("Test 2 passed: unmap_and_free_page recycles physical frame");

        // =====================================================================
        // Test 3: unmap_page does not free the physical frame
        // Verifies the separation between unmapping and freeing. After
        // unmap_page the virtual address should be gone, but the physical frame
        // should not be back in the free list yet.
        // =====================================================================
        let phys3 = mm!().alloc_page().expect("Test 3: alloc failed");
        mm!().map_page(virt2, phys3, PRESENT | WRITABLE);
        mm!().unmap_page(virt2);

        assert!(mm!().get_physical_addr(virt2).is_none(),
            "Test 3 FAILED: address should be unmapped");

        // Allocate a fresh page - it should not be phys3, since we never freed it
        let next_alloc = mm!().alloc_page().expect("Test 3: second alloc failed");
        assert!(next_alloc != phys3,
            "Test 3 FAILED: physical frame was freed when it should not have been");
        mm!().free_page(next_alloc);  // Clean up
        mm!().free_page(phys3);       // Now manually free the orphaned frame
        //sprintln!("Test 3 passed: unmap_page does not free physical frame");
        //fbprintln!("Test 3 passed: unmap_page does not free physical frame");

        // =====================================================================
        // Test 4: Unmapping an already-unmapped address returns false
        // =====================================================================
        let result = mm!().unmap_page(virt3);
        assert!(!result, "Test 4 FAILED: expected false for unmapped address");
        //sprintln!("Test 4 passed: unmap_page returns false for unmapped address");
        //fbprintln!("Test 4 passed: unmap_page returns false for unmapped address");

        // =====================================================================
        // Test 5: Intermediate page table reclamation
        // Maps a single page, unmaps it, and verifies that the physical frames
        // used for the PT, PD, and PDPT are returned to the PMM.
        // =====================================================================
        let free_before = mm!().free_page_count();
        let phys5 = mm!().alloc_page().expect("Test 5: alloc failed");
        mm!().map_page(virt4, phys5, PRESENT | WRITABLE);

        // Mapping consumed: 1 data page + 1 PT + 1 PD + 1 PDPT = 4 pages
        assert!(mm!().free_page_count() == free_before - 4,
            "Test 5 FAILED: expected 4 pages consumed after map");

        mm!().unmap_and_free_page(virt4);

        // All 4 should be back
        assert!(mm!().free_page_count() == free_before,
            "Test 5 FAILED: expected full reclamation after unmap, got {} free (expected {})",
            mm!().free_page_count(), free_before);
        //sprintln!("Test 5 passed: intermediate page tables reclaimed on unmap");
        //fbprintln!("Test 5 passed: intermediate page tables reclaimed on unmap");
    }
}

#[allow(dead_code)]
pub fn test_heap_allocator() {
    use alloc::vec::Vec;
    use alloc::boxed::Box;
    use alloc::string::String;

    // =========================================================================
    // Test 1: Basic Box allocation and deallocation
    // Allocates a Box<u64>, writes a value, reads it back, then drops it.
    // If the allocator is broken at a fundamental level this will fault.
    // =========================================================================
    let b = Box::new(0xDEAD_BEEF_u64);
    assert!(*b == 0xDEAD_BEEF_u64, "Test 1 FAILED: Box value mismatch");
    drop(b);
    //sprintln!("Test 1 passed: basic Box alloc/dealloc");
    //fbprintln!("Test 1 passed: basic Box alloc/dealloc");

    // =========================================================================
    // Test 2: Vec growth
    // Pushes enough elements to force Vec to reallocate internally several
    // times, exercising both alloc and dealloc on the same type.
    // =========================================================================
    let mut v: Vec<u64> = Vec::new();
    for i in 0..1024_u64 {
        v.push(i);
    }
    for i in 0..1024_u64 {
        assert!(v[i as usize] == i, "Test 2 FAILED: Vec value mismatch at {}", i);
    }
    drop(v);
    //sprintln!("Test 2 passed: Vec growth and dealloc");
    //fbprintln!("Test 2 passed: Vec growth and dealloc");

    // =========================================================================
    // Test 3: String allocation
    // =========================================================================
    let mut s = String::new();
    s.push_str("hello from the kernel heap");
    assert!(s == "hello from the kernel heap", "Test 3 FAILED: String mismatch");
    drop(s);
    //sprintln!("Test 3 passed: String alloc/dealloc");
    //fbprintln!("Test 3 passed: String alloc/dealloc");

    // =========================================================================
    // Test 4: Alignment
    // Allocates types with non-trivial alignment requirements and checks that
    // the returned pointer satisfies them.
    // =========================================================================
    #[repr(align(64))]
    struct Aligned64 { val: u64 }

    let a = Box::new(Aligned64 { val: 0xCAFE });
    let ptr = &*a as *const Aligned64 as usize;
    assert!(ptr % 64 == 0,
        "Test 4 FAILED: pointer {:#x} is not 64-byte aligned", ptr);
    assert!(a.val == 0xCAFE, "Test 4 FAILED: value corrupted");
    drop(a);
    //sprintln!("Test 4 passed: alignment");
    //fbprintln!("Test 4 passed: alignment");

    // =========================================================================
    // Test 5: Coalescing - repeated alloc/free of same size should not
    // fragment the heap. We measure free pages before and after a long
    // alloc/free cycle. Without coalescing the free list would splinter into
    // unusable small blocks and the heap would need to keep growing.
    // =========================================================================
    let free_before = globals::MEMORY_MANAGER.lock()
        .as_ref().unwrap().free_page_count();

    for _ in 0..512 {
        let b = Box::new([0u8; 256]);
        drop(b);
    }

    let free_after = globals::MEMORY_MANAGER.lock()
        .as_ref().unwrap().free_page_count();

    // With coalescing the heap should not have needed to grow at all, so the
    // PMM free page count should be identical before and after.
    assert!(free_before == free_after,
        "Test 5 FAILED: heap grew during alloc/free cycle \
         (before={}, after={}), coalescing may be broken",
        free_before, free_after);
    //sprintln!("Test 5 passed: coalescing");
    //fbprintln!("Test 5 passed: coalescing");

    // =========================================================================
    // Test 6: Multiple live allocations with distinct values
    // Allocates several boxes simultaneously, verifies none overlap or
    // corrupt each other, then drops them all.
    // =========================================================================
    let b1 = Box::new(0x1111_u64);
    let b2 = Box::new(0x2222_u64);
    let b3 = Box::new(0x3333_u64);
    let b4 = Box::new(0x4444_u64);

    // Check no two boxes share an address
    let p1 = &*b1 as *const u64 as usize;
    let p2 = &*b2 as *const u64 as usize;
    let p3 = &*b3 as *const u64 as usize;
    let p4 = &*b4 as *const u64 as usize;
    assert!(p1 != p2 && p1 != p3 && p1 != p4
         && p2 != p3 && p2 != p4 && p3 != p4,
        "Test 6 FAILED: two allocations returned the same address");

    // Check values are still intact after all four are live simultaneously
    assert!(*b1 == 0x1111, "Test 6 FAILED: b1 corrupted");
    assert!(*b2 == 0x2222, "Test 6 FAILED: b2 corrupted");
    assert!(*b3 == 0x3333, "Test 6 FAILED: b3 corrupted");
    assert!(*b4 == 0x4444, "Test 6 FAILED: b4 corrupted");
    drop(b1); drop(b2); drop(b3); drop(b4);
    //sprintln!("Test 6 passed: multiple simultaneous allocations");
    //fbprintln!("Test 6 passed: multiple simultaneous allocations");

    // =========================================================================
    // Test 7: Heap growth
    // Allocates more memory than the initial committed pages to force at least
    // one call to grow(). Each box is kept live to prevent the heap from
    // reusing freed blocks.
    // =========================================================================
    let mut boxes: Vec<Box<[u8; 1024]>> = Vec::new();
    // 128 * 1024 = 128 KB, more than the default 16-page (64 KB) init
    for _ in 0..128 {
        boxes.push(Box::new([0xAB_u8; 1024]));
    }
    for b in &boxes {
        assert!(b.iter().all(|&x| x == 0xAB),
            "Test 7 FAILED: value corrupted after heap growth");
    }
    drop(boxes);
    //sprintln!("Test 7 passed: heap growth");
    //fbprintln!("Test 7 passed: heap growth");
}
