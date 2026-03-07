ENTRY(_start)

KERNEL_PHYS_BASE   = 0x100000;
KERNEL_VIRT_BASE   = 0xFFFFFFFF80000000;
KERNEL_VIRT_OFFSET = KERNEL_VIRT_BASE - KERNEL_PHYS_BASE;

SECTIONS {
    . = KERNEL_VIRT_BASE + KERNEL_PHYS_BASE;
    __kernel_virt_start = .;

    .text ALIGN(4K) : AT(ADDR(.text) - KERNEL_VIRT_OFFSET) {
        KEEP(*(.text._start))
        *(.text .text.*)
    }

    .rodata ALIGN(4K) : AT(ADDR(.rodata) - KERNEL_VIRT_OFFSET) {
        *(.rodata .rodata.*)
    }

    .data ALIGN(4K) : AT(ADDR(.data) - KERNEL_VIRT_OFFSET) {
        *(.data .data.*)
    }

    .bss ALIGN(4K) : AT(ADDR(.bss) - KERNEL_VIRT_OFFSET) {
        __bss_start = .;
        *(.bss .bss.*)
        *(COMMON)
        __bss_end = .;
    }

    __kernel_virt_end = .;

    /* Physical addresses */
    __kernel_phys_start = KERNEL_PHYS_BASE;
    __kernel_phys_end = __kernel_virt_end - KERNEL_VIRT_BASE;
}
