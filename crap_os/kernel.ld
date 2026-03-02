ENTRY(_start)

SECTIONS {
    . = 0x100000;
    __kernel_start = .;
    
    .text : ALIGN(4K) {
        KEEP(*(.text._start))
        *(.text .text.*)
    }
    
    .rodata : ALIGN(4K) {
        *(.rodata .rodata.*)
    }
    
    .data : ALIGN(4K) {
        *(.data .data.*)
    }
    
    .bss : ALIGN(4K) {
        *(.bss .bss.*)
        *(COMMON)
    }

    __kernel_end = .;
}
