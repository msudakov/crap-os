#include <efi.h>
#include <efilib.h>


#define COM1_PORT 0x3F8  // COM1 port base address
#define KERNEL_STACK_PAGES 32  // 32 * 4096 = 128 KiB; lower it to 64 later on
#define KERNEL_STACK_SIZE  (KERNEL_STACK_PAGES * 0x1000)

// Debug message levels
enum DebugLevel
{
    DEBUG = 1,
    INFO = 2,
    WARNING = 3,
    ERROR = 4,
    CRITICAL = 5
};

// Framebuffer information structure to pass to kernel
typedef struct {
    UINT64 framebuffer_addr;
    UINT32 framebuffer_width;
    UINT32 framebuffer_height;
    UINT32 framebuffer_pitch;
    UINT32 framebuffer_bpp;
} FramebufferInfo;

// Memory map information structure to pass to kernel
typedef struct {
    UINT64 memory_map_addr;
    UINT64 memory_map_size;
    UINT64 descriptor_size;
    UINT32 descriptor_ver;
    UINT64 kernel_load_addr;
    UINT64 kernel_image_size;
    UINT64 stack_base_addr;
    UINT64 stack_size;
} MemoryMapInfo;

// Boot information structure to pass to kernel
typedef struct {
    UINT64 magic;
    FramebufferInfo* framebuffer_info;
    MemoryMapInfo* memory_map_info;
} BootInfo;

const enum DebugLevel DEBUG_LEVEL = INFO;  // Global debug message level

// Kernel entry point that receives boot info
typedef void (*kernel_entry)(BootInfo *);

/**
  Reads one byte of data from a specified I/O port address.

  @param[in]  port   Serial port base address.
  
  @return The UINT8 value received from the specified serial port.
**/
static inline unsigned char inb(unsigned short port) {
    unsigned char ret;
    __asm__ volatile ( "inb %1, %0" : "=a"(ret) : "Nd"(port) );
    return ret;
}

/**
  Writes one byte of data to a specified I/O port address.

  @param[in]  port    Serial port base address.
  @param[in]  value   Byte value to write to serial port.
**/
static inline void outb(unsigned short port, unsigned char value) {
    __asm__ volatile ( "outb %0, %1" : : "a"(value), "Nd"(port) );
}

/**
  Initializes serial port for debugging purposes.
**/
void init_serial() {
    outb(COM1_PORT + 1, 0x00);    // Disable interrupts on COM1
    outb(COM1_PORT + 3, 0x80);    // Enable DLAB (set baud rate divisor)
    outb(COM1_PORT + 0, 0x03);    // Set divisor 3 (lo byte) 38400 baud
    outb(COM1_PORT + 1, 0x00);    //               (hi byte)
    outb(COM1_PORT + 3, 0x03);    // 8 bits, no parity, one stop bit
    outb(COM1_PORT + 2, 0xC7);    // Enable FIFO, clear them (14-byte threshold)
    outb(COM1_PORT + 4, 0x0B);    // IRQs enabled, RTS/DSR set
}

/**
  Writes a byte to COM1 serial port.

  @param[in]  c   Byte to write.
**/
void serial_write_byte(char c) {
    // Wait for transmit buffer to be empty
    while ((inb(COM1_PORT + 5) & 0x20) == 0);

    // Send the byte
    outb(COM1_PORT, c);
}

/**
  Counts the number of characters in a string (its length), until it encounters
    a null terminator.

  @param[in]  str   Message string to write.

  @return The int value of the message length.
**/
int get_strlen(const char* str) {
    int len = 0;
    while (*str++ != '\0') {
        len++;
    }

    return len;
}

/**
  Writes a given message to COM1 serial port.

  @param[in]  str   Message string to write.
**/
void serial_write(const char* str) {
    int strLen = get_strlen(str);

    for (int i = 0; i < strLen; i++) {
        serial_write_byte(str[i]);
    }
}

/**
  Prints a debug message to console if its debug level is equal to or greater
    than the global debug message level.

  @param[in]  debug_level   Specified debug level of the given message.
  @param[in]  message       Debug message to print.
**/
void print_debug(enum DebugLevel debug_level, CHAR16* message) {
    if (debug_level < DEBUG_LEVEL || message == NULL) {
        return;
    }

    Print(message);
}

/**
  UEFI bootloader entry point routine.
**/
EFI_STATUS
EFIAPI
efi_main(EFI_HANDLE ImageHandle, EFI_SYSTEM_TABLE *SystemTable) {
    EFI_STATUS Status;
    EFI_INPUT_KEY Key;
    EFI_FILE_PROTOCOL *Root;
    EFI_FILE_PROTOCOL *KernelFile;
    EFI_LOADED_IMAGE_PROTOCOL *LoadedImage;
    EFI_SIMPLE_FILE_SYSTEM_PROTOCOL *FileSystem;
    EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop;
    UINTN KernelSize;
    EFI_PHYSICAL_ADDRESS KernelBuffer;
    EFI_PHYSICAL_ADDRESS KernelStackBase = 0;
    EFI_PHYSICAL_ADDRESS KernelStackTop;
    UINTN MapKey;
    UINTN DescriptorSize;
    UINT32 DescriptorVersion;
    UINTN MemoryMapSize = 0;
    EFI_MEMORY_DESCRIPTOR *MemoryMap = NULL;
    FramebufferInfo *FramebufferInfoStruct;
    MemoryMapInfo *MemoryMapInfoStruct;
    BootInfo *BootInfoStruct;
    
    InitializeLib(ImageHandle, SystemTable);  // Initialize the GNU-EFI library
    init_serial();  // Initialize serial port for debugging

    print_debug(INFO, L"[+] Hello from CrapLoader!\n\r");
    //uefi_call_wrapper(SystemTable->BootServices->Stall, 1, 1000000);

    // Allocate boot info structures
    FramebufferInfoStruct = AllocatePool(sizeof(FramebufferInfo));
    MemoryMapInfoStruct = AllocatePool(sizeof(MemoryMapInfo));
    BootInfoStruct = AllocatePool(sizeof(BootInfo));
    BootInfoStruct->magic = 0xDEADBEEFB007CAFE;

    print_debug(DEBUG, L"[*] Getting Graphics Output Protocol...\n\r");
    Status = uefi_call_wrapper(SystemTable->BootServices->LocateProtocol, 3,
        &gEfiGraphicsOutputProtocolGuid, NULL, (void**)&Gop);
    
    if (EFI_ERROR(Status)) {
        Print(L"[-] WARNING: Could not get GOP (status: %r)\n\r", Status);
        print_debug(WARNING, L"[*] Will try VGA text mode fallback\n\r");
        FramebufferInfoStruct->framebuffer_addr = 0xB8000;  // VGA text buffer
        FramebufferInfoStruct->framebuffer_width = 80;
        FramebufferInfoStruct->framebuffer_height = 25;
        FramebufferInfoStruct->framebuffer_pitch = 160;
        FramebufferInfoStruct->framebuffer_bpp = 16;
    }
    else {
        // Get framebuffer info
        FramebufferInfoStruct->framebuffer_addr = Gop->Mode->FrameBufferBase;
        FramebufferInfoStruct->framebuffer_width = Gop->Mode->Info->HorizontalResolution;
        FramebufferInfoStruct->framebuffer_height = Gop->Mode->Info->VerticalResolution;
        FramebufferInfoStruct->framebuffer_pitch = Gop->Mode->Info->PixelsPerScanLine * 4;
        FramebufferInfoStruct->framebuffer_bpp = 32;
        print_debug(DEBUG, L"[+] Located Graphics Output Protocol\n\r");
    }
    BootInfoStruct->framebuffer_info = FramebufferInfoStruct;
    //uefi_call_wrapper(SystemTable->BootServices->Stall, 1, 1000000);

    Status = uefi_call_wrapper(SystemTable->BootServices->HandleProtocol, 3,
        ImageHandle, &gEfiLoadedImageProtocolGuid, (void**)&LoadedImage);
    if (EFI_ERROR(Status)) {
        print_debug(ERROR, L"[-] Failed to get LoadedImage protocol\n\r");
        Print(L"[!FATAL!] Press any key to exit...\n\r");
        while (uefi_call_wrapper(SystemTable->ConIn->ReadKeyStroke, 2,
            SystemTable->ConIn, &Key) != EFI_SUCCESS);
        return Status;
    }
    print_debug(DEBUG, L"[+] Located LoadedImage protocol\n\r");

    Status = uefi_call_wrapper(SystemTable->BootServices->HandleProtocol, 3,
        LoadedImage->DeviceHandle, &gEfiSimpleFileSystemProtocolGuid,
        (void**)&FileSystem);
    if (EFI_ERROR(Status)) {
        print_debug(ERROR, L"[-] Failed to get FileSystem protocol\n\r");
        Print(L"[!FATAL!] Press any key to exit...\n\r");
        while (uefi_call_wrapper(SystemTable->ConIn->ReadKeyStroke, 2,
            SystemTable->ConIn, &Key) != EFI_SUCCESS);
        return Status;
    }
    print_debug(DEBUG, L"[+] Located FileSystem protocol\n\r");

    Status = uefi_call_wrapper(FileSystem->OpenVolume, 2, FileSystem, &Root);
    if (EFI_ERROR(Status)) {
        print_debug(ERROR, L"[-] Failed to open root volume\n\r");
        Print(L"[!FATAL!] Press any key to exit...\n\r");
        while (uefi_call_wrapper(SystemTable->ConIn->ReadKeyStroke, 2,
            SystemTable->ConIn, &Key) != EFI_SUCCESS);
        return Status;
    }
    print_debug(DEBUG, L"[+] Opened the root directory\n\r");

    Status = uefi_call_wrapper(Root->Open, 5, Root, &KernelFile, L"kernel.bin",
        EFI_FILE_MODE_READ, 0);
    if (EFI_ERROR(Status)) {
        print_debug(ERROR, L"[-] Failed to open kernel.bin\n\r");
        Print(L"[!FATAL!] Press any key to exit...\n\r");
        while (uefi_call_wrapper(SystemTable->ConIn->ReadKeyStroke, 2,
            SystemTable->ConIn, &Key) != EFI_SUCCESS);
        return Status;
    }
    print_debug(DEBUG, L"[+] Opened the kernel file\n\r");
    
    // Get the kernel file size
    EFI_FILE_INFO *FileInfo;
    UINTN FileInfoSize = SIZE_OF_EFI_FILE_INFO + 200;
    FileInfo = AllocatePool(FileInfoSize);
    Status = uefi_call_wrapper(KernelFile->GetInfo, 4, KernelFile,
        &gEfiFileInfoGuid, &FileInfoSize, FileInfo);
    if (EFI_ERROR(Status)) {
        print_debug(ERROR, L"[-] Failed to get kernel file info\n\r");
        Print(L"[!FATAL!] Press any key to exit...\n\r");
        while (uefi_call_wrapper(SystemTable->ConIn->ReadKeyStroke, 2,
            SystemTable->ConIn, &Key) != EFI_SUCCESS);
        return Status;
    }
    KernelSize = FileInfo->FileSize;
    FreePool(FileInfo);
    Print(L"[+] Kernel file size: %d bytes\n\r", KernelSize);
    //uefi_call_wrapper(SystemTable->BootServices->Stall, 1, 1000000);
    
    // Allocate memory for the kernel
    KernelBuffer = 0x100000;  // Load at 1MB
    Status = uefi_call_wrapper(SystemTable->BootServices->AllocatePages, 4,
        AllocateAddress, EfiLoaderCode, (KernelSize + 0xFFF) / 0x1000,
        &KernelBuffer);
    if (EFI_ERROR(Status)) {
        print_debug(ERROR, L"[-] Failed to allocate memory for kernel\n\r");
        Print(L"[!FATAL!] Press any key to exit...\n\r");
        while (uefi_call_wrapper(SystemTable->ConIn->ReadKeyStroke, 2,
            SystemTable->ConIn, &Key) != EFI_SUCCESS);
        return Status;
    }
    print_debug(DEBUG, L"[+] Allocated memory for kernel\n\r");
    
    // Read the kernel into memory
    Status = uefi_call_wrapper(KernelFile->Read, 3, KernelFile, &KernelSize,
        (void*)KernelBuffer);
    if (EFI_ERROR(Status)) {
        print_debug(ERROR, L"[-] Failed to read kernel\n\r");
        Print(L"[!FATAL!] Press any key to exit...\n\r");
        while (uefi_call_wrapper(SystemTable->ConIn->ReadKeyStroke, 2,
            SystemTable->ConIn, &Key) != EFI_SUCCESS);
        return Status;
    }
    print_debug(DEBUG, L"[+] Successfully read kernel into memory\n\r");
    
    // Close the kernel file
    uefi_call_wrapper(KernelFile->Close, 1, KernelFile);
    uefi_call_wrapper(Root->Close, 1, Root);

    Print(L"[+] Kernel loaded at 0x%x\n\r", KernelBuffer);
    //uefi_call_wrapper(SystemTable->BootServices->Stall, 1, 1000000);

    // Allocate initial kernel stack
    Status = uefi_call_wrapper(SystemTable->BootServices->AllocatePages, 4,
        AllocateAnyPages, EfiLoaderData, KERNEL_STACK_PAGES, &KernelStackBase);
    if (EFI_ERROR(Status)) {
        print_debug(ERROR, L"[-] Failed to allocate kernel stack\n\r");
        Print(L"[!FATAL!] Press any key to exit...\n\r");
        while (uefi_call_wrapper(SystemTable->ConIn->ReadKeyStroke, 2,
            SystemTable->ConIn, &Key) != EFI_SUCCESS);
        return Status;
    }
    
    // Zero out allocated mmory just in case
    SetMem((void*)KernelStackBase, KERNEL_STACK_SIZE, 0);

    /*
        Stack grows downward, so RSP should start at the top of the allocation.
        We need to subtract 8 bytes to keep the RSP 16-byte aligned on entry.
        This is because the x86-64 ABI requires RSP % 16 == 8 just before a call
        instruction, but since we're jumping directly rather than using CALL,
        we want RSP % 16 == 0 at _start).
    */
    KernelStackTop = KernelStackBase + KERNEL_STACK_SIZE - 8;

    MemoryMapInfoStruct->stack_base_addr = KernelStackBase;
    MemoryMapInfoStruct->stack_size = KERNEL_STACK_SIZE;

    print_debug(DEBUG, L"[+] Allocated kernel stack\n\r");
    Print(L"[+] Stack: base=0x%lx  top=0x%lx  size=%d bytes\n\r",
        KernelStackBase, KernelStackTop, KERNEL_STACK_SIZE);
    //print_debug(DEBUG, L"[+] Successfully allocated kernel stack memory\n\r");




    Print(L"[*] Crap loader is starting to load CrapOS\n\r");
    Print(L"[!] Press any key to cancel. It's about to hit the fan...\n\r");
    //uefi_call_wrapper(SystemTable->BootServices->Stall, 1, 1000000);
    Print(L"[!] Start in 3 seconds...\n\r");
    //uefi_call_wrapper(SystemTable->BootServices->Stall, 1, 1000000);
    Print(L"[!] Start in 2 seconds...\n\r");
    //uefi_call_wrapper(SystemTable->BootServices->Stall, 1, 1000000);
    Print(L"[!] Start in 1 second...\n\r");
    //uefi_call_wrapper(SystemTable->BootServices->Stall, 1, 1000000);

    print_debug(INFO,
        L"[*] Getting memory map and exiting boot services...\n\r");

    /*
        We have to call GetMemoryMap twice. This first call gets the
        initial size of the memory map structure. It may or may not change
        before the second and final call.
    */
    Status = uefi_call_wrapper(SystemTable->BootServices->GetMemoryMap, 5,
        &MemoryMapSize, NULL, &MapKey, &DescriptorSize, &DescriptorVersion);

    /*
        Allocate memory map buffer, with some extra space. It is possible for
        the memory map to grow in size between the initial call and the second
        call. To be safe, we add the size of 4 additional memory descriptors,
        which should be enough in all cases.
    */
    MemoryMapSize += 4 * DescriptorSize;
    MemoryMap = AllocatePool(MemoryMapSize);

    // Finalize memory map struct fields
    MemoryMapInfoStruct->descriptor_size = DescriptorSize;
    MemoryMapInfoStruct->descriptor_ver = DescriptorVersion;
    MemoryMapInfoStruct->memory_map_addr = (UINT64)MemoryMap;
    MemoryMapInfoStruct->kernel_load_addr = KernelBuffer;
    MemoryMapInfoStruct->kernel_image_size = KernelSize;

    BootInfoStruct->memory_map_info = MemoryMapInfoStruct;
    

    /*
        Get memory map and exit boot services.

        Before we transfer execution into the OS kernel by jumping to its
        exported entry routine, we need to exit the UEFI boot services. It's
        important to understand that once we do that, all hand-holding ends;
        UEFI will at that point unload its helper libraries that we've been
        using and will reclaim that memory.

        For the call to ExitBootServices to succeed, the MapKey parameter must
        still be valid since the last call to GetMemoryMap. This is why these
        two calls must be executed back-to-back. No other UEFI-assisted
        operation can take place in between, as that will almost certainly
        modify the memory region mapping and invalidate the MapKey received
        from the last call to GetMemoryMap.

        It is fine that the buffer passed to GetMemoryMap second call is larger
        than it needs to be because the routine will return the actual size in
        one of its [out] arguments. This is the size that we'll insert into the
        boot structure passed to the kernel.
    */
    UINTN retries = 0;
    while (retries < 10) {
        UINTN CurrentMapSize = MemoryMapSize;
        Status = uefi_call_wrapper(SystemTable->BootServices->GetMemoryMap, 5,
            &CurrentMapSize, MemoryMap, &MapKey, &DescriptorSize,
            &DescriptorVersion);

        if (EFI_ERROR(Status)) {
            retries++;
            continue;
        }

        Status = uefi_call_wrapper(SystemTable->BootServices->ExitBootServices,
            2, ImageHandle, MapKey);

        /*
            It is possible for the call to ExitBootServices to still fail even
            though we called the two routines back-to-back. Although rare, it
            can happen. The 10-try loop ensures that the only time it will not
            eventually succeed is if there is an actual problem with our code,
            and not due to a machine's normal operation and bootstrap protocol
            sequence.
        */
        if (Status == EFI_SUCCESS) {
            // Fill in the actual structure size in the boot info
            MemoryMapInfoStruct->memory_map_size = CurrentMapSize;
            break;
        }
        
        retries++;
    }

    /*
        If the above did not succeed and we weren't able to get the memory map
        and exit boot services, there's nothing more to do than crash here and
        debug the reason.
    */
    if (EFI_ERROR(Status)) {
        print_debug(ERROR, L"[-] Failed to exit boot services\n\r");
        Print(L"[!FATAL!] Press any key to exit...\n\r");
        while (uefi_call_wrapper(SystemTable->ConIn->ReadKeyStroke, 2,
            SystemTable->ConIn, &Key) != EFI_SUCCESS);
        return Status;
    }

    /*
        Send a final debug string to serial port from the bootloader; this is a
        point of no-return. What follows is the execution transfer to the OS.
    */
    serial_write("[*] Just before jumping to kernel...\n\r");

    /*
        Assembly stub to switch to kernel stack and jump to entry point. This
        atomically sets RSP and RDI (the first System V argument register,
        which receives boot_info struct) without ever using the old UEFI stack
        again after the mov. This intentionally clobbers RSP and RDI; there will
        be no return.

        The reason we want to jmp and not call the function pointer is that
        calling KernelEntry(BootInfoStruct) through a C function pointer still
        uses the old UEFI stack (to push a return address). Using jmp means
        we're fully committed to the new stack from the very first instruction
        of the kernel.
    */
    __asm__ __volatile__ (
        "mov %0, %%rsp\n\t"     // Switch to the kernel stack
        "mov %1, %%rdi\n\t"     // First arg: boot_info pointer (System V ABI)
        "xor %%rbp, %%rbp\n\t"  // Clear frame pointer (no caller frame to unwind to)
        "jmp *%2\n\t"           // jump (not call) to _start; never returns
        :
        : "r"((UINT64)KernelStackTop),
        "r"((UINT64)BootInfoStruct),
        "r"((UINT64)KernelBuffer)
        : "memory"
    );

    // Will never reach here
    while(1) {
        __asm__ __volatile__("hlt");
    }

    // Will never reach here
    return EFI_SUCCESS;
}
