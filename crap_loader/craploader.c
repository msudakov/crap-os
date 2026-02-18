#include <efi.h>
#include <efilib.h>


// Boot information structure to pass to kernel
typedef struct {
    UINT64 framebuffer_addr;
    UINT32 framebuffer_width;
    UINT32 framebuffer_height;
    UINT32 framebuffer_pitch;
    UINT32 framebuffer_bpp;
} BootInfo;

// Kernel entry point that receives boot info
typedef void (*kernel_entry)(BootInfo *);

EFI_STATUS
EFIAPI
efi_main(EFI_HANDLE ImageHandle, EFI_SYSTEM_TABLE *SystemTable)
{
    EFI_STATUS Status;
    EFI_FILE_PROTOCOL *Root;
    EFI_FILE_PROTOCOL *KernelFile;
    EFI_LOADED_IMAGE_PROTOCOL *LoadedImage;
    EFI_SIMPLE_FILE_SYSTEM_PROTOCOL *FileSystem;
    EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop;
    UINTN KernelSize;
    EFI_PHYSICAL_ADDRESS KernelBuffer;
    kernel_entry EntryPoint;
    UINTN MapKey;
    UINTN DescriptorSize;
    UINT32 DescriptorVersion;
    UINTN MemoryMapSize = 0;
    EFI_MEMORY_DESCRIPTOR *MemoryMap = NULL;
    BootInfo *BootInfoStruct;
    
    // Initialize the GNU-EFI library
    InitializeLib(ImageHandle, SystemTable);
    
    Print(L"[+] Hello from CrapLoader!\n\r");
    uefi_call_wrapper(SystemTable->BootServices->Stall, 1, 1000000);
    
    // Allocate boot info structure
    BootInfoStruct = AllocatePool(sizeof(BootInfo));
    
    // Get GOP (Graphics Output Protocol)
    Print(L"[*] Getting Graphics Output Protocol...\n\r");
    Status = uefi_call_wrapper(SystemTable->BootServices->LocateProtocol, 3,
        &gEfiGraphicsOutputProtocolGuid, NULL, (void**)&Gop);
    
    if (EFI_ERROR(Status)) {
        Print(L"[-] WARNING: Could not get GOP (status: %r)\n\r", Status);
        Print(L"[*] Will try VGA text mode fallback\n\r");
        BootInfoStruct->framebuffer_addr = 0xB8000;  // VGA text buffer
        BootInfoStruct->framebuffer_width = 80;
        BootInfoStruct->framebuffer_height = 25;
        BootInfoStruct->framebuffer_pitch = 160;
        BootInfoStruct->framebuffer_bpp = 16;
    }
    else {
        // Get framebuffer info
        BootInfoStruct->framebuffer_addr = Gop->Mode->FrameBufferBase;
        BootInfoStruct->framebuffer_width = Gop->Mode->Info->HorizontalResolution;
        BootInfoStruct->framebuffer_height = Gop->Mode->Info->VerticalResolution;
        BootInfoStruct->framebuffer_pitch = Gop->Mode->Info->PixelsPerScanLine * 4;
        BootInfoStruct->framebuffer_bpp = 32;
        
        Print(L"[+] Located GOP\n\r");
    }
    uefi_call_wrapper(SystemTable->BootServices->Stall, 1, 1000000);
    
    // Get the loaded image protocol
    Status = uefi_call_wrapper(SystemTable->BootServices->HandleProtocol, 3,
        ImageHandle, &gEfiLoadedImageProtocolGuid, (void**)&LoadedImage);
    if (EFI_ERROR(Status)) {
        Print(L"[-] Failed to get LoadedImage protocol\n\r");
        return Status;
    }
    
    // Get the file system protocol
    Status = uefi_call_wrapper(SystemTable->BootServices->HandleProtocol, 3,
        LoadedImage->DeviceHandle, &gEfiSimpleFileSystemProtocolGuid,
        (void**)&FileSystem);
    if (EFI_ERROR(Status)) {
        Print(L"[-] Failed to get FileSystem protocol\n\r");
        return Status;
    }
    
    // Open the root directory
    Status = uefi_call_wrapper(FileSystem->OpenVolume, 2, FileSystem, &Root);
    if (EFI_ERROR(Status)) {
        Print(L"[-] Failed to open root volume\n\r");
        return Status;
    }
    
    // Open the kernel file
    Status = uefi_call_wrapper(Root->Open, 5, Root, &KernelFile, L"kernel.bin",
        EFI_FILE_MODE_READ, 0);
    if (EFI_ERROR(Status)) {
        Print(L"[-] Failed to open kernel.bin\n\r");
        return Status;
    }
    
    // Get the kernel file size
    EFI_FILE_INFO *FileInfo;
    UINTN FileInfoSize = SIZE_OF_EFI_FILE_INFO + 200;
    FileInfo = AllocatePool(FileInfoSize);
    Status = uefi_call_wrapper(KernelFile->GetInfo, 4, KernelFile,
        &gEfiFileInfoGuid, &FileInfoSize, FileInfo);
    if (EFI_ERROR(Status)) {
        Print(L"[-] Failed to get kernel file info\n\r");
        return Status;
    }
    KernelSize = FileInfo->FileSize;
    FreePool(FileInfo);
    
    Print(L"[+] Opened kernel; file size: %d bytes\n\r", KernelSize);
    uefi_call_wrapper(SystemTable->BootServices->Stall, 1, 1000000);
    
    // Allocate memory for the kernel
    KernelBuffer = 0x100000; // Load at 1MB
    Status = uefi_call_wrapper(SystemTable->BootServices->AllocatePages, 4,
                               AllocateAddress, EfiLoaderCode, 
                               (KernelSize + 0xFFF) / 0x1000, &KernelBuffer);
    if (EFI_ERROR(Status)) {
        Print(L"[-] Failed to allocate memory for kernel\n\r");
        return Status;
    }
    
    // Read the kernel into memory
    Status = uefi_call_wrapper(KernelFile->Read, 3, KernelFile, &KernelSize,
        (void*)KernelBuffer);
    if (EFI_ERROR(Status)) {
        Print(L"[-] Failed to read kernel\n\r");
        return Status;
    }
    
    // Close the file
    uefi_call_wrapper(KernelFile->Close, 1, KernelFile);
    uefi_call_wrapper(Root->Close, 1, Root);

    Print(L"[+] Kernel loaded at 0x%x\n\r", KernelBuffer);
    uefi_call_wrapper(SystemTable->BootServices->Stall, 1, 1000000);
    
    //EFI_INPUT_KEY Key;
    //while (uefi_call_wrapper(SystemTable->ConIn->ReadKeyStroke, 2,
    //  SystemTable->ConIn, &Key) != EFI_SUCCESS);
    Print(L"[*] Crap loader is starting to load CrapOS\n\r");
    Print(L"[!] Press any key to cancel. It's about to hit the fan...\n\r");
    uefi_call_wrapper(SystemTable->BootServices->Stall, 1, 1000000);
    Print(L"[!] Start in 3 seconds...\n\r");
    uefi_call_wrapper(SystemTable->BootServices->Stall, 1, 1000000);
    Print(L"[!] Start in 2 seconds...\n\r");
    uefi_call_wrapper(SystemTable->BootServices->Stall, 1, 1000000);
    Print(L"[!] Start in 1 second...\n\r");
    uefi_call_wrapper(SystemTable->BootServices->Stall, 1, 1000000);
    
    Print(L"[*] Exiting boot services...\n\r");
    
    // Get initial size
    MemoryMapSize = 0;
    Status = uefi_call_wrapper(SystemTable->BootServices->GetMemoryMap, 5,
        &MemoryMapSize, NULL, &MapKey, &DescriptorSize, &DescriptorVersion);
    
    // Allocate with extra space
    MemoryMapSize += 10 * DescriptorSize;
    MemoryMap = AllocatePool(MemoryMapSize);
    
    // Exit boot services
    UINTN Retries = 0;
    while (Retries < 10) {
        UINTN CurrentMapSize = MemoryMapSize;
        Status = uefi_call_wrapper(SystemTable->BootServices->GetMemoryMap, 5,
            &CurrentMapSize, MemoryMap, &MapKey, &DescriptorSize,
            &DescriptorVersion);
        
        if (EFI_ERROR(Status)) {
            Retries++;
            continue;
        }
        
        Status = uefi_call_wrapper(SystemTable->BootServices->ExitBootServices,
            2, ImageHandle, MapKey);
        
        if (Status == EFI_SUCCESS) {
            break;
        }
        
        Retries++;
    }
    
    if (EFI_ERROR(Status)) {
        Print(L"[-] Failed to exit boot services\n\r");
        return Status;
    }
    
    // Boot services exited! Jump to kernel with boot info
    EntryPoint = (kernel_entry)KernelBuffer;
    EntryPoint(BootInfoStruct);
    
    // Should never reach here
    while(1) {
        __asm__ __volatile__("hlt");
    }
    
    return EFI_SUCCESS;
}
