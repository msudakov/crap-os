ARCH            = x86_64
OBJS            = craploader.o
TARGET          = craploader.efi

# Compiler and linker flags
EFIINC          = /usr/include/efi
EFIINCS         = -I$(EFIINC) -I$(EFIINC)/$(ARCH) -I$(EFIINC)/protocol
LIB             = /usr/lib
EFILIB          = /usr/lib
EFI_CRT_OBJS    = $(EFILIB)/crt0-efi-$(ARCH).o
EFI_LDS         = $(EFILIB)/elf_$(ARCH)_efi.lds

CFLAGS          = $(EFIINCS) -fno-stack-protector -fpic \
                  -fshort-wchar -mno-red-zone -Wall -DEFI_FUNCTION_WRAPPER

LDFLAGS         = -nostdlib -znocombreloc -T $(EFI_LDS) -shared \
                  -Bsymbolic -L $(EFILIB) -L $(LIB) $(EFI_CRT_OBJS) 

.PHONY: all clean

all: $(TARGET)

craploader.so: $(OBJS)
	ld $(LDFLAGS) $(OBJS) -o $@ -lefi -lgnuefi

%.efi: %.so
	objcopy -j .text -j .sdata -j .data -j .dynamic \
		-j .dynsym  -j .rel -j .rela -j .reloc \
		--target=efi-app-$(ARCH) $^ $@

%.o: %.c
	gcc $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) craploader.so $(TARGET)
	rm -f boot.img boot.vmdk boot.vdi kernel.bin

# Create a bootable disk image
disk: $(TARGET)
	@if [ ! -f kernel.bin ]; then \
		echo "ERROR: kernel.bin not found!"; \
		echo "Please place your kernel binary as kernel.bin in this directory"; \
		exit 1; \
	fi
	@echo "Creating bootable disk image..."
	@bash create_bootable_image.sh

# Test with QEMU
test: disk
	@bash test_qemu.sh

help:
	@echo "Available targets:"
	@echo "  make          - Build the craploader"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make disk     - Create bootable disk image (requires kernel.bin)"
	@echo "  make test     - Build and test in QEMU"
	@echo "  make help     - Show this help message"
