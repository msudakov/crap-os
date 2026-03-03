#!/bin/bash
set -e

# This script runs the bootable image in QEMU for testing
# QEMU is easier for quick testing before moving to VMware

IMAGE_FILE="boot.img"

if [ ! -f "$IMAGE_FILE" ]; then
    echo "Error: $IMAGE_FILE not found"
    echo "Run 'make disk' first"
    exit 1
fi

echo "Starting QEMU with EFI firmware..."
echo "Controls:"
echo "  - Press Ctrl+Alt+G to release mouse"
echo "  - Press Ctrl+Alt+2 for QEMU monitor"
echo "  - Press Ctrl+Alt+1 to return to console"
echo "  - Press Ctrl+C in this terminal to quit"
echo ""

# Check if OVMF firmware is available
OVMF_CODE="/usr/share/ovmf/OVMF.fd"

if [ ! -f "$OVMF_CODE" ]; then
    # Try alternate locations
    if [ -f "/usr/share/edk2-ovmf/x64/OVMF_CODE.fd" ]; then
        OVMF_CODE="/usr/share/edk2-ovmf/x64/OVMF_CODE.fd"
    elif [ -f "/usr/share/qemu/ovmf-x86_64-code.bin" ]; then
        OVMF_CODE="/usr/share/qemu/ovmf-x86_64-code.bin"
    else
        echo "Error: OVMF firmware not found"
        echo ""
        echo "Install it with:"
        echo "  Ubuntu/Debian: sudo apt install ovmf"
        echo "  Fedora/RHEL:   sudo dnf install edk2-ovmf"
        echo "  Arch Linux:    sudo pacman -S edk2-ovmf"
        exit 1
    fi
fi

echo "Using OVMF firmware:"
echo "  Code: $OVMF_CODE"
echo ""

# Run QEMU with EFI
qemu-system-x86_64 \
    -drive if=pflash,format=raw,readonly=on,file="$OVMF_CODE" \
    -drive format=raw,file="$IMAGE_FILE" \
    -m 512M \
    -machine q35 \
    -cpu qemu64 \
    -serial stdio \
    -no-reboot \
    -no-shutdown \
    -net none #\        # Uncomment this and the line below to view interrupts
    #-d int,cpu_reset   # and CPU resets while debugging

# Cleanup
rm -f ./OVMF_VARS_temp.fd

echo ""
echo "QEMU session ended."
