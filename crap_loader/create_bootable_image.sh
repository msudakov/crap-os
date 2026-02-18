#!/bin/bash
set -e

# This script creates a bootable disk image for VMware with the EFI craploader and kernel

IMAGE_FILE="boot.img"
VMDK_FILE="boot.vmdk"
IMAGE_SIZE_MB=64
MOUNT_POINT="./mnt"

echo "Creating bootable disk image..."

# Check if craploader exists
if [ ! -f "craploader.efi" ]; then
    echo "ERROR: craploader.efi not found!"
    echo "Run 'make' first to build the craploader"
    exit 1
fi

# Check if kernel exists
if [ ! -f "kernel.bin" ]; then
    echo "ERROR: kernel.bin not found!"
    echo "Please place your kernel binary as kernel.bin in this directory"
    exit 1
fi

# Create a blank disk image
dd if=/dev/zero of="$IMAGE_FILE" bs=1M count=$IMAGE_SIZE_MB status=progress

# Create a GPT partition table
parted "$IMAGE_FILE" -s mklabel gpt
parted "$IMAGE_FILE" -s mkpart primary fat32 1MiB 100%
parted "$IMAGE_FILE" -s set 1 esp on

# Setup loop device
LOOP_DEVICE=$(sudo losetup --find --show --partscan "$IMAGE_FILE")
PARTITION="${LOOP_DEVICE}p1"

# Wait for partition to be ready
sleep 1

# Format the ESP partition as FAT32
sudo mkfs.vfat -F 32 "$PARTITION"

# Create mount point and mount the partition
mkdir -p "$MOUNT_POINT"
sudo mount "$PARTITION" "$MOUNT_POINT"

# Create EFI directory structure
sudo mkdir -p "$MOUNT_POINT/EFI/BOOT"

# Copy the craploader (renamed to the default EFI boot filename)
sudo cp craploader.efi "$MOUNT_POINT/EFI/BOOT/BOOTX64.EFI"

# Copy the kernel binary
sudo cp kernel.bin "$MOUNT_POINT/kernel.bin"

# List contents to verify
echo ""
echo "Disk contents:"
ls -lh "$MOUNT_POINT/"
ls -lh "$MOUNT_POINT/EFI/BOOT/"

# Unmount and cleanup
sudo umount "$MOUNT_POINT"
sudo losetup -d "$LOOP_DEVICE"
rmdir "$MOUNT_POINT"

qemu-img convert -f raw -O vmdk $IMAGE_FILE $VMDK_FILE
