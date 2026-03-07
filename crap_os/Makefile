TARGET          = kernel.bin
TARGET_ELF		= kernel.elf
TARGET_DIR	    = target/release
LINKER_SCRIPT   = kernel.ld

.PHONY: all clean

all:
	rustc --edition 2021 --target x86_64-unknown-none --crate-type bin \
		-C link-arg=-T$(LINKER_SCRIPT) \
		-C link-arg=--oformat=binary \
		-C opt-level=2 \
		-C relocation-model=static \
		src/main.rs -o $(TARGET_DIR)/$(TARGET)

debug:
	rustc --edition 2021 --target x86_64-unknown-none --crate-type bin \
		-C link-arg=-T$(LINKER_SCRIPT) \
		-C link-arg=--oformat=binary \
		-C opt-level=1 \
		-C debuginfo=2 \
		-C relocation-model=static \
		src/main.rs -o $(TARGET_DIR)/$(TARGET)

elf:
	rustc --edition 2021 --target x86_64-unknown-none --crate-type bin \
        -C link-arg=-Tkernel.ld \
        -C opt-level=2 \
        -C relocation-model=static \
        src/main.rs -o $(TARGET_DIR)/$(TARGET_ELF)

clean:
	rm -f $(TARGET_DIR)/$(TARGET)
	rm -f $(TARGET_DIR)/$(TARGET_ELF)
