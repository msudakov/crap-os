TARGET          = kernel.bin
TARGET_DIR	    = target/release
LINKER_SCRIPT   = kernel.ld

.PHONY: all clean

all:
	rustc --edition 2021 --target x86_64-unknown-none --crate-type bin \
		-C link-arg=-T$(LINKER_SCRIPT) \
		-C link-arg=--oformat=binary \
		-C opt-level=2 \
		src/main.rs -o $(TARGET_DIR)/$(TARGET)

clean:
	rm -f $(TARGET_DIR)/$(TARGET)
