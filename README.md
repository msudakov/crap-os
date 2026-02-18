# CrapOS Project

Nine hells, why am I doing this..

This is an operating system, written in Rust. As the name suggests, it should
not be used by anyone. For any reason. It is really really bad, like total crap.

I wanted to learn more about and get some practice with Rust. As any sane person would do, I decided to write an OS with it. I also wanted to learn more about general OS internals. So as any sane person would do, I picked a language and paradigm I have very little experience with. If you think that a call to a mental institution is in order, you are absolutely correct.

## Demos

- [Demo 1](https://youtu.be/yjD1NYdGLt8)

## Setting Up Rust Environment

```
rustup toolchain install nightly
rustup component add rust-src
rustup component add rust-src --toolchain nightly-x86_64-apple-darwin
rustup component add llvm-tools-preview
rustup component add rust-std --target x86_64-crap_os
rustup target add x86_64-unknown-none
```

```
brew tap SergioBenitez/osxct
brew install x86_64-elf-gcc
```

`~/.cargo/config.toml`:
```
[unstable]
json-target-spec = true
build-std-features = ["compiler-builtins-mem"]
build-std = ["core", "compiler_builtins"]

[build]
target = "SHARE/crap-os/crap_os/x86_64-crap_os.json"
```

To build:
```
cd crap_os
rustc --edition 2021 --target x86_64-unknown-none --crate-type bin -C link-arg=-Tkernel.ld -C link-arg=--oformat=binary -C opt-level=2 src/main.rs -o target/release/kernel.bin
```

## Setting Up Bootloader Environment

Installing dependencies:

```
sudo apt-get install -y gnu-efi build-essential nasm qemu-system-x86 ovmf
```

```
cd crap_loader
make clean
cp ../crap_os/target/release/kernel.bin .
make disk
#make test (optional QEMU test)
#qemu-img convert -f raw -O vmdk boot.img boot.vmdk (make disk already does this)
```
