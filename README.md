# CrapOS Project

It is an operating system that is really really bad, like total crap.

## Setting Up Environment

```
rustup target add x86_64-unknown-none
```

```
brew tap SergioBenitez/osxct
brew install x86_64-elf-gcc
```

`~/.cargo/config.toml`:
```
[target.x86_64-unknown-none]
linker = "x86_64-elf-gcc"
rustflags = ["-C", "link-arg=-nostartfiles"]
```

Creating project:
```
cargo new crap_os --bin --edition 2024
```

To build:
```
cd crap_os
cargo build --target x86_64-unknown-none
```