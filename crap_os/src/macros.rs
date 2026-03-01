// This file contains global macros used throughout the system.

#[macro_export]
macro_rules! fbprint {
    ($($arg:tt)*) => {{
        use core::fmt::Write;
        $crate::globals::FRAMEBUFFER.lock()
            .as_mut()
            .unwrap()
            .write_fmt(format_args!($($arg)*))
            .unwrap();
    }};
}

/// Prints a formatted string in the framebuffer.
#[macro_export]
macro_rules! fbprintln {
    () => ($crate::print!("\n"));
    ($($arg:tt)*) => ($crate::fbprint!("{}\n", format_args!($($arg)*)));
}
