//! Global Macros

/// Prints a formatted string to serial port.
#[macro_export]
macro_rules! sprint {
    ($($arg:tt)*) => {{
        use core::fmt::Write;
        $crate::globals::SERIAL.lock()
            .as_mut()
            .unwrap()
            .write_fmt(format_args!($($arg)*))
            .unwrap();
    }};
}

/// Prints a formatted string to serial port, appending newline at the end.
#[macro_export]
macro_rules! sprintln {
    () => ($crate::print!("\n"));
    ($($arg:tt)*) => ($crate::sprint!("{}\n", format_args!($($arg)*)));
}

/// Prints a formatted string to the framebuffer.
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

/// Prints a formatted string to the framebuffer, appending newline at the end.
#[macro_export]
macro_rules! fbprintln {
    () => ($crate::print!("\n"));
    ($($arg:tt)*) => ($crate::fbprint!("{}\n", format_args!($($arg)*)));
}

/// Macro for the serial port debugging messages based on debug level.
#[macro_export]
macro_rules! sprint_debug {
    ($dbg_lvl:expr, $msg:expr) => {
        globals::SERIAL.lock().as_mut().unwrap().print_debug($dbg_lvl, $msg);
    };
}
