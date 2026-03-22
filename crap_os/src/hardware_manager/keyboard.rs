// =============================================================================
// PS/2 Keyboard Scancode Set 1 Driver
// =============================================================================
//
// Translates raw scancode set 1 bytes, delivered by IRQ 1 from the PS/2
// controller, into printable ASCII characters for a standard US QWERTY layout
// including the numpad.
//
// The PS/2 keyboard controller sits between the physical keyboard and the CPU.
// When a key is pressed or released, the keyboard sends a scancode byte to the
// controller, which buffers it and raises IRQ 1. The interrupt handler reads
// the byte from I/O port 0x60 (the 8042 data port) and passes it to this
// decoder.
//
// Scancode set 1 is the legacy set delivered by default by most PS/2 keyboards
// (and emulated by USB keyboards in legacy mode). Its encoding is as follows:
//   - Make code (key down): a single byte with bit 7 clear (0x01–0x58)
//   - Break code (key up): the make code with bit 7 set (0x81–0xD8)
//   - Extended prefix 0xE0: precedes a second byte for extended keys
//     (right Ctrl, right Alt, arrow keys, Insert, Delete, Home, End, etc.).
//     We do not handle extended keys yet, and both the 0xE0 prefix byte and the
//     following byte are silently consumed and discarded for now.
//
// We track three modifier states using atomic booleans:
//   - Shift (left or right): held while either shift key is physically down
//   - Caps Lock: toggled on the make code of the caps lock key
//   - Extended: set when 0xE0 is received, cleared after the next byte,
//     so the two-byte extended sequence is consumed atomically.
//
// Caps lock only affects alphabetic keys (a–z). Symbols on number/punctuation
// keys are unaffected by caps lock, which mirrors real keyboard behaviour.
// When both caps lock and shift are active on an alphabetic key, they cancel
// out (shift + caps = lowercase), which is also standard behaviour.
//
// Non-printable keys (F1–F12, Ctrl, Alt, arrow keys, etc.) are silently
// ignored.

use core::sync::atomic::{AtomicBool, Ordering};

// =============================================================================
// Modifier / sequence state
// =============================================================================
//
// These are `AtomicBool` rather than plain `bool` behind a mutex because they
// are written from an interrupt handler context where taking a lock could
// deadlock (if the lock is already held by the preempted thread). `Relaxed`
// ordering is sufficient, as the modifier flags are only read and written by
// the keyboard interrupt handler itself, so there is no cross-thread
// happens-before relationship to establish.

/// `true` while the left shift key is physically held down.
/// This is set on the shift make code and cleared on the shift break code.
static LSHIFT_HELD: AtomicBool = AtomicBool::new(false);

/// `true` while the right shift key is physically held down.
/// This is set on the shift make code and cleared on the shift break code.
static RSHIFT_HELD: AtomicBool = AtomicBool::new(false);

/// `true` when caps lock is active (toggled by each caps lock key press).
/// Unlike shift, caps lock is not cleared on key release; it latches.
static CAPS_ACTIVE: AtomicBool = AtomicBool::new(false);

/// `true` after receiving an 0xE0 extended-scancode prefix byte.
/// Cleared after the following (second) byte is consumed, discarding the
/// entire two-byte extended sequence.
static EXTENDED: AtomicBool = AtomicBool::new(false);

/// `true` while the left Ctrl key is physically held down.
/// This is set on the left Ctrl make code and cleared on the left Ctrl break
/// code.
static CTRL_HELD:  AtomicBool = AtomicBool::new(false);

/// `true` while the left Alt key (the Option key on MacOS) is physically held
/// down. This is set on the Alt make code and cleared on the Alt break code.
static ALT_HELD:   AtomicBool = AtomicBool::new(false);

// =============================================================================
// Scancode constants
// =============================================================================
//
// These are the specific make/break scancode bytes we need to handle as
// modifiers. All other make codes are handled by direct table lookup.

/// Make code for the left Shift key.
const SC_LSHIFT_MAKE: u8 = 0x2A;

/// Make code for the right Shift key.
const SC_RSHIFT_MAKE: u8 = 0x36;

/// Break code for the left Shift key (= SC_LSHIFT_MAKE | 0x80).
const SC_LSHIFT_BREAK: u8 = 0xAA;

/// Break code for the right Shift key (= SC_RSHIFT_MAKE | 0x80).
const SC_RSHIFT_BREAK: u8 = 0xB6;

/// Make code for the Caps Lock key.
/// We only act on the make code; the break code (0xBA) is ignored because
/// caps lock toggles on press, not on release.
const SC_CAPS_MAKE: u8 = 0x3A;

/// The extended scancode prefix byte. When this byte is received, the next
/// byte is the second half of a two-byte extended scancode.
const SC_EXTENDED: u8 = 0xE0;

/// Make code for the left Ctrl key.
const SC_CTRL_MAKE:   u8 = 0x1D;

/// Break code for the left Ctrl key (= SC_CTRL_MAKE | 0x80).
const SC_CTRL_BREAK:  u8 = 0x9D;

/// Make code for the left Alt (MacOS Option) key.
const SC_ALT_MAKE:    u8 = 0x38;

/// Break code for the left Alt (MacOS Option) key (= SC_ALT_MAKE | 0x80).
const SC_ALT_BREAK:   u8 = 0xB8;

/// Make code for the Escape key.
const SC_ESC_MAKE: u8 = 0x01;

/// Break code for the Escape key (= SC_ESC_MAKE | 0x80).
const _SC_ESC_BREAK: u8 = 0x81;

// =============================================================================
// Translation tables
// =============================================================================
//
// These two parallel lookup tables are indexed by the make scancode byte
// (0x00–0x53). A table entry of 0x00 means "no printable character"
// (the key is a modifier, control key, function key, etc.).
//
// The tables cover make codes 0x00–0x53 (84 entries = 0x54). Scancodes above
// 0x53 are either break codes (>= 0x80) or extended (prefixed by 0xE0) and are
// not present in these tables.
//
// `#[rustfmt::skip]` preserves the hand-aligned tabular layout, which makes
// the correspondence between scancode values and characters much easier to
// verify visually against a scancode reference sheet.

/// Unshifted (normal) characters for scancode set 1 make codes 0x00–0x53.
///
/// We index into this table with the raw make scancode byte. A value of 0
/// indicates no printable character for that scancode.
///
/// Notable entries:
///   - 0x00 = Unused
///   - 0x01 = Escape
///   - 0x0E (index 14) = 0x08 = ASCII backspace
///   - 0x0F (index 15) = '\t' = ASCII horizontal tab
///   - 0x1C (index 28) = '\n' = ASCII newline (Enter key)
///   - 0x1D = LCtrl
///   - 0x2A = LShift
///   - 0x36 = RShift
///   - 0x37 = Numpad*
///   - 0x38 = LAlt
///   - 0x39 (index 57) = ' '  = ASCII space
///   - 0x3A = CapsLock
///   - 0x3B-0x3F = F1–F5
///   - 0x40–0x46 = F6–F12/misc
///   - 0x47 = Numpad7
///   - Numpad keys at 0x47–0x53 produce digit/symbol characters regardless of
///       Num Lock state (we do not track Num Lock for now).
#[rustfmt::skip]
const UNSHIFTED: [u8; 0x54] = [
//  +0     +1     +2     +3     +4     +5     +6     +7
    0,     0,     b'1',  b'2',  b'3',  b'4',  b'5',  b'6',   // 0x00
    b'7',  b'8',  b'9',  b'0',  b'-',  b'=',  0x08,  b'\t',  // 0x08
    b'q',  b'w',  b'e',  b'r',  b't',  b'y',  b'u',  b'i',   // 0x10
    b'o',  b'p',  b'[',  b']',  b'\n', 0,     b'a',  b's',   // 0x18
    b'd',  b'f',  b'g',  b'h',  b'j',  b'k',  b'l',  b';',   // 0x20
    b'\'', b'`',  0,     b'\\', b'z',  b'x',  b'c',  b'v',   // 0x28
    b'b',  b'n',  b'm',  b',',  b'.',  b'/',  0,     b'*',   // 0x30
    0,     b' ',  0,     0,     0,     0,     0,     0,      // 0x38
    0,     0,     0,     0,     0,     0,     0,     b'7',   // 0x40
    b'8',  b'9',  b'-',  b'4',  b'5',  b'6',  b'+',  b'1',   // 0x48
    b'2',  b'3',  b'0',  b'.',                               // 0x50
];

/// Shifted characters for the same scancode range (0x00–0x53).
///
/// Consulted when `SHIFT_HELD` is true and the key is not an alphabetic
/// affected by caps lock. Digit keys produce symbols, and alphabetic keys
/// produce uppercase letters.
///
/// Numpad keys are unchanged between `UNSHIFTED` and `SHIFTED` because shift
/// would normally activate the numpad's alternate functions (arrows, Insert,
/// etc.), but we ignore Num Lock and do not handle those extended functions
/// for now.
#[rustfmt::skip]
const SHIFTED: [u8; 0x54] = [
//  +0     +1     +2     +3     +4     +5     +6     +7
    0,     0,     b'!',  b'@',  b'#',  b'$',  b'%',  b'^',   // 0x00
    b'&',  b'*',  b'(',  b')',  b'_',  b'+',  0x08,  b'\t',  // 0x08
    b'Q',  b'W',  b'E',  b'R',  b'T',  b'Y',  b'U',  b'I',   // 0x10
    b'O',  b'P',  b'{',  b'}',  b'\n', 0,     b'A',  b'S',   // 0x18
    b'D',  b'F',  b'G',  b'H',  b'J',  b'K',  b'L',  b':',   // 0x20
    b'"',  b'~',  0,     b'|',  b'Z',  b'X',  b'C',  b'V',   // 0x28
    b'B',  b'N',  b'M',  b'<',  b'>',  b'?',  0,     b'*',   // 0x30
    0,     b' ',  0,     0,     0,     0,     0,     0,      // 0x38
    0,     0,     0,     0,     0,     0,     0,     b'7',   // 0x40
    b'8',  b'9',  b'-',  b'4',  b'5',  b'6',  b'+',  b'1',   // 0x48
    b'2',  b'3',  b'0',  b'.',                               // 0x50
];

/// Processes one raw scancode byte from the PS/2 controller, and is called
/// directly from the keyboard IRQ handler with the byte read from I/O port
/// 0x60 immediately after the interrupt fires.
///
/// # Arguments
/// 
/// * `scancode` - Raw scancode byte to process.
/// 
/// # Returns
/// 
/// Returns the corresponding ASCII character, or `None` if the byte should be
/// silently consumed.
/// 
/// `None` is returned when:
///   - The byte is the 0xE0 extended-scancode prefix (sets `EXTENDED` flag)
///   - The byte is the second byte of a two-byte extended sequence (clears
///     `EXTENDED` flag)
///   - The byte is a shift make/break code (updates `SHIFT_HELD`)
///   - The byte is a caps lock make code (toggles `CAPS_ACTIVE`)
///   - The byte has bit 7 set (break code / key release)
///   - The scancode is outside the table range (>= 0x54)
///   - The table entry for this scancode is 0 (modifier, function key, etc.)
///
/// # Thread Safety
/// 
/// Safe to call from an interrupt context. All shared state (`SHIFT_HELD`,
/// `CAPS_ACTIVE`, `EXTENDED`) is accessed via `AtomicBool` with `Relaxed`
/// ordering, which is sufficient because this function is the sole writer and
/// reader of those flags, and no cross-thread synchronisation is required.
pub fn process_scancode(scancode: u8) -> Option<u8> {
    // Extended scancode prefix (0xE0)
    if scancode == SC_EXTENDED {
        EXTENDED.store(true, Ordering::Relaxed);
        return None;
    }

    // Second byte of an extended (0xE0 xx) sequence. The previous interrupt set
    // EXTENDED, and this byte is the second half of that sequence. We clear the
    // flag and discard the byte.
    if EXTENDED.load(Ordering::Relaxed) {
        EXTENDED.store(false, Ordering::Relaxed);
        return None;
    }

    // Shift and caps lock must be processed before the break-code check below,
    // because shift break codes (0xAA, 0xB6) have bit 7 set and would otherwise
    // be incorrectly discarded as generic break codes.
    match scancode {
        // Left shift pressed: activate shifted table
        SC_LSHIFT_MAKE => {
            LSHIFT_HELD.store(true, Ordering::Relaxed);
            return None;
        }

        // Right shift pressed: activate shifted table
        SC_RSHIFT_MAKE => {
            RSHIFT_HELD.store(true, Ordering::Relaxed);
            return None;
        }

        // Left shift released: deactivate shifted table
        SC_LSHIFT_BREAK => {
            LSHIFT_HELD.store(false, Ordering::Relaxed);
            return None;
        }

        // Right shift released: deactivate shifted table.
        SC_RSHIFT_BREAK => {
            RSHIFT_HELD.store(false, Ordering::Relaxed);
            return None;
        }

        // Left Ctrl pressed
        SC_CTRL_MAKE => {
            CTRL_HELD.store(true, Ordering::Relaxed);
            return None;
        }

        // Left Ctrl released
        SC_CTRL_BREAK => {
            CTRL_HELD.store(false, Ordering::Relaxed);
            return None;
        }

        // Left Alt (MacOS Option) key pressed
        SC_ALT_MAKE => {
            ALT_HELD.store(true, Ordering::Relaxed);
            return None;
        }

        // Left Alt (MacOS Option) key released
        SC_ALT_BREAK => {
            ALT_HELD.store(false, Ordering::Relaxed);
            return None;
        }

        // Caps lock key pressed: toggle the caps lock state.
        // We act only on the make code; the break code (0xBA) is a regular
        // break that will be discarded by the bit-7 check below.
        SC_CAPS_MAKE => {
            let current = CAPS_ACTIVE.load(Ordering::Relaxed);
            CAPS_ACTIVE.store(!current, Ordering::Relaxed);
            return None;
        }
        _ => {}
    }

    // For all non-modifier keys, a break code is the make code with bit 7 set.
    // We have already handled the only modifier break codes we care about
    // (shift), so any remaining byte with bit 7 set is a key-release event
    // for a key we track only on press, so we just discard it.
    if scancode & 0x80 != 0 {
        return None;
    }

    // Handle the ESC key (scancode 0x01)
    if scancode == SC_ESC_MAKE {
        if CTRL_HELD.load(Ordering::Relaxed) && ALT_HELD.load(Ordering::Relaxed
        ) {
            // TODO:
            // Placeholder system shutdown control sequence (CTRL+ALT+ESC)
            return Some(0xFF);
        }

        return Some(0x1B);  // ASCII ESC character
    }

    // Our tables only cover make codes 0x00–0x53 (84 entries). Scancodes at
    // or beyond index 0x54 have no entry, so we can silently ignore them.
    if scancode as usize >= UNSHIFTED.len() {
        return None;
    }

    // Here, we determine which table to use based solely on the shift state.
    // Caps lock does not influence the table choice here, as it is applied as a
    // post-processing step below, only for alphabetic characters.
    let caps_lock  = CAPS_ACTIVE.load(Ordering::Relaxed);
    let shift = LSHIFT_HELD.load(Ordering::Relaxed) || RSHIFT_HELD.load(
        Ordering::Relaxed);

    let base_char = if shift {
        SHIFTED[scancode as usize]
    } else {
        UNSHIFTED[scancode as usize]
    };

    // A table entry of 0x00 means this scancode has no printable mapping, so
    // we discard it (e.g., Ctrl, Alt, F-keys, the unused 0x00 and 0x01 slots,
    // etc.).
    if base_char == 0 {
        return None;
    }

    // Caps lock only affects alphabetic characters (a–z / A–Z). Symbols that
    // share a key with a letter (there are none in standard QWERTY, but the
    // rule applies regardless) are not affected.
    //
    // This is the interaction between shift and caps lock on alphabetic keys:
    //   shift=false, caps=false  ->  lowercase  (UNSHIFTED table: 'a')
    //   shift=false, caps=true   ->  uppercase  (SHIFTED   table: 'A')
    //   shift=true,  caps=false  ->  uppercase  (SHIFTED   table: 'A')
    //   shift=true,  caps=true   ->  lowercase  (UNSHIFTED table: 'a')
    //
    // This is the "XOR" rule, and caps lock simply inverts the effect of shift
    // for letters. We implemented this by swapping which table we use whenever
    // caps is active and the character is alphabetic. For non-alphabetic
    // characters (digits, punctuation, space, etc.), we always use the result
    // from the shift-based lookup above, as caps lock has no effect on them.
    let final_char = if base_char.is_ascii_alphabetic() && caps_lock {
        // Caps lock is active and this is a letter, so we swap the table.
        if shift {
            UNSHIFTED[scancode as usize]  // shift + caps lock -> lowercase
        } else {
            SHIFTED[scancode as usize]    // no shift + caps lock -> uppercase
        }
    } else {
        // Either this is not an alphabetic character, or caps lock is inactive.
        base_char
    };

    Some(final_char)
}
