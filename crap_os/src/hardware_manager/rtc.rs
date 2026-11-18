//! # RTC - Real-Time Clock (CMOS)
//!
//! This module reads the legacy CMOS (Complementary Metal-Oxide-Semiconductor)
//! RTC at boot to establish a Unix epoch anchor for the kernel's wall clock.
//! The CMOS RTC only has whole-second resolution, and is far too slow to poll
//! on every timestamp request, so we read it exactly once during boot and pair
//! that reading with a snapshot of the HPET main counter taken at the same
//! instant. All later wall-clock queries derive elapsed time since that anchor
//! from the HPET counter and report it in milliseconds:
//!
//! elapsed_ms = (hpet_now - hpet_ticks_at_anchor) *hpet_period_fs/1000000000000
//! now()      = epoch_seconds_at_anchor * 1000 + elapsed_ms
//!
//! Milliseconds is the only granularity exposed. The RTC-derived
//! `epoch_seconds_at_anchor` carries whole-second-only precision (a CMOS RTC
//! limitation, so the anchor's absolute value may be offset from true UTC by
//! up to ~999ms), but elapsed time past that point is limited only by the
//! HPET's own resolution, so `now()` still produces genuine millisecond-
//! granularity timestamps relative to the anchor.
//! 
//! The HPET is used as the fixed point instead of `TIMER_TICKS` because the
//! HPET is already free-running at the point we call this (no APIC timer
//! configuration or `sti` required), whereas `TIMER_TICKS` does not start
//! advancing until interrupts are enabled much later in `_start`.
//!
//! We assume the RTC is set to UTC, and no timezone offset is applied.
//!
//! The classic CMOS register set (0x00-0x09) only stores a 2-digit year.
//! Whether a separate CMOS "century" register exists, and at what index, is
//! reported by the FADT's `century` field (byte offset 108, ACPI 2.0+ only).
//! If the FADT is too short to contain this field (ACPI 1.0) or reports 0
//! (not supported), we fall back to assuming the 21st century, which is a
//! reasonable assumption for any system this kernel will realistically boot
//! on.
//!
//! CMOS registers are not read atomically as a group; the RTC can update
//! mid-read. We guard against this two ways, per the standard OSDev
//! algorithm:
//!   1. Before each read attempt, we poll Status Register A's UIP
//!      (Update-In-Progress) bit until it clears.
//!   2. We read all fields twice and compare; if the two readings differ, an
//!      update happened between reads and we retry.

use core::ptr;
use super::acpi::find_acpi_table;
use super::hpet::HpetInfo;
use super::serial::{inb, outb};

/// CMOS index/address port. Writing a register number here selects it for the
/// next read/write on [`CMOS_DATA`]. Bit 7 additionally controls NMI masking;
/// we always leave it clear (NMI enabled).
const CMOS_ADDRESS: u16 = 0x70;

/// CMOS data port. Reflects the register most recently selected via
/// [`CMOS_ADDRESS`].
const CMOS_DATA: u16 = 0x71;

// CMOS RTC register indices (standard MC146818-compatible layout).
const REG_SECONDS: u8 = 0x00;
const REG_MINUTES: u8 = 0x02;
const REG_HOURS: u8 = 0x04;
const REG_DAY: u8 = 0x07;
const REG_MONTH: u8 = 0x08;
const REG_YEAR: u8 = 0x09;
const REG_STATUS_A: u8 = 0x0A;
const REG_STATUS_B: u8 = 0x0B;

/// Status Register A, bit 7: Update-In-Progress. Set for roughly the last
/// 244us before, and during, the RTC's once-per-second update of the time
/// registers. Registers must not be read while this bit is set.
const STATUS_A_UIP: u8 = 1 << 7;

/// Status Register B, bit 2: 1 = registers are binary, 0 = registers are BCD.
const STATUS_B_BINARY_MODE: u8 = 1 << 2;

/// Status Register B, bit 1: 1 = 24-hour mode, 0 = 12-hour mode.
const STATUS_B_24HOUR_MODE: u8 = 1 << 1;

/// Reads a single CMOS register by index.
///
/// # Arguments
///
/// * `reg` - CMOS register index to select via [`CMOS_ADDRESS`] before reading.
///
/// # Returns
///
/// Returns the byte value currently held in the selected register.
///
/// # Safety
///
/// Executes raw port I/O against the CMOS controller.
#[inline(always)]
unsafe fn cmos_read(reg: u8) -> u8 {
    outb(CMOS_ADDRESS, reg);
    inb(CMOS_DATA)
}

/// Busy-waits until the RTC's Update-In-Progress bit is clear, meaning it is
/// currently safe to read the time registers without racing an update.
///
/// # Safety
///
/// Executes raw port I/O against the CMOS controller.
unsafe fn wait_update_not_in_progress() {
    while unsafe { cmos_read(REG_STATUS_A) } & STATUS_A_UIP != 0 {
        core::hint::spin_loop();
    }
}

/// Byte offset of the `century` field within the FADT, per the ACPI
/// specification. Only present when the table is long enough (ACPI 2.0+).
const FADT_CENTURY_OFFSET: usize = 108;

/// Locates the FADT via ACPI and reads its `century` field, which gives the
/// CMOS register index holding the current century (e.g. `0x32`), if the
/// platform provides one.
///
/// # Arguments
///
/// * `rsdp_virt` - RSDP virtual address (already translated through the
///   kernel's direct physical map), same requirement as [`find_acpi_table`].
///
/// # Returns
///
/// Returns `Some(register_index)` if the FADT is present, long enough to
/// contain the `century` field, and that field is non-zero. Returns `None`
/// otherwise, in which case the caller should assume the 21st century.
///
/// # Safety
///
/// `rsdp_virt` must be a valid, mapped virtual address pointing to a genuine
/// RSDP, same requirement as `find_acpi_table`.
unsafe fn find_century_register(rsdp_virt: u64) -> Option<u8> {
    let fadt_sdt = unsafe { find_acpi_table(rsdp_virt, b"FACP")? };

    let table_len = unsafe {
        ptr::read_unaligned(core::ptr::addr_of!((*fadt_sdt).length)) as usize
    };

    // ACPI 1.0 FADTs are shorter than this offset and simply don't have the
    // field; reading past `table_len` would read garbage beyond the table.
    if table_len < FADT_CENTURY_OFFSET + 1 {
        return None;
    }

    let century_ptr = (fadt_sdt as usize + FADT_CENTURY_OFFSET) as *const u8;
    let century_reg = unsafe { ptr::read_unaligned(century_ptr) };

    if century_reg == 0 {
        None
    } else {
        Some(century_reg)
    }
}

/// One complete, untranslated snapshot of the CMOS time/date registers.
/// Values are still in whatever format the hardware reports (BCD or binary,
/// 12-hour or 24-hour) at the point of capture; `normalize()` converts them.
#[derive(Copy, Clone, PartialEq, Eq)]
struct RtcReading {
    second: u8,
    minute: u8,
    hour: u8,
    day: u8,
    month: u8,
    year: u8,
    /// Raw century register value, if a century register is present.
    century: Option<u8>,
}

/// Reads all RTC time/date registers once, without any tearing protection.
/// Callers must pair this with [`wait_update_not_in_progress()`] beforehand and
/// a second read to check for consistency; see [`read_stable_rtc()`].
///
/// # Arguments
///
/// * `century_reg` - CMOS register index holding the century, if the
///   platform provides one (from [`find_century_register`]), or `None` to skip
///   reading a century register.
///
/// # Returns
///
/// Returns a raw, untranslated `RtcReading` snapshot of the current register
/// values.
///
/// # Safety
///
/// Executes raw port I/O against the CMOS controller.
unsafe fn read_rtc_once(century_reg: Option<u8>) -> RtcReading {
    unsafe {
        RtcReading {
            second: cmos_read(REG_SECONDS),
            minute: cmos_read(REG_MINUTES),
            hour: cmos_read(REG_HOURS),
            day: cmos_read(REG_DAY),
            month: cmos_read(REG_MONTH),
            year: cmos_read(REG_YEAR),
            century: century_reg.map(|reg| cmos_read(reg)),
        }
    }
}

/// Reads the RTC time/date registers, retrying until two consecutive reads
/// (each preceded by a UIP wait) agree, guaranteeing no update happened
/// mid-read.
///
/// # Arguments
///
/// * `century_reg` - CMOS register index holding the century, if the
///   platform provides one (from [`find_century_register`]), or `None` to skip
///   reading a century register.
///
/// # Returns
///
/// Returns a raw, untranslated `RtcReading` snapshot confirmed stable across
/// two consecutive reads.
///
/// # Safety
///
/// Executes raw port I/O against the CMOS controller.
unsafe fn read_stable_rtc(century_reg: Option<u8>) -> RtcReading {
    loop {
        unsafe { wait_update_not_in_progress() };
        let first = unsafe { read_rtc_once(century_reg) };

        unsafe { wait_update_not_in_progress() };
        let second = unsafe { read_rtc_once(century_reg) };

        if first == second {
            return second;
        }
        // Registers changed between reads (an update landed in the gap);
        // loop and try again.
    }
}

/// Converts a single BCD byte to binary. BCD packs two decimal digits per
/// byte: high nibble = tens digit, low nibble = ones digit.
///
/// # Arguments
///
/// * `value` - A byte in BCD format (each nibble a decimal digit 0-9).
///
/// # Returns
///
/// Returns the equivalent value in plain binary.
#[inline(always)]
fn bcd_to_binary(value: u8) -> u8 {
    (value & 0x0F) + ((value >> 4) * 10)
}

/// A fully normalized, decoded RTC reading: binary-encoded, 24-hour, with a
/// resolved 4-digit year.
struct NormalizedRtc {
    second: u32,
    minute: u32,
    hour: u32,
    day: u32,
    month: u32,
    year: u32,
}

impl RtcReading {
    /// Converts this raw reading into normalized, decoded fields, using the
    /// RTC's mode bits (Status Register B) to determine whether BCD and/or
    /// 12-hour decoding is needed.
    ///
    /// # Returns
    ///
    /// Returns a `NormalizedRtc` with all fields in plain binary, 24-hour
    /// time, and a resolved 4-digit year.
    ///
    /// # Safety
    ///
    /// Executes raw port I/O against the CMOS controller to read Status
    /// Register B.
    unsafe fn normalize(&self) -> NormalizedRtc {
        let status_b = unsafe { cmos_read(REG_STATUS_B) };
        let is_binary = status_b & STATUS_B_BINARY_MODE != 0;
        let is_24hour = status_b & STATUS_B_24HOUR_MODE != 0;

        // In 12-hour mode, bit 7 of the hour register is the PM flag and the
        // remaining bits hold the 1-12 hour value (still possibly in BCD).
        let hour_pm = !is_24hour && (self.hour & 0x80) != 0;
        let hour_raw = self.hour & 0x7F;

        let (second, minute, mut hour, day, month, year) = if is_binary {
            (
                self.second as u32,
                self.minute as u32,
                hour_raw as u32,
                self.day as u32,
                self.month as u32,
                self.year as u32,
            )
        } else {
            (
                bcd_to_binary(self.second) as u32,
                bcd_to_binary(self.minute) as u32,
                bcd_to_binary(hour_raw) as u32,
                bcd_to_binary(self.day) as u32,
                bcd_to_binary(self.month) as u32,
                bcd_to_binary(self.year) as u32,
            )
        };

        // Convert 12-hour to 24-hour. Midnight (12 AM) reads as 12 and maps
        // to 0; noon (12 PM) reads as 12 and stays 12.
        if !is_24hour {
            if hour_pm && hour != 12 {
                hour += 12;
            }
            else if !hour_pm && hour == 12 {
                hour = 0;
            }
        }

        // Resolve the 4-digit year. If a century register was present, its
        // value may itself be BCD-encoded under the same mode bit.
        let full_year = match self.century {
            Some(raw_century) => {
                let century = if is_binary {
                    raw_century as u32
                } else {
                    bcd_to_binary(raw_century) as u32
                };
                century * 100 + year
            }
            // No century register: assume the 21st century. Reasonable for
            // any real hardware this kernel boots on.
            None => 2000 + year,
        };

        NormalizedRtc {
            second,
            minute,
            hour,
            day,
            month,
            year: full_year,
        }
    }
}

/// Converts a Gregorian calendar date to the number of days since the Unix
/// epoch (1970-01-01), using Howard Hinnant's well-known constant-time
/// civil-to-days algorithm. Valid for the entire proleptic Gregorian
/// calendar; correctness for our purposes only matters post-1970.
///
/// # Arguments
///
/// * `year` - Full 4-digit Gregorian year (e.g. `2026`).
/// * `month` - Month of the year, `1`-`12`.
/// * `day` - Day of the month, `1`-`31`.
///
/// # Returns
///
/// Returns the signed number of days since 1970-01-01 (negative for dates
/// before the epoch, unused here since the RTC always reads a post-1970
/// date).
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;  // [0, 399]
    let doy = (153 * (if month > 2 { month - 3 } else { month + 9 }) + 2) / 5
        + day - 1;  // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;  // [0, 146096]
    era * 146097 + doe - 719468
}

impl NormalizedRtc {
    /// Converts this normalized reading to a Unix epoch timestamp, in whole
    /// seconds, assuming the RTC is set to UTC.
    ///
    /// # Returns
    ///
    /// Returns the Unix epoch time in whole seconds.
    fn to_epoch_seconds(&self) -> u64 {
        let days = days_from_civil(
            self.year as i64,
            self.month as i64,
            self.day as i64,
        );
        let seconds_in_day = self.hour as i64 * 3600
            + self.minute as i64 * 60
            + self.second as i64;
        (days * 86400 + seconds_in_day) as u64
    }
}

// =============================================================================
// Public Interface
// =============================================================================

/// Wall-clock anchor: a Unix epoch timestamp paired with the HPET main
/// counter value at the instant that timestamp was captured. Later queries
/// derive the current wall-clock time from elapsed HPET ticks rather than
/// re-reading the (slow, whole-second-resolution) RTC.
#[allow(dead_code)]
pub struct WallClock {
    /// Unix epoch seconds (UTC) at the moment `hpet_ticks_at_anchor` was
    /// sampled.
    epoch_seconds_at_anchor: u64,

    /// HPET main counter value sampled immediately after the RTC reading was
    /// confirmed stable.
    hpet_ticks_at_anchor: u64,
}

#[allow(dead_code)]
impl WallClock {
    /// Computes the current Unix epoch time in milliseconds.
    ///
    /// The RTC anchor itself only has whole-second resolution (the CMOS RTC
    /// cannot report sub-second time), so the absolute value returned here
    /// may be offset from true UTC by up to ~999ms, fixed at boot. However,
    /// elapsed time between calls to this function is derived entirely from
    /// the HPET counter and is accurate to the HPET's actual resolution
    /// (nanoseconds), so this is safe to use for timestamps, ordering events,
    /// and measuring durations at millisecond granularity. Callers needing
    /// coarser granularity (e.g., whole seconds) can divide/round the result
    /// themselves. Uses `wrapping_sub` for the elapsed-ticks calculation, so
    /// this remains correct even if the HPET main counter has wrapped.
    ///
    /// # Arguments
    ///
    /// * `hpet` - Reference to the same `HpetInfo` this `WallClock` was
    ///   anchored against (from `read_rtc_epoch_anchor`).
    ///
    /// # Returns
    ///
    /// Returns the current Unix epoch time in milliseconds.
    pub unsafe fn now(&self, hpet: &HpetInfo) -> u64 {
        let ticks_now = unsafe { hpet.read_counter() };
        let elapsed_ticks = ticks_now.wrapping_sub(self.hpet_ticks_at_anchor);
        let elapsed_ms = hpet.ticks_to_ns(elapsed_ticks) / 1_000_000;
        self.epoch_seconds_at_anchor * 1000 + elapsed_ms
    }
}

/// Reads the CMOS RTC once, resolves the century via the FADT if available,
/// and returns a [`WallClock`] anchored to the HPET counter at the instant the
/// reading was confirmed stable.
///
/// This should be called once during boot, after `parse_hpet()` has
/// succeeded (the HPET main counter must already be running) and while the
/// FADT/RSDP are still reachable.
///
/// # Arguments
///
/// * `rsdp_virt` - RSDP virtual address (already translated through the
///   kernel's direct physical map), same as passed to `parse_acpi`/
///   `parse_hpet`.
/// * `hpet` - Reference to the already-initialized [`HpetInfo`].
///
/// # Returns
///
/// Returns a [`WallClock`] anchored to the current Unix epoch time (UTC) and
/// the HPET main counter value at the instant that time was captured.
///
/// # Safety
///
/// `rsdp_virt` must be valid per `find_acpi_table`'s requirements, and `hpet`
/// must refer to an HPET whose MMIO page is mapped and whose main counter is
/// running (guaranteed by a successful `parse_hpet()`).
pub unsafe fn read_rtc_epoch_anchor(rsdp_virt: u64, hpet: &HpetInfo) -> WallClock {
    let century_reg = unsafe { find_century_register(rsdp_virt) };
    let raw = unsafe { read_stable_rtc(century_reg) };
    let normalized = unsafe { raw.normalize() };
    let epoch_seconds_at_anchor = normalized.to_epoch_seconds();

    // Snapshot the HPET counter immediately after the RTC value is confirmed
    // stable, so the two are as close to simultaneous as possible. The
    // dominant source of imprecision remains the RTC's own 1-second
    // resolution, not the small gap between this read and the snapshot.
    let hpet_ticks_at_anchor = unsafe { hpet.read_counter() };

    WallClock {
        epoch_seconds_at_anchor,
        hpet_ticks_at_anchor,
    }
}
