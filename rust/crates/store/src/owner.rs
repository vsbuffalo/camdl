//! Owners, their beat, and liveness.
//!
//! A process that holds claims is an owner, named by host, process id and the
//! process's start time. It keeps one beat file under `.owners/` and raises a
//! counter in it every [`BEAT_INTERVAL`]. Every claim names its owner.
//!
//! Liveness is read two ways. On the owner's host, an owner is alive when a
//! process with its id exists and has its start time; the start time is what
//! keeps a reused process id from reading as the owner. From another host, an
//! owner is alive while its beat counter changes, observed on the reader's own
//! clock. No timestamp is compared across hosts.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{json_err, At, Result};
use crate::fsutil::write_file_atomic;

/// How often an owner raises its beat counter.
pub const BEAT_INTERVAL: Duration = Duration::from_secs(2);

/// How long a reader on another host watches a beat counter before calling an
/// owner whose counter has not changed stale: three beat intervals.
pub const STALE_AFTER: Duration = Duration::from_secs(6);

/// The process holding claims.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Owner {
    pub host: String,
    pub pid: u32,
    /// The process start time, as this host's OS reports it.
    pub started: Timestamp,
}

impl std::fmt::Display for Owner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "process {} on {} (started {})", self.pid, self.host, self.started.0)
    }
}

impl Owner {
    /// The name of this owner's beat file and index log, without extension.
    pub(crate) fn file_stem(&self) -> String {
        crate::layout::path_label(&format!("{}-{}-{}", self.host, self.pid, self.started.0))
    }

    pub(crate) fn is_this_host(&self) -> bool {
        self.host == this_host()
    }
}

/// A time for people to read, in UTC to the millisecond, such as
/// `2026-11-20T09:14:03.120Z`. Never compared across hosts and never used to
/// order events.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Timestamp(pub String);

impl Timestamp {
    pub fn now() -> Self {
        Self::from_system(SystemTime::now())
    }

    pub fn from_system(t: SystemTime) -> Self {
        let d = t.duration_since(UNIX_EPOCH).unwrap_or_default();
        Self::from_unix_millis(d.as_millis() as i64)
    }

    fn from_unix_millis(ms: i64) -> Self {
        let secs = ms.div_euclid(1000);
        let millis = ms.rem_euclid(1000);
        let days = secs.div_euclid(86_400);
        let sod = secs.rem_euclid(86_400);
        let (y, m, d) = civil_from_days(days);
        Timestamp(format!(
            "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}.{millis:03}Z",
            sod / 3600,
            (sod % 3600) / 60,
            sod % 60
        ))
    }
}

/// The proleptic Gregorian date of a count of days since 1970-01-01, by
/// Howard Hinnant's `civil_from_days` (<https://howardhinnant.github.io/date_algorithms.html>).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Whether an owner is alive, as the reader can tell.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Liveness {
    Alive,
    Dead,
    /// On another host, with a beat counter that did not change while watched.
    Stale { unchanged_for: Duration },
    /// Cannot be told: the start time of a process on this host cannot be
    /// read, or the beat file of an owner on another host cannot be read.
    Unknown,
}

impl Liveness {
    /// Whether a claim with this owner may still be in use: alive, or not
    /// known to have stopped.
    pub fn may_be_alive(&self) -> bool {
        !matches!(self, Liveness::Dead)
    }
}

/// This machine's host name; empty when it cannot be read.
pub fn this_host() -> &'static str {
    static HOST: OnceLock<String> = OnceLock::new();
    HOST.get_or_init(host_name)
}

#[cfg(unix)]
fn host_name() -> String {
    let mut buf = [0u8; 256];
    // SAFETY: the buffer is valid for its length; the name is cut at its first NUL.
    let rc = unsafe { libc::gethostname(buf.as_mut_ptr().cast(), buf.len()) };
    if rc != 0 {
        return String::new();
    }
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).into_owned()
}

#[cfg(not(unix))]
fn host_name() -> String {
    String::new()
}

/// This process as an owner.
pub fn this_owner() -> &'static Owner {
    static OWNER: OnceLock<Owner> = OnceLock::new();
    OWNER.get_or_init(|| {
        let pid = std::process::id();
        let started = process_start(pid).unwrap_or_else(|| {
            // A process whose own start time cannot be read names itself by the
            // time it first needed one; another process on this host then reads
            // it as Unknown, never as dead.
            Timestamp::now()
        });
        Owner { host: this_host().to_string(), pid, started }
    })
}

/// Whether a process with this id exists on this host.
#[cfg(unix)]
fn pid_exists(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    // SAFETY: signal 0 delivers nothing; it only probes for the process.
    let rc = unsafe { libc::kill(pid as libc::pid_t, 0) };
    rc == 0 || std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
}

#[cfg(not(unix))]
fn pid_exists(_pid: u32) -> bool {
    true
}

/// The start time of process `pid` on this host, when the OS reports one.
#[cfg(target_os = "macos")]
pub(crate) fn process_start(pid: u32) -> Option<Timestamp> {
    let mut info: libc::proc_bsdinfo = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of::<libc::proc_bsdinfo>() as libc::c_int;
    // SAFETY: `info` is a valid, writable proc_bsdinfo of `size` bytes.
    let n = unsafe {
        libc::proc_pidinfo(pid as libc::c_int, libc::PROC_PIDTBSDINFO, 0, (&mut info as *mut libc::proc_bsdinfo).cast(), size)
    };
    if n != size {
        return None;
    }
    let ms = info.pbi_start_tvsec as i64 * 1000 + info.pbi_start_tvusec as i64 / 1000;
    Some(Timestamp::from_unix_millis(ms))
}

/// The start time of process `pid` on this host: field 22 of
/// `/proc/<pid>/stat`, in clock ticks since boot, added to the boot time in
/// `/proc/stat`.
#[cfg(target_os = "linux")]
pub(crate) fn process_start(pid: u32) -> Option<Timestamp> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    // The command name, field 2, is in parentheses and may hold spaces.
    let after = &stat[stat.rfind(')')? + 1..];
    let ticks: i64 = after.split_whitespace().nth(19)?.parse().ok()?;
    let btime: i64 = std::fs::read_to_string("/proc/stat")
        .ok()?
        .lines()
        .find_map(|l| l.strip_prefix("btime "))?
        .trim()
        .parse()
        .ok()?;
    // SAFETY: sysconf has no preconditions.
    let hz = unsafe { libc::sysconf(libc::_SC_CLK_TCK) } as i64;
    if hz <= 0 {
        return None;
    }
    Some(Timestamp::from_unix_millis(btime * 1000 + ticks * 1000 / hz))
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub(crate) fn process_start(_pid: u32) -> Option<Timestamp> {
    None
}

/// The liveness of an owner on this host, with no waiting.
pub(crate) fn local_liveness(owner: &Owner) -> Liveness {
    if !pid_exists(owner.pid) {
        return Liveness::Dead;
    }
    match process_start(owner.pid) {
        Some(t) if t == owner.started => Liveness::Alive,
        Some(_) => Liveness::Dead,
        None => Liveness::Unknown,
    }
}

/// The liveness of the owner whose file stem ([`Owner::file_stem`]) is
/// `stem`, when that owner is on this host: dead when its process is gone or
/// its process id now names a process with another start time. `None` when
/// the stem names another host, or cannot be read as an owner of this one.
pub(crate) fn local_liveness_of_stem(stem: &str) -> Option<Liveness> {
    let rest = stem.strip_prefix(&format!("{}-", crate::layout::path_label(this_host())))?;
    let pid: u32 = rest.split('-').next()?.parse().ok()?;
    if !pid_exists(pid) {
        return Some(Liveness::Dead);
    }
    match process_start(pid) {
        Some(started) => {
            let now = Owner { host: this_host().to_string(), pid, started };
            Some(if now.file_stem() == stem { Liveness::Alive } else { Liveness::Dead })
        }
        None => Some(Liveness::Unknown),
    }
}

/// An owner's beat file, as written.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct BeatFile {
    pub format: u16,
    pub host: String,
    pub pid: u32,
    pub started: Timestamp,
    pub beat: u64,
    pub invocation: Option<u64>,
}

pub(crate) const BEAT_FORMAT: u16 = 1;

pub(crate) fn beat_path(root: &Path, owner: &Owner) -> PathBuf {
    root.join(".owners").join(format!("{}.json", owner.file_stem()))
}

pub(crate) fn read_beat(root: &Path, owner: &Owner) -> Option<u64> {
    let bytes = std::fs::read(beat_path(root, owner)).ok()?;
    serde_json::from_slice::<BeatFile>(&bytes).ok().map(|b| b.beat)
}

/// What a store remembers about the beat counters it has read, so a reader
/// that has looked once needs no wait the next time.
#[derive(Default)]
pub(crate) struct BeatMemory {
    seen: Mutex<std::collections::HashMap<Owner, (u64, Instant)>>,
}

impl BeatMemory {
    /// The liveness of `owner` read from another host: the counter now, and
    /// either the counter remembered from at least `wait` ago, or the counter
    /// again after waiting the rest of `wait`.
    pub(crate) fn remote_liveness(&self, root: &Path, owner: &Owner, wait: Duration) -> Liveness {
        let Some(now) = read_beat(root, owner) else { return Liveness::Unknown };
        let earlier = self.seen.lock().unwrap_or_else(|e| e.into_inner()).get(owner).copied();
        let rest = match earlier {
            Some((b, _)) if b != now => {
                self.remember(owner, now);
                return Liveness::Alive;
            }
            Some((_, at)) if at.elapsed() >= wait => {
                return Liveness::Stale { unchanged_for: at.elapsed() };
            }
            Some((_, at)) => wait - at.elapsed(),
            None => {
                self.remember(owner, now);
                wait
            }
        };
        std::thread::sleep(rest);
        let Some(after) = read_beat(root, owner) else { return Liveness::Unknown };
        if after != now {
            self.remember(owner, after);
            Liveness::Alive
        } else {
            Liveness::Stale { unchanged_for: wait }
        }
    }

    fn remember(&self, owner: &Owner, beat: u64) {
        let mut seen = self.seen.lock().unwrap_or_else(|e| e.into_inner());
        match seen.get(owner) {
            Some((b, _)) if *b == beat => {}
            _ => {
                seen.insert(owner.clone(), (beat, Instant::now()));
            }
        }
    }
}

/// This process's beat in one store root: the beat file, and the thread that
/// raises its counter. Removing the last handle stops the thread and removes
/// the file.
pub(crate) struct Beat {
    path: PathBuf,
    stop: Arc<AtomicBool>,
    invocation: Arc<AtomicU64>,
    thread: Option<std::thread::JoinHandle<()>>,
}

/// The invocation number a beat file names when none is set.
const NO_INVOCATION: u64 = u64::MAX;

impl Beat {
    pub(crate) fn start(root: &Path) -> Result<Beat> {
        let owner = this_owner().clone();
        let path = beat_path(root, &owner);
        let dir = path.parent().expect("a beat file has a directory");
        std::fs::create_dir_all(dir).at("create directory", dir)?;
        let invocation = Arc::new(AtomicU64::new(NO_INVOCATION));
        write_beat(&path, &owner, 0, None)?;
        let stop = Arc::new(AtomicBool::new(false));
        let thread = {
            let (path, stop, invocation) = (path.clone(), Arc::clone(&stop), Arc::clone(&invocation));
            std::thread::Builder::new()
                .name("store-beat".into())
                .spawn(move || {
                    let mut beat = 0u64;
                    let step = Duration::from_millis(100);
                    let mut waited = Duration::ZERO;
                    while !stop.load(Ordering::Relaxed) {
                        std::thread::sleep(step);
                        waited += step;
                        if waited < BEAT_INTERVAL {
                            continue;
                        }
                        waited = Duration::ZERO;
                        beat += 1;
                        let inv = invocation.load(Ordering::Relaxed);
                        // A failed write leaves the last beat in place; a reader
                        // on another host then sees the owner as stale, which is
                        // the safe reading.
                        let _ = write_beat(&path, &owner, beat, (inv != NO_INVOCATION).then_some(inv));
                    }
                })
                .map_err(|e| crate::Error::Io { op: "start the beat thread for", path: dir.to_path_buf(), source: e })?
        };
        Ok(Beat { path, stop, invocation, thread: Some(thread) })
    }

    pub(crate) fn set_invocation(&self, seq: u64) {
        self.invocation.store(seq, Ordering::Relaxed);
    }
}

fn write_beat(path: &Path, owner: &Owner, beat: u64, invocation: Option<u64>) -> Result<()> {
    let file = BeatFile {
        format: BEAT_FORMAT,
        host: owner.host.clone(),
        pid: owner.pid,
        started: owner.started.clone(),
        beat,
        invocation,
    };
    let bytes = serde_json::to_vec(&file).map_err(|e| json_err("encode", path, e))?;
    write_file_atomic(path, &bytes)
}

impl Drop for Beat {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        let _ = std::fs::remove_file(&self.path);
    }
}

#[cfg(test)]
mod tests;
