//! Installing the shell scripts that integration tests put on `PATH` in place
//! of a real program, such as a `camdlc` that counts its compiles.
//!
//! A test binary runs its tests on several threads, and each thread spawns
//! processes. On Linux, `execve` of a file fails with `ETXTBSY` ("text file
//! busy") while any process holds that file open for writing. A script
//! written with `std::fs::write` is open for writing in the test process for
//! the length of the write, and a child that another thread forks in that
//! interval inherits the descriptor and holds it until its own `exec` closes
//! it. A `camdl` started on the script while that child has not yet reached
//! `exec` cannot run it: the spawn of `camdlc --camdl-version` fails, and
//! `camdl` reports "camdlc not found" or skips the IR cache for that run.
//!
//! [`install_script`] has a child `sh` write the file, so the test process
//! never holds a writable descriptor to it and no process the test forks can
//! inherit one. When `install_script` returns, the processes that wrote the
//! file have exited and nothing holds it open for writing.

use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

/// Writes `script` to `path` as an executable file (mode 0755), from a child
/// `sh`, so that no process forked by this test binary holds the file open for
/// writing afterwards. Panics if the file cannot be written.
pub fn install_script(path: &Path, script: &str) {
    let mut sh = Command::new("sh")
        .args(["-c", "cat > \"$1\" && chmod 755 \"$1\"", "sh"])
        .arg(path)
        .stdin(Stdio::piped())
        .spawn()
        .expect("spawn sh to install a script");
    sh.stdin
        .take()
        .expect("sh has a piped stdin")
        .write_all(script.as_bytes())
        .expect("write the script to sh");
    let status = sh.wait().expect("wait for sh");
    assert!(status.success(), "installing the script {} failed: {status}", path.display());
}
