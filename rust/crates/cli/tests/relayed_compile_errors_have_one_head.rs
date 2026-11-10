//! A command that fails because camdlc refused the model relays camdlc's
//! report, which already opens with its own head (`error: nope.camdl: No such
//! file or directory`, `error[E310]: …`). The command must not put a second
//! `error: ` in front of it: `simulate nope.camdl` printed
//! `error: error: nope.camdl: No such file or directory`.

use std::path::PathBuf;
use std::process::Command;

fn manifest() -> PathBuf {
    PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
}

/// A model camdlc refuses with a coded diagnostic (E310: a transition whose
/// sources and destinations cancel).
const BROKEN: &str = "time_unit = 'days\ncompartments { S }\nparameters { b : rate }\ntransitions { x : S --> S @ b }\n";

#[test]
fn a_relayed_compile_error_is_headed_once() {
    let bin = manifest().join("../../target/release/camdl");
    assert!(bin.exists(), "release camdl binary missing: {} — run `make build-rust`", bin.display());
    let tmp = std::env::temp_dir().join(format!("camdl_one_head_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("broken.camdl"), BROKEN).unwrap();

    let cases: [&[&str]; 4] = [
        &["simulate", "nope.camdl"],
        &["pfilter", "nope.camdl", "--particles", "5"],
        &["simulate", "broken.camdl"],
        &["pfilter", "broken.camdl", "--particles", "5"],
    ];
    for args in cases {
        let out = Command::new(&bin)
            .env_remove("CLICOLOR_FORCE")
            .env("NO_COLOR", "1")
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .env("CAMDL_NO_IR_CACHE", "1")
            .current_dir(&tmp)
            .args(args)
            .output()
            .expect("spawn camdl");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(!out.status.success(), "{args:?} must fail:\n{stderr}");
        let error_lines: Vec<&str> = stderr.lines().filter(|l| l.starts_with("error")).collect();
        assert_eq!(error_lines.len(), 1, "{args:?}: one error line:\n{stderr}");
        let line = error_lines[0];
        assert!(
            !line.starts_with("error: error"),
            "{args:?}: the relayed report is headed twice:\n{stderr}"
        );
        if args[1] == "nope.camdl" {
            assert_eq!(line, "error: nope.camdl: No such file or directory", "{args:?}");
        } else {
            assert!(line.starts_with("error[E310]: transition 'x' has no net effect"), "{args:?}: {line}");
        }
    }
    let _ = std::fs::remove_dir_all(&tmp);
}
