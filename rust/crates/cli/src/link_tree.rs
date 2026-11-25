//! The link tree: symbolic links to a command's stored result files, under a
//! directory a person can browse, written when the command has a link root
//! ([`link_root`]). The layout and the lifecycle are `docs/camdl-run-spec.md`
//! §2.11.
//!
//! A command claims its label's links ([`LinkTree::claim`]) when it claims
//! its output manifest, before its first call to the store: every link
//! directory of the label is removed, in every group, so a command that then
//! fails, or is killed, leaves no links that look current. Once the command
//! has succeeded it writes a link for each result file of each result its
//! output manifest lists ([`LinkTree::write`]). Every place is checked before
//! any directory is written, and a failure part way removes what was
//! written, so a refused write leaves no links. Each link directory is built
//! beside its place and renamed into it, so a reader finds the earlier
//! directory, none, or the new one, and never a mix of two runs' links.
//!
//! Each link directory holds [`RECORD`]: the directory's name as camdl gave
//! it, and the links camdl wrote in it with their targets. A directory is
//! removed or replaced only when its record names it byte for byte and every
//! entry in it is that record, a directory, or a link the record names with
//! the target it records; an empty directory is removed too. A directory is
//! renamed aside before it is checked, so what is checked is what is removed.
//! Anything else at a link directory's path, a file, a symbolic link, a
//! directory holding something camdl did not write, or one whose record names
//! another label that the filesystem resolves to the same directory, is
//! refused by its path and left as it is.
//!
//! Links are an export. They are never part of an identity, nothing camdl
//! computes reads them, and they follow the label, which moves when a later
//! command takes it. A workflow step that must name exactly what an earlier
//! step made depends on that step's output manifest, whose run ids and
//! digests do not move.

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use store::Key;

use crate::output_manifest::{ManifestEntry, OutputManifest};

/// The record of the links camdl wrote in a link directory.
pub(crate) const RECORD: &str = ".camdl-links.json";

/// The record's `schema` value.
const RECORD_SCHEMA: &str = "camdl.link-dir/v1";

/// The groups of the link tree, one per kind that gets links of its own.
const GROUPS: [&str; 6] = ["fits", "sims", "filters", "predictions", "profiles", "scores"];

/// The link root of a command that stores results: `flag` (`--link-root`),
/// else `CAMDL_LINK_ROOT`, else none. A relative root is relative to the
/// working directory, as a relative store root is (run spec §2.1), and is
/// returned joined to it, so the links land where the command was run from
/// whatever directory a later step works in.
pub(crate) fn link_root(flag: Option<&Path>) -> Option<PathBuf> {
    let cwd = std::env::current_dir().unwrap_or_default();
    root_from(flag, crate::env::get().vars.link_root.as_deref(), &cwd)
}

/// [`link_root`], given the value of `CAMDL_LINK_ROOT` and the working
/// directory. An empty flag or variable names no root.
fn root_from(flag: Option<&Path>, env: Option<&Path>, cwd: &Path) -> Option<PathBuf> {
    let given = [flag, env].into_iter().flatten().find(|p| !p.as_os_str().is_empty())?;
    Some(if given.is_absolute() { given.to_path_buf() } else { cwd.join(given) })
}

/// Where a result's links go.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Placement {
    /// Its own directory in this group.
    Group(&'static str),
    /// Its upstream's directory, beside the upstream's files: a by-product.
    WithUpstream,
    /// Nowhere: a content artifact or a fit problem, which a command stores
    /// as an input of the results it computes.
    None,
}

/// Where the links of a result of `kind` go; `None` for a kind camdl does
/// not store.
fn placement(kind: &str) -> Option<Placement> {
    Some(match kind {
        "fit_run" => Placement::Group("fits"),
        "sim" => Placement::Group("sims"),
        "pfilter" => Placement::Group("filters"),
        "prediction" => Placement::Group("predictions"),
        "profile_point" => Placement::Group("profiles"),
        "score" => Placement::Group("scores"),
        "obs" | "event_log" | "sim_quantities" | "trace" | "pf_health" | "filtering" | "filtered_streams" | "final_state"
        | "saved_paths" | "prequential" => Placement::WithUpstream,
        "compiled_model" | "quantities" | "data_table" | "model_source" | "fit" => Placement::None,
        _ => return None,
    })
}

/// A link's name in its directory: the file's logical name, then the stored
/// name's extension (`chain_0/trace` and `chain_0/trace.tsv` give
/// `chain_0/trace.tsv`; `joint` and `prequential.tsv` give `joint.tsv`).
fn link_name(logical: &str, stored: &str) -> String {
    match Path::new(stored).extension().and_then(|e| e.to_str()) {
        Some(ext) => format!("{logical}.{ext}"),
        None => logical.to_string(),
    }
}

/// The record a link directory holds: the name camdl gave the directory (the
/// label, or the run id's digits), and each link camdl wrote in it, by its
/// path relative to the directory, with the target as written. The name is
/// compared byte for byte, so two labels that a filesystem resolves to one
/// directory (`Up` and `up` on a case-insensitive volume, or two Unicode
/// normalizations of one word) never take each other's links.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    schema: String,
    name: String,
    links: BTreeMap<String, PathBuf>,
}

/// The link tree of one command: its absolute root and its `--label`.
#[derive(Debug, Clone)]
pub(crate) struct LinkTree {
    root: PathBuf,
    label: Option<String>,
}

/// One link directory to write: its group, its name in the group, and its
/// links by relative path, each with the stored file it points at.
struct LinkDir {
    group: &'static str,
    name: String,
    links: BTreeMap<String, PathBuf>,
}

impl LinkTree {
    /// Claims the links of `label` under `root`, an absolute path: refuses a
    /// root that is not a directory, a label beginning with `.`, and anything at one of the label's link directories that camdl did
    /// not write, then removes every link directory of the label, and the
    /// temporary directories of the label that a killed command left. Each
    /// directory is renamed aside before it is checked, so what is checked is
    /// what is removed; nothing is removed unless every one passes. A command
    /// with no label names its results' directories by run id, and has none
    /// to remove before it runs.
    ///
    /// A [`store::Label`] is one to 64 ASCII characters with no `/`, so every
    /// other label names a directory, and its temporary names (`label` plus at
    /// most 48 bytes) fit the 255 bytes a file name can have on Linux and
    /// macOS.
    pub(crate) fn claim(root: &Path, label: Option<&store::Label>) -> Result<LinkTree, String> {
        let shown = root.display();
        let refuse = |e: String| format!("--link-root {shown}: {e}");
        if !cfg!(unix) {
            return Err(refuse("symbolic links are written only on Linux and macOS".into()));
        }
        if root.symlink_metadata().is_ok() && !root.is_dir() {
            return Err(refuse("is not a directory".into()));
        }
        if let Some(l) = label.map(store::Label::as_str) {
            if l.starts_with('.') {
                return Err(refuse(format!(
                    "the label `{l}` names its link directories, and is not a directory name: \
                     it must not begin with `.`"
                )));
            }
            for g in GROUPS {
                clean_temporaries(&root.join(g), l).map_err(refuse)?;
            }
            let tag = temporary_tag();
            let mut aside: Vec<(PathBuf, PathBuf)> = Vec::new();
            let mut taken = Ok(());
            for g in GROUPS {
                let place = root.join(g).join(l);
                match take_aside(&place, l, &tag) {
                    Ok(Some(old)) => aside.push((place, old)),
                    Ok(None) => {}
                    Err(e) => {
                        taken = Err(e);
                        break;
                    }
                }
            }
            if let Err(e) = taken {
                for (place, old) in aside.iter().rev() {
                    let _ = std::fs::rename(old, place);
                }
                return Err(refuse(e));
            }
            for (place, old) in &aside {
                std::fs::remove_dir_all(old).map_err(|e| refuse(format!("cannot remove the earlier links {}: {e}", place.display())))?;
            }
        }
        Ok(LinkTree { root: root.to_path_buf(), label: label.map(|l| l.as_str().to_string()) })
    }

    /// Writes the links of `manifest`'s results, which `store` holds, and
    /// returns the link directories written. Every directory's place is
    /// checked before any is written, and a failure part way removes the
    /// directories already written, so a refused write leaves none. Refused
    /// when the link root, or a group directory, lies in the run store; when
    /// a link directory's place holds something camdl did not write; and when
    /// a written link does not resolve to its stored file.
    pub(crate) fn write(&self, store: &store::Store, manifest: &OutputManifest) -> Result<Vec<PathBuf>, String> {
        let shown = self.root.display();
        let refuse = |e: String| format!("--link-root {shown}: {e}");
        let dirs = self.plan(store, &manifest.results).map_err(refuse)?;
        if dirs.is_empty() {
            return Ok(Vec::new());
        }
        let store_root = canonical(store.root()).map_err(refuse)?;
        let outside_the_store = |p: &Path| -> Result<(), String> {
            let resolved = resolved_prefix(p).map_err(refuse)?;
            if resolved.starts_with(&store_root) {
                return Err(refuse(format!(
                    "{} is inside the run store {}; links are written outside it",
                    p.display(),
                    store_root.display()
                )));
            }
            Ok(())
        };
        outside_the_store(&self.root)?;
        // Each group directory, as created and with every symbolic link in
        // it resolved: links are computed from where they will sit.
        let mut groups: BTreeMap<&str, PathBuf> = BTreeMap::new();
        for d in &dirs {
            if groups.contains_key(d.group) {
                continue;
            }
            let group = self.root.join(d.group);
            outside_the_store(&group)?;
            std::fs::create_dir_all(&group).map_err(|e| refuse(format!("{}: {e}", group.display())))?;
            let resolved = canonical(&group).map_err(refuse)?;
            outside_the_store(&resolved)?;
            groups.insert(d.group, resolved);
        }
        for d in &dirs {
            let place = groups[d.group].join(&d.name);
            ours(&place, &self.root.join(d.group).join(&d.name), &d.name).map_err(refuse)?;
        }
        let mut written = Vec::new();
        for d in &dirs {
            let group = &groups[d.group];
            let relative = shares_a_parent(&store_root, group);
            let place = group.join(&d.name);
            let mut record = Record { schema: RECORD_SCHEMA.into(), name: d.name.clone(), links: BTreeMap::new() };
            for (rel, target) in &d.links {
                let from = place.join(rel);
                let at = if relative { relative_path(from.parent().expect("a link has a directory"), target) } else { target.clone() };
                record.links.insert(rel.clone(), at);
            }
            let done = write_dir(group, &d.name, &record).and_then(|()| {
                written.push(place.clone());
                resolves(&place, &d.links)
            });
            if let Err(e) = done {
                Self::remove(&written);
                return Err(refuse(e));
            }
        }
        Ok(written)
    }

    /// Removes the link directories `dirs`, which [`LinkTree::write`] wrote:
    /// for a command that fails after its links were written.
    pub(crate) fn remove(dirs: &[PathBuf]) {
        for d in dirs {
            let _ = std::fs::remove_dir_all(d);
        }
    }

    /// The link directories of `results`, with the stored file each link
    /// points at, by absolute canonical path.
    fn plan(&self, store: &store::Store, results: &[ManifestEntry]) -> Result<Vec<LinkDir>, String> {
        let mut per_group: BTreeMap<&str, Vec<Key>> = BTreeMap::new();
        for r in results {
            match placement(&r.kind) {
                Some(Placement::Group(g)) => per_group.entry(g).or_default().push(r.run_id),
                Some(_) => {}
                None => return Err(format!("internal error: the result {} is of kind `{}`, which has no place in the link tree", r.run_id.short12(), r.kind)),
            }
        }
        // A run id's first 12 hex digits name its directory, unless another
        // result of the group shares them.
        let short = |g: &str, id: Key| -> String {
            let s = id.short12();
            if per_group[g].iter().filter(|o| o.short12() == s).count() > 1 { id.to_string() } else { s }
        };
        let mut dirs: Vec<LinkDir> = Vec::new();
        // Each primary result's directory, and the prefix of its links in it.
        let mut at: BTreeMap<Key, (usize, String)> = BTreeMap::new();
        for r in results {
            let Some(Placement::Group(g)) = placement(&r.kind) else { continue };
            let (name, prefix) = match &self.label {
                Some(l) if per_group[g].len() == 1 => (l.clone(), String::new()),
                Some(l) => (l.clone(), format!("{}/", short(g, r.run_id))),
                None => (short(g, r.run_id), String::new()),
            };
            let i = match dirs.iter().position(|d| d.group == g && d.name == name) {
                Some(i) => i,
                None => {
                    dirs.push(LinkDir { group: g, name, links: BTreeMap::new() });
                    dirs.len() - 1
                }
            };
            at.insert(r.run_id, (i, prefix));
        }
        for r in results {
            let (i, prefix, by_product) = match placement(&r.kind) {
                Some(Placement::Group(_)) => {
                    let (i, p) = &at[&r.run_id];
                    (*i, p.clone(), false)
                }
                Some(Placement::WithUpstream) => {
                    let up = r.upstream.ok_or_else(|| format!("internal error: the {} result {} has no upstream", r.kind, r.run_id.short12()))?;
                    let (i, p) = at.get(&up).ok_or_else(|| {
                        format!("internal error: the {} result {} is listed without its upstream {}", r.kind, r.run_id.short12(), up.short12())
                    })?;
                    (*i, p.clone(), true)
                }
                _ => continue,
            };
            let folder = canonical(store.locate(r.run_id).map_err(|e| e.to_string())?.folder())?;
            for (logical, f) in &r.files {
                let name = link_name(logical, &f.file);
                let links = &mut dirs[i].links;
                // A by-product's file whose name its upstream already uses
                // goes under the by-product's kind.
                let rel = if by_product && links.contains_key(&format!("{prefix}{name}")) {
                    format!("{prefix}{}/{name}", r.kind)
                } else {
                    format!("{prefix}{name}")
                };
                if links.insert(rel.clone(), folder.join(&f.file)).is_some() {
                    return Err(format!("internal error: two files of one link directory are named `{rel}`"));
                }
            }
        }
        Ok(dirs)
    }
}

/// The absolute path of `p` with every symbolic link resolved.
fn canonical(p: &Path) -> Result<PathBuf, String> {
    p.canonicalize().map_err(|e| format!("{}: {e}", p.display()))
}

/// Whether `a` and `b`, absolute and canonical, share a directory below the
/// filesystem's root.
fn shares_a_parent(a: &Path, b: &Path) -> bool {
    a.components().zip(b.components()).take_while(|(x, y)| x == y).filter(|(x, _)| matches!(x, Component::Normal(_))).count() > 0
}

/// The path of `to` relative to the directory `from`, both absolute and
/// canonical.
fn relative_path(from: &Path, to: &Path) -> PathBuf {
    let (f, t): (Vec<_>, Vec<_>) = (from.components().collect(), to.components().collect());
    let common = f.iter().zip(&t).take_while(|(x, y)| x == y).count();
    let mut out = PathBuf::new();
    for _ in common..f.len() {
        out.push("..");
    }
    for c in &t[common..] {
        out.push(c);
    }
    out
}

/// `p`, absolute, with every symbolic link in its longest existing prefix
/// resolved and the rest appended: where `p` is, or would be once created.
fn resolved_prefix(p: &Path) -> Result<PathBuf, String> {
    let mut rest = Vec::new();
    let mut at = p;
    loop {
        if at.symlink_metadata().is_ok() {
            let mut out = canonical(at)?;
            out.extend(rest.iter().rev());
            return Ok(out);
        }
        match (at.parent(), at.file_name()) {
            (Some(parent), Some(name)) => {
                rest.push(name.to_os_string());
                at = parent;
            }
            _ => return Ok(p.to_path_buf()),
        }
    }
}

/// Whether a link directory named `name` is at `dir`, which camdl may
/// replace: `false` when nothing is there, `true` for an empty directory or
/// one that holds only [`RECORD`], naming `name`, directories, and the links
/// the record names with their recorded targets. Anything else is refused,
/// naming the first path camdl did not write; paths are shown under `shown`,
/// the directory's place, when `dir` is a copy renamed aside.
fn ours(dir: &Path, shown: &Path, name: &str) -> Result<bool, String> {
    let meta = match dir.symlink_metadata() {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(e) => return Err(format!("{}: {e}", shown.display())),
    };
    let not_ours = |p: &Path, what: &str| format!("{} is {what} that camdl did not write; move it, or use another --label or --link-root", p.display());
    if meta.file_type().is_symlink() {
        return Err(not_ours(shown, "a symbolic link"));
    }
    if !meta.is_dir() {
        return Err(not_ours(shown, "a file"));
    }
    let record = read_record(dir, shown)?;
    if let Some(r) = &record {
        if r.name != name {
            return Err(format!(
                "{} holds the links of `{}`, not of `{name}`: this filesystem names the two by one directory, \
                 as a case-insensitive volume does `Up` and `up`; use another --label or --link-root",
                shown.display(),
                r.name
            ));
        }
    }
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let entries = std::fs::read_dir(&d).map_err(|e| format!("{}: {e}", d.display()))?;
        for e in entries {
            let p = e.map_err(|e| format!("{}: {e}", d.display()))?.path();
            let rel = p.strip_prefix(dir).expect("under the directory").to_string_lossy().into_owned();
            let at = shown.join(&rel);
            let ft = p.symlink_metadata().map_err(|e| format!("{}: {e}", at.display()))?.file_type();
            if ft.is_symlink() {
                let target = std::fs::read_link(&p).map_err(|e| format!("{}: {e}", at.display()))?;
                if record.as_ref().and_then(|r| r.links.get(&rel)) != Some(&target) {
                    return Err(not_ours(&at, "a symbolic link"));
                }
            } else if ft.is_dir() {
                stack.push(p);
            } else if rel != RECORD || record.is_none() {
                return Err(not_ours(&at, "a file"));
            }
        }
    }
    Ok(true)
}

/// The record of the link directory `dir`, or `None` when it holds none.
#[expect(clippy::disallowed_methods, reason = "reads the record of links camdl wrote in a link directory, an export it is about to replace, to refuse replacing anything else")]
fn read_record(dir: &Path, shown: &Path) -> Result<Option<Record>, String> {
    let path = dir.join(RECORD);
    let at = shown.join(RECORD);
    match std::fs::read(&path) {
        Ok(bytes) => {
            let record: Record =
                serde_json::from_slice(&bytes).map_err(|e| format!("{} is not a record of links camdl wrote: {e}", at.display()))?;
            if record.schema != RECORD_SCHEMA {
                return Err(format!("{} is a record of version `{}`; this camdl reads `{RECORD_SCHEMA}`", at.display(), record.schema));
            }
            Ok(Some(record))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("{}: {e}", at.display())),
    }
}

/// A suffix no other command's temporary directory has: this process's id
/// and the time.
fn temporary_tag() -> String {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    format!("{}.{nanos}", std::process::id())
}

/// Renames the link directory `name` at `place` aside, to a temporary name
/// beside it, and checks the renamed copy ([`ours`]): `None` when nothing is
/// there, the copy's path when it passes. A copy that fails is renamed back,
/// and the refusal returned. A file or a symbolic link at `place` is refused
/// without being moved.
fn take_aside(place: &Path, name: &str, tag: &str) -> Result<Option<PathBuf>, String> {
    match place.symlink_metadata() {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("{}: {e}", place.display())),
        Ok(m) if !m.is_dir() => return ours(place, place, name).map(|_| None),
        Ok(_) => {}
    }
    let old = place.with_file_name(format!(".{name}.camdl-old.{tag}"));
    std::fs::rename(place, &old).map_err(|e| format!("{}: {e}", place.display()))?;
    match ours(&old, place, name) {
        Ok(_) => Ok(Some(old)),
        Err(e) => {
            let _ = std::fs::rename(&old, place);
            Err(e)
        }
    }
}

/// Removes the temporary directories of the label `label` in the group
/// directory `group` that a killed command left: each
/// `.<label>.camdl-new.<pid>.…` or `.<label>.camdl-old.<pid>.…` whose process
/// is no longer running on this host. One that holds anything camdl did not
/// write is refused.
fn clean_temporaries(group: &Path, label: &str) -> Result<(), String> {
    let Ok(entries) = std::fs::read_dir(group) else { return Ok(()) };
    for e in entries.flatten() {
        let file_name = e.file_name().to_string_lossy().into_owned();
        let Some(rest) = [".camdl-new.", ".camdl-old."]
            .iter()
            .find_map(|kind| file_name.strip_prefix(&format!(".{label}{kind}")).map(str::to_string))
        else {
            continue;
        };
        let Some(pid) = rest.split('.').next().and_then(|p| p.parse::<u32>().ok()) else { continue };
        if store::pid_exists(pid) {
            continue;
        }
        let p = e.path();
        if ours(&p, &p, label)? {
            std::fs::remove_dir_all(&p).map_err(|e| format!("cannot remove {}: {e}", p.display()))?;
        }
    }
    Ok(())
}

/// Writes the link directory `name` of `group` holding `record`'s links:
/// built at a temporary name in `group`, the record first, and renamed into
/// place ([`swap_in`]).
fn write_dir(group: &Path, name: &str, record: &Record) -> Result<(), String> {
    let tag = temporary_tag();
    let new = group.join(format!(".{name}.camdl-new.{tag}"));
    let io = |p: &Path, e: std::io::Error| format!("{}: {e}", p.display());
    std::fs::create_dir(&new).map_err(|e| io(&new, e))?;
    let built = (|| -> Result<(), String> {
        let mut bytes = serde_json::to_vec_pretty(record).expect("a record serializes");
        bytes.push(b'\n');
        write_new(&new.join(RECORD), &bytes).map_err(|e| io(&new.join(RECORD), e))?;
        for (rel, target) in &record.links {
            let at = new.join(rel);
            if let Some(parent) = at.parent() {
                std::fs::create_dir_all(parent).map_err(|e| io(parent, e))?;
            }
            symlink(target, &at).map_err(|e| io(&at, e))?;
        }
        Ok(())
    })();
    let placed = built.and_then(|()| swap_in(group, name, &new, &tag));
    if placed.is_err() {
        let _ = std::fs::remove_dir_all(&new);
    }
    placed
}

/// Writes `bytes` to a file at `path` that must not exist yet, so a link
/// planted at `path` is never followed.
#[expect(clippy::disallowed_methods, reason = "creates the record of a link directory camdl has just made; nothing is read")]
fn write_new(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    std::fs::OpenOptions::new().write(true).create_new(true).open(path)?.write_all(bytes)
}

/// Renames the built directory `new` to `name` in `group`, first renaming
/// aside, checking and then removing a link directory camdl wrote there.
fn swap_in(group: &Path, name: &str, new: &Path, tag: &str) -> Result<(), String> {
    let place = group.join(name);
    for _ in 0..3 {
        if let Some(old) = take_aside(&place, name, tag)? {
            return match std::fs::rename(new, &place) {
                Ok(()) => {
                    let _ = std::fs::remove_dir_all(&old);
                    Ok(())
                }
                Err(e) => {
                    let _ = std::fs::rename(&old, &place);
                    Err(format!("{}: {e}", place.display()))
                }
            };
        }
        match std::fs::rename(new, &place) {
            Ok(()) => return Ok(()),
            // Another command placed a directory there meanwhile.
            Err(_) if place.symlink_metadata().is_ok() => continue,
            Err(e) => return Err(format!("{}: {e}", place.display())),
        }
    }
    Err(format!("{}: other commands kept replacing it", place.display()))
}

/// Checks that each link of the directory at `place` resolves to the stored
/// file `links` names for it.
fn resolves(place: &Path, links: &BTreeMap<String, PathBuf>) -> Result<(), String> {
    for (rel, target) in links {
        let at = place.join(rel);
        match at.canonicalize() {
            Ok(got) if &got == target => {}
            Ok(got) => return Err(format!("the link {} resolves to {}, not to the stored file {}", at.display(), got.display(), target.display())),
            Err(e) => return Err(format!("the link {} does not resolve to the stored file {}: {e}", at.display(), target.display())),
        }
    }
    Ok(())
}

#[cfg(unix)]
fn symlink(target: &Path, at: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(target, at)
}

#[cfg(not(unix))]
fn symlink(_target: &Path, _at: &Path) -> std::io::Result<()> {
    Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "symbolic links are written only on Linux and macOS"))
}

#[cfg(test)]
mod tests;
