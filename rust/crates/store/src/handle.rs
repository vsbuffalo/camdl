//! Handles: how a command line names a stored result.

use std::marker::PhantomData;
use std::path::PathBuf;

use crate::artifact::{Artifact, Id};
use crate::error::{Error, Result};
use crate::hash::Key;
use crate::invocation::Invocation;
use crate::record::{read_record, IdentityFile, IDENTITY_FILE};
use crate::storage::{label_holders, Store};
use crate::stored::{Entry, Stored};

/// A handle as typed: a path, an id prefix, `@L` or `@L~n`.
///
/// Parsing is syntactic. `@L` and `@L~n` are labels; a string of hex digits
/// that names no file or directory is an id prefix; anything else is a path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Handle {
    Path(PathBuf),
    IdPrefix(String),
    /// `@L` is `back = 0`.
    Label { name: String, back: u32 },
}

impl std::str::FromStr for Handle {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        if let Some(rest) = s.strip_prefix('@') {
            let (name, back) = match rest.rsplit_once('~') {
                Some((n, b)) => match b.parse::<u32>() {
                    Ok(b) => (n, b),
                    Err(_) => return Err(Error::NotFound { handle: s.to_string() }),
                },
                None => (rest, 0),
            };
            if name.is_empty() {
                return Err(Error::NotFound { handle: s.to_string() });
            }
            return Ok(Handle::Label { name: name.to_string(), back });
        }
        let hex = !s.is_empty() && s.len() <= 64 && s.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase());
        if hex && !std::path::Path::new(s).exists() {
            Ok(Handle::IdPrefix(s.to_string()))
        } else {
            Ok(Handle::Path(PathBuf::from(s)))
        }
    }
}

impl std::fmt::Display for Handle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Handle::Path(p) => write!(f, "{}", p.display()),
            Handle::IdPrefix(p) => f.write_str(p),
            Handle::Label { name, back: 0 } => write!(f, "@{name}"),
            Handle::Label { name, back } => write!(f, "@{name}~{back}"),
        }
    }
}

/// A handle that must name a result of kind `A`.
pub struct RunRef<A: Artifact> {
    pub handle: Handle,
    _kind: PhantomData<fn() -> A>,
}

impl<A: Artifact> RunRef<A> {
    pub fn new(handle: Handle) -> Self {
        RunRef { handle, _kind: PhantomData }
    }
}

impl<A: Artifact> Clone for RunRef<A> {
    fn clone(&self) -> Self {
        RunRef::new(self.handle.clone())
    }
}

impl<A: Artifact> std::fmt::Debug for RunRef<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RunRef<{}>({})", A::KIND, self.handle)
    }
}

impl<A: Artifact> std::str::FromStr for RunRef<A> {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(RunRef::new(s.parse()?))
    }
}

impl Store {
    /// The run id a handle names, with its kind as its record states it.
    pub(crate) fn resolve_key(&self, handle: &Handle, kind: &'static str) -> Result<(Key, String)> {
        let text = handle.to_string();
        match handle {
            Handle::Path(p) => {
                let f: IdentityFile = match read_record(&p.join(IDENTITY_FILE)) {
                    crate::record::Read::Absent => return Err(Error::NoRecord { path: p.clone() }),
                    r => r.required(&p.join(IDENTITY_FILE))?,
                };
                Ok((f.run_id, f.kind))
            }
            Handle::IdPrefix(prefix) => {
                // A full run id is found through the index, confirmed against
                // its folder, before any walk.
                if let Ok(key) = Key::from_hex(prefix) {
                    if let Some((_, f)) = self.inner.find_any_kind(&key)? {
                        return Ok((f.run_id, f.kind));
                    }
                    return Err(Error::NotFound { handle: text });
                }
                let found = crate::maintenance::walk(&self.inner);
                let mut matches: Vec<(Key, String)> =
                    found.into_iter().filter(|(_, f)| f.run_id.to_hex().starts_with(prefix.as_str())).map(|(_, f)| (f.run_id, f.kind)).collect();
                matches.sort();
                matches.dedup();
                match matches.len() {
                    0 => Err(Error::NotFound { handle: text }),
                    1 => Ok(matches.remove(0)),
                    _ => Err(Error::Ambiguous { handle: text, matches: matches.into_iter().map(|(k, _)| k).collect() }),
                }
            }
            Handle::Label { .. } => {
                let inv = self.invocation(handle)?;
                let members: Vec<Key> = inv.record.listings().iter().filter(|r| r.kind == kind).map(|r| r.run_id).collect();
                let mut distinct = members.clone();
                distinct.sort();
                distinct.dedup();
                match distinct.len() {
                    1 => Ok((distinct[0], kind.to_string())),
                    // The invocation holds no result of this kind: whatever it
                    // holds is of other kinds, named in the refusal.
                    0 => {
                        let listings = inv.record.listings();
                        let mut kinds: Vec<&str> = listings.iter().map(|r| r.kind).collect();
                        kinds.sort();
                        kinds.dedup();
                        Err(Error::WrongKind { handle: text, expected: kind, found: kinds.join(", ") })
                    }
                    _ => Err(Error::NotOneResult { handle: text, members: distinct }),
                }
            }
        }
    }

    /// The completed result a handle names. A label that names an invocation
    /// with several results of kind `A` is refused with
    /// [`Error::NotOneResult`].
    pub fn resolve<A: Artifact>(&self, handle: &RunRef<A>) -> Result<Stored<A>> {
        let (key, kind) = self.resolve_key(&handle.handle, A::KIND)?;
        if kind != A::KIND {
            return Err(Error::WrongKind { handle: handle.handle.to_string(), expected: A::KIND, found: kind });
        }
        self.get(&Id::<A>::from_key(key))?.ok_or_else(|| Error::NotFound { handle: handle.handle.to_string() })
    }

    /// The result a handle names, in any status.
    pub fn resolve_entry<A: Artifact>(&self, handle: &RunRef<A>) -> Result<Entry<A>> {
        let (key, kind) = self.resolve_key(&handle.handle, A::KIND)?;
        if kind != A::KIND {
            return Err(Error::WrongKind { handle: handle.handle.to_string(), expected: A::KIND, found: kind });
        }
        self.inspect(&Id::<A>::from_key(key))?.ok_or_else(|| Error::NotFound { handle: handle.handle.to_string() })
    }

    /// The invocations that hold or held `label`, newest first: `@L` is the
    /// first, `@L~n` the one `n` places after it. An invocation whose results
    /// did not all complete holds no label.
    pub fn label_holders(&self, label: &str) -> Vec<Invocation> {
        label_holders(&self.inner, label)
    }

    /// The invocation that answers for what a handle names
    /// ([`Store::listings`]): for `@L` and `@L~n`, that invocation; for a
    /// path or an id, the newest that listed the result computed or reused,
    /// else the newest running one that lists it. Refused with
    /// [`Error::NotFound`] when there is none.
    pub fn invocation(&self, handle: &Handle) -> Result<Invocation> {
        self.listings(handle)?
            .answering()
            .map(|l| l.invocation().clone())
            .ok_or_else(|| Error::NotFound { handle: handle.to_string() })
    }
}

#[cfg(test)]
mod tests;
