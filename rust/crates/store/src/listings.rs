//! Which invocation answers for a stored result: [`Store::listings`].
//!
//! A handle names a result, and every invocation that listed it gave it
//! attributes of its own. `listings` returns each invocation's listing of what
//! the handle names, and decides which one answers:
//!
//! - a label (`@f`, `@f~1`) names one invocation, which answers alone;
//! - a run id, a prefix or a folder names every closed listing of the result
//!   that ended computed or reused, newest first, and the newest answers. When
//!   none did, the newest listing of it in a running record answers, marked
//!   unfinished.
//!
//! The closed listings come from the store's snapshot of the records
//! ([`Store::invocations`]); the running ones are read from the directory,
//! since another process's running record is in no snapshot. That read takes
//! no lock and writes nothing: each record is replaced by renaming a whole
//! version over it.

use crate::artifact::{Artifact, Id};
use crate::error::{Error, Result};
use crate::handle::Handle;
use crate::hash::Key;
use crate::invocation::Invocation;
use crate::label::Label;
use crate::record::Record;
use crate::storage::{label_holders, Store};

/// Every listing of what a handle names, newest first, and how the handle
/// named it.
#[derive(Debug, Clone)]
pub struct Listings {
    by: By,
    listed: Vec<Listed>,
}

/// How a handle named what [`Listings`] lists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum By {
    /// A label: the one invocation, by its sequence number.
    Label { seq: u64 },
    /// A run id, a prefix or a folder: the result, by its run id.
    Result { run_id: Key },
}

/// One invocation's listing of what a handle names.
#[derive(Debug, Clone)]
pub struct Listed {
    pub seq: u64,
    /// The invocation's `--label`.
    pub label: Option<Label>,
    /// True for a listing in a running record, which answers only when no
    /// closed invocation listed the result computed or reused.
    pub unfinished: bool,
    invocation: Invocation,
    /// The result the handle named, for [`By::Result`].
    result: Option<Key>,
}

impl Listed {
    fn new(invocation: Invocation, unfinished: bool, result: Option<Key>) -> Listed {
        let label = invocation.record.head().label.clone();
        Listed { seq: invocation.seq(), label, unfinished, invocation, result }
    }

    /// The invocation, whole.
    pub fn invocation(&self) -> &Invocation {
        &self.invocation
    }

    /// The attributes this invocation gave its listing of kind `A`: for a
    /// result named by run id, its own listing; for a label, its one listing
    /// of kind `A`. `None` when it lists none (or, for a label, several), or
    /// when they do not read as `A::Attributes`.
    pub fn attributes<A: Artifact>(&self) -> Option<A::Attributes> {
        let listings = self.invocation.record.listings();
        let mut of_kind = listings.iter().filter(|l| l.kind == A::KIND);
        let chosen = match self.result {
            Some(run_id) => of_kind.find(|l| l.run_id == run_id),
            None => {
                let first = of_kind.next()?;
                of_kind.all(|l| l.run_id == first.run_id).then_some(first)
            }
        }?;
        serde_json::from_value(chosen.attributes.clone()).ok()
    }

    /// The attributes this invocation gave its listing of `id`, read as
    /// `A::Attributes`.
    pub fn attributes_of<A: Artifact>(&self, id: &Id<A>) -> Option<A::Attributes> {
        serde_json::from_value(self.invocation.attributes_of(A::KIND, id.key())?).ok()
    }
}

impl Listings {
    pub fn by(&self) -> By {
        self.by
    }

    /// Every listing, newest first.
    pub fn listed(&self) -> &[Listed] {
        &self.listed
    }

    /// The listing that answers: the label's invocation, or the newest.
    pub fn answering(&self) -> Option<&Listed> {
        self.listed.first()
    }

    /// The newest listing whose invocation has a `--label`.
    pub fn labelled(&self) -> Option<&Listed> {
        self.listed.iter().find(|l| l.label.is_some())
    }

    /// The oldest listing.
    pub fn first(&self) -> Option<&Listed> {
        self.listed.last()
    }

    /// The source the answering invocation recorded for its listing of `of`
    /// ([`crate::RunInfo::record_source`]), with the other sources listings
    /// recorded for it: further listings of `of` in the answering invocation
    /// itself, as an import writes one per source the copied records gave a
    /// result, and those of earlier invocations. [`Recorded::None`] when the
    /// answering invocation recorded none, even if an earlier one did: no
    /// other invocation stands in for it. A reference to a kind other than
    /// `B` is no source of `B`.
    pub fn source<A: Artifact, B: Artifact>(&self, of: &Id<A>) -> Recorded<B> {
        // Each distinct source an invocation recorded for `of`, in the order
        // of its listings.
        let recorded = |l: &Listed| -> Vec<Id<B>> {
            let mut out: Vec<Id<B>> = Vec::new();
            for r in l.invocation.record.listings().iter().filter(|r| r.kind == A::KIND && r.run_id == of.key()) {
                if let Some(s) = r.source.filter(|s| s.kind == B::KIND).map(|s| Id::from_key(s.run_id)) {
                    if !out.contains(&s) {
                        out.push(s);
                    }
                }
            }
            out
        };
        let Some(answering) = self.answering() else { return Recorded::None };
        let mine = recorded(answering);
        let Some((&latest, further)) = mine.split_first() else { return Recorded::None };
        let earlier = self.listed[1..].iter().flat_map(|l| recorded(l).into_iter().map(move |s| (l.seq, s)));
        let others: Vec<(u64, Id<B>)> =
            further.iter().map(|s| (answering.seq, *s)).chain(earlier).filter(|(_, s)| *s != latest).collect();
        if others.is_empty() {
            Recorded::One { source: latest, unfinished: answering.unfinished }
        } else {
            Recorded::Several { latest, others }
        }
    }
}

/// The source the answering invocation recorded for a listed result
/// ([`Listings::source`]).
pub enum Recorded<B: Artifact> {
    /// The answering invocation recorded none, or no invocation answers.
    None,
    /// One source; `unfinished` when the answering listing is in a running
    /// record.
    One { source: Id<B>, unfinished: bool },
    /// The answering invocation recorded `latest`, and the listings in
    /// `others`, by the sequence number of their invocation, recorded other
    /// sources. An entry with the answering invocation's own number is a
    /// further source that invocation recorded.
    Several { latest: Id<B>, others: Vec<(u64, Id<B>)> },
}

impl Store {
    /// Every listing of what `handle` names, one per invocation that listed
    /// it, newest first, by the rule of this module. A label that names no
    /// invocation is refused with [`Error::NotFound`]; a result no invocation
    /// listed has none.
    pub fn listings(&self, handle: &Handle) -> Result<Listings> {
        match handle {
            Handle::Label { name, back } => {
                let inv = label_holders(&self.inner, name)
                    .into_iter()
                    .nth(*back as usize)
                    .ok_or_else(|| Error::NotFound { handle: handle.to_string() })?;
                Ok(Listings { by: By::Label { seq: inv.seq() }, listed: vec![Listed::new(inv, false, None)] })
            }
            Handle::Path(_) | Handle::IdPrefix(_) => {
                let (key, _) = self.resolve_key(handle, "")?;
                self.listings_of(key)
            }
        }
    }

    /// Every listing of the result `run_id`, newest first.
    pub(crate) fn listings_of(&self, run_id: Key) -> Result<Listings> {
        let lists = |r: &Record| r.listings().iter().any(|l| l.run_id == run_id && l.completed());
        let snapshot = self.inner.invocations();
        let closed: Vec<Listed> = snapshot
            .iter()
            .rev()
            .filter(|i| !i.record.is_running() && lists(&i.record))
            .map(|i| Listed::new(i.clone(), false, Some(run_id)))
            .collect();
        if !closed.is_empty() {
            return Ok(Listings { by: By::Result { run_id }, listed: closed });
        }
        let in_running = |r: &Record| r.is_running() && r.listings().iter().any(|l| l.run_id == run_id);
        let mut running: Vec<Invocation> = snapshot.iter().filter(|i| in_running(&i.record)).cloned().collect();
        running.extend(
            self.inner
                .running()
                .into_iter()
                .map(|r| Invocation { record: Record::Running(r), took_label_from: None })
                .filter(|i| in_running(&i.record)),
        );
        let newest = running.into_iter().max_by_key(Invocation::seq);
        Ok(Listings { by: By::Result { run_id }, listed: newest.into_iter().map(|i| Listed::new(i, true, Some(run_id))).collect() })
    }
}

#[cfg(test)]
mod tests;

// By hand, so that `B`, a marker type, need not be `Debug` or `PartialEq`.
impl<B: Artifact> std::fmt::Debug for Recorded<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Recorded::None => f.write_str("None"),
            Recorded::One { source, unfinished } => {
                f.debug_struct("One").field("source", &source.key()).field("unfinished", unfinished).finish()
            }
            Recorded::Several { latest, others } => {
                let others: Vec<(u64, Key)> = others.iter().map(|(s, id)| (*s, id.key())).collect();
                f.debug_struct("Several").field("latest", &latest.key()).field("others", &others).finish()
            }
        }
    }
}

impl<B: Artifact> PartialEq for Recorded<B> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Recorded::None, Recorded::None) => true,
            (Recorded::One { source: a, unfinished: u }, Recorded::One { source: b, unfinished: v }) => a == b && u == v,
            (Recorded::Several { latest: a, others: x }, Recorded::Several { latest: b, others: y }) => a == b && x == y,
            _ => false,
        }
    }
}
