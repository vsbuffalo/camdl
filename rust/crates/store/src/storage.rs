//! The store: the only code that writes or reads stored folders.
//!
//! [`Store::begin`] reads what the store holds for a run id, classifies it,
//! and acts by the table in the run store proposal, "What `begin` does". A
//! quick (`Atomic`) result that another live process is computing is waited
//! for; a long (`InPlace`) one is refused with `Busy`; a completed one is
//! reused; a damaged one is moved to quarantine and computed again. A claim
//! is the staging folder `.staging/<run id>/` for an atomic result, created
//! whole by a rename, and the `.lock` in the final folder for an in-place run.

use std::any::Any;
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::artifact::{record_key, Artifact, Build, ByProduct, ContentFiles, Id, Kinds, Primary, RecordCheck, TopLevel, WriteMode};
use crate::check::damage;
use crate::error::{json_err, At, Error, Result, RunError};
use crate::fsutil::{
    claim_draft_name, create_dirs_durable, fsync_dir, is_lock_entry, lock_owner, quarantine, remove_all, try_lock, unique_name,
    write_file_synced, write_json_synced, LockAttempt, LockGuard,
};
use crate::hash::{ContentAddressed, Key, HASH_VERSION};
use crate::identity::Identity;
use crate::index::Index;
use crate::invocation::{all_invocations, config_digests, encode_attributes, Invocation, RunInfo, Shared, UnreadRecord};
use crate::layout::{dir_name, Base, Placed};
use crate::owner::{local_liveness, this_owner, Beat, BeatMemory, Liveness, Owner, BEAT_INTERVAL, STALE_AFTER};
use crate::record::{
    read_record, Failure, IdentityFile, IndexLine, Listing, Outcome, Parent, Pending, Read, Record, Relation, Role, RunFile, Running, Status,
    StoreFile, ATTEMPTS_DIR, ATTEMPT_RECORD, IDENTITY_FILE, LOCK_FILE, MAX_PRESERVED_ATTEMPTS, RECORD_FORMAT, RUN_FILE,
    STAGING_DIR, STORE_FORMAT,
};
use crate::run::{
    reproduce_dir, running_record, staging_dir, write_claim_records, ByProductSlot, Core, FileTable, Place, Run,
};
use crate::stored::{Entry, EntryState, Stored};
use crate::Provenance;

/// How long a claimant waits before looking again at a claim lock another
/// claimant holds: today's `LOCK_RETRY_INTERVAL`.
const CLAIM_LOCK_RETRY: Duration = Duration::from_millis(20);

/// The prefix of an opener's draft of `store.json`, written whole before it is
/// linked into place.
const STORE_JSON_DRAFT: &str = ".store.json.draft";

/// Whether a completed result may be reused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reuse {
    IfCompleted,
    /// Recompute: a completed result of the same id, and a run that stopped on
    /// another host, are moved to quarantine first. A live owner's claim is
    /// never taken.
    Force,
}

/// The answer to "may I compute this".
pub enum Begin<A: Artifact> {
    /// A completed result exists, with every by-product that was wanted.
    Reused(Stored<A>),
    /// An in-place run's live owner holds the claim, an owner on another host
    /// is not known to have stopped, or this process holds it.
    Busy(Owner),
    /// This process now owns the run id.
    Claimed(Run<A>),
}

/// The by-products wanted from one run of `A`, each with its identity and
/// attributes.
pub struct Wanted<A: Artifact> {
    pub(crate) items: Vec<WantedItem>,
    _of: PhantomData<fn() -> A>,
}

#[derive(Clone)]
pub(crate) struct WantedItem {
    kind: &'static str,
    identity: Arc<dyn Any + Send + Sync>,
    identity_key: Key,
    upstream_key: Key,
    record: serde_json::Value,
    attributes: serde_json::Value,
    label: String,
    held: Vec<Key>,
    deps: Vec<crate::identity::DepInfo>,
}

impl<A: Artifact> Clone for Wanted<A> {
    fn clone(&self) -> Self {
        Wanted { items: self.items.clone(), _of: PhantomData }
    }
}

impl<A: Artifact> Wanted<A> {
    pub fn none() -> Self {
        Wanted { items: Vec::new(), _of: PhantomData }
    }

    pub fn with<B: ByProduct<Of = A>>(mut self, identity: B::Identity, attributes: B::Attributes) -> Self
    where
        B::Identity: Send + Sync,
    {
        let label = B::label(&identity, &attributes);
        self.items.push(WantedItem {
            kind: B::KIND,
            identity_key: identity.key(),
            upstream_key: B::upstream(&identity).content_hash(),
            record: serde_json::to_value(identity.record()).expect("an identity record encodes"),
            attributes: serde_json::to_value(&attributes).expect("attributes encode"),
            label,
            held: identity.held_content(),
            deps: identity.deps(),
            identity: Arc::new(identity),
        });
        self
    }
}

/// A store root opened by one command. A clone is the same opening: it shares
/// the snapshot of the invocation records and this process's beat.
#[derive(Clone)]
pub struct Store {
    pub(crate) inner: Arc<Inner>,
}

pub(crate) struct Inner {
    pub(crate) root: PathBuf,
    pub(crate) ambient_key: Key,
    pub(crate) ambient: serde_json::Value,
    ambient_check: RecordCheck,
    pub(crate) known_build: bool,
    pub(crate) top: TopLevel,
    pub(crate) kinds: Kinds,
    beat: Mutex<Option<Arc<Beat>>>,
    beats: BeatMemory,
    pub(crate) index: Mutex<Option<Index>>,
    pub(crate) scratch: bool,
    /// How many times this store has read the directory of invocation
    /// records.
    pub(crate) record_reads: AtomicUsize,
    records: Mutex<Records>,
}

/// The invocation records a store's lookups answer from: the closed records
/// read from `.invocations/` once, at the first lookup, and the records this
/// store's own invocations have written since, in whatever state each was
/// last written. A record of another process that is still running is in
/// neither; [`Inner::running`] reads those from the directory.
#[derive(Default)]
struct Records {
    read: Option<Vec<Record>>,
    own: BTreeMap<u64, Record>,
    /// Every record a read of the directory found and did not read, by path.
    unread: BTreeMap<PathBuf, UnreadRecord>,
    /// Whether the unread records have been reported.
    reported: bool,
    /// `read` and `own` merged by sequence number, own winning; dropped when
    /// an own record is written.
    merged: Option<Arc<[Invocation]>>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        if self.scratch {
            self.beat.lock().unwrap_or_else(|e| e.into_inner()).take();
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }
}

/// What the store holds for one run id.
pub(crate) enum State {
    Absent,
    /// An atomic claim: a staging folder naming its owner.
    Claim { dir: PathBuf, owner: Owner },
    /// An in-place folder whose claim has written its `.lock` and no record.
    HalfMade { dir: PathBuf, owner: Option<Owner> },
    Running { dir: PathBuf, owner: Owner },
    Failed { dir: PathBuf, failure: Option<Failure> },
    Completed { dir: PathBuf, run: Box<RunFile> },
    Damaged { dir: PathBuf, what: String },
}

/// Where to look for a result's folder.
pub(crate) enum Where<'a> {
    Placed(&'a Placed),
    /// A by-product, under `<upstream folder>/<kind>/`.
    Under(&'a Path),
}

/// The invocation record entries one `begin` updates.
#[derive(Clone, Default)]
pub(crate) struct Entries {
    upstream: Vec<usize>,
    by_products: Vec<Vec<usize>>,
}

impl Store {
    /// Opens the store at `root`, creating it when the directory is absent or
    /// empty. Refuses with [`Error::NotAStore`] a root that is not empty and
    /// has no `store.json`, a `store.json` it cannot read, and a format this
    /// program does not know; and with [`Error::TopLevelDiffers`] a store laid
    /// out the other way.
    pub fn open<E: Identity>(root: &Path, ambient: Build<E>, top: TopLevel, kinds: Kinds) -> Result<Store> {
        Self::open_inner(root, ambient, top, kinds, false)
    }

    /// The same API over a temporary root, laid out by kind, deleted when the
    /// store is dropped.
    pub fn scratch<E: Identity>(ambient: Build<E>, kinds: Kinds) -> Result<Store> {
        let root = std::env::temp_dir().join(unique_name("store-scratch"));
        Self::open_inner(&root, ambient, TopLevel::ByKind, kinds, true)
    }

    fn open_inner<E: Identity>(root: &Path, ambient: Build<E>, top: TopLevel, kinds: Kinds, scratch: bool) -> Result<Store> {
        kinds.validate()?;
        let store_json = root.join("store.json");
        let not_a_store = |why: String| Error::NotAStore { root: root.to_path_buf(), why };
        let read_record = |bytes: &[u8]| -> Result<()> {
            let f: StoreFile = serde_json::from_slice(bytes).map_err(|e| not_a_store(format!("store.json: {e}")))?;
            if f.format != STORE_FORMAT {
                return Err(not_a_store(format!("store.json has format {}; this program reads format {STORE_FORMAT}", f.format)));
            }
            if f.top_level != top.wire() {
                return Err(Error::TopLevelDiffers { root: root.to_path_buf(), stored: f.top_level, asked: top.wire().into() });
            }
            Ok(())
        };
        // A root is a store once `store.json` exists. A root without one is
        // made a store when it holds nothing but other openers' drafts of
        // `store.json`: each opener writes a whole draft of its own and links
        // it to `store.json`, which succeeds for exactly one of them, and the
        // others read what it linked. A root holding anything else is refused
        // at once.
        loop {
            match std::fs::read(&store_json) {
                Ok(bytes) => {
                    read_record(&bytes)?;
                    break;
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(not_a_store(format!("store.json cannot be read: {e}"))),
            }
            std::fs::create_dir_all(root).at("create directory", root)?;
            let foreign = std::fs::read_dir(root)
                .at("list", root)?
                .flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .any(|n| n != "store.json" && !n.starts_with(STORE_JSON_DRAFT));
            if foreign {
                return Err(not_a_store("the directory is not empty and has no store.json".into()));
            }
            let draft = root.join(unique_name(STORE_JSON_DRAFT));
            let bytes = serde_json::to_vec_pretty(&StoreFile { format: STORE_FORMAT, top_level: top.wire().into() })
                .map_err(|e| json_err("encode", &store_json, e))?;
            write_file_synced(&draft, &bytes)?;
            let linked = std::fs::hard_link(&draft, &store_json);
            let _ = std::fs::remove_file(&draft);
            match linked {
                Ok(()) => {
                    fsync_dir(root)?;
                    break;
                }
                // Another opener linked its draft first: read it.
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e).at("create", &store_json),
            }
        }
        let (e, known) = ambient.parts();
        let ambient_json = serde_json::to_value(e.record())
            .map_err(|err| Error::Parse { what: "the ambient identity".into(), why: err.to_string() })?;
        Ok(Store {
            inner: Arc::new(Inner {
                root: root.to_path_buf(),
                ambient_key: e.key(),
                ambient: ambient_json,
                ambient_check: record_key::<E>,
                known_build: known,
                top,
                kinds,
                beat: Mutex::new(None),
                beats: BeatMemory::default(),
                index: Mutex::new(None),
                scratch,
                record_reads: AtomicUsize::new(0),
                records: Mutex::new(Records::default()),
            }),
        })
    }

    pub fn root(&self) -> &Path {
        &self.inner.root
    }

    /// The ambient key this store was opened with: the hash of the build and
    /// the environment switches that every run id it computes holds.
    pub fn ambient_key(&self) -> Key {
        self.inner.ambient_key
    }

    /// Begins `identity`: reuses a completed result, refuses a busy one, or
    /// claims the run id. Writes the attributes into the invocation record's entry
    /// for this result, whatever it returns.
    pub fn begin<A: Primary>(&self, identity: &A::Identity, attributes: &A::Attributes, info: &RunInfo, reuse: Reuse) -> Result<Begin<A>> {
        self.begin_with(identity, attributes, &Wanted::none(), info, reuse)
    }

    /// As [`Store::begin`], also wanting by-products of the same computation.
    pub fn begin_with<A: Primary>(
        &self,
        identity: &A::Identity,
        attributes: &A::Attributes,
        wanted: &Wanted<A>,
        info: &RunInfo,
        reuse: Reuse,
    ) -> Result<Begin<A>> {
        let id = Id::<A>::of(&self.inner.ambient_key, identity);
        let entries = self.register(info, id.key(), identity.key(), A::KIND, attributes, wanted, reuse)?;
        self.inner.begin_inner(identity, attributes, wanted, &info.shared, reuse, &entries)
    }

    /// Begins, then reuses, refuses a busy id with [`Error::InProgress`], or
    /// runs `compute` on a claim that expects `expected` and finishes, or
    /// fails with its error's text under `operation`.
    #[allow(clippy::too_many_arguments)] // the begin's four inputs, the claim's file set, and the computation with its name
    pub fn run<A: Primary, E: std::fmt::Display>(
        &self,
        identity: &A::Identity,
        attributes: &A::Attributes,
        info: &RunInfo,
        reuse: Reuse,
        expected: crate::run::ExpectedFiles,
        operation: &str,
        compute: impl FnOnce(&mut Run<A>) -> std::result::Result<(), E>,
    ) -> std::result::Result<Stored<A>, RunError<E>> {
        match self.begin::<A>(identity, attributes, info, reuse)? {
            Begin::Reused(s) => Ok(s),
            Begin::Busy(owner) => Err(RunError::Store(Error::InProgress(owner))),
            Begin::Claimed(mut run) => {
                run.expect(expected);
                match compute(&mut run) {
                    Ok(()) => Ok(run.finish()?),
                    Err(e) => {
                        run.fail(operation, &e.to_string())?;
                        Err(RunError::Compute(e))
                    }
                }
            }
        }
    }

    /// A completed, undamaged result.
    pub fn get<A: Artifact>(&self, id: &Id<A>) -> Result<Option<Stored<A>>> {
        match self.inner.find_completed(A::KIND, &id.key())? {
            Some((dir, _)) => Stored::open_at(&self.inner, id.key(), &dir).map(Some),
            None => Ok(None),
        }
    }

    /// A result in any status.
    pub fn inspect<A: Artifact>(&self, id: &Id<A>) -> Result<Option<Entry<A>>> {
        self.inner.entry(id.key())
    }

    /// Stores the content if it is not there yet, and returns it as a result
    /// to depend on. The attributes and the provenance are the first writer's; the
    /// attributes name the folder. Takes no claim.
    pub fn put<C: ContentFiles>(
        &self,
        content: C::Identity,
        attributes: &C::Attributes,
        provenance: &Provenance,
        info: &RunInfo,
    ) -> Result<Stored<C>> {
        let id = Id::<C>::of(&self.inner.ambient_key, &content);
        let entries = self.register(info, id.key(), content.key(), C::KIND, attributes, &Wanted::<C>::none(), Reuse::IfCompleted)?;
        let r = self.inner.put_inner::<C>(&content, attributes, provenance, &info.shared, id.key());
        match &r {
            Ok((_, computed)) => info.shared.set(&entries.upstream, if *computed { Outcome::Computed } else { Outcome::Reused }),
            Err(e) => info.shared.set(&entries.upstream, Outcome::Failed(Failure { operation: Some("storing".into()), reason: Some(e.to_string()) })),
        }
        r.map(|(s, _)| s)
    }

    /// Lists the stored result `id` in the invocation `info` with `attributes`,
    /// as reused, and stores nothing: for a command that reads a result
    /// stored by another invocation, perhaps by another build, and records
    /// attributes of its own for it. Refused with [`Error::NotFound`] when no
    /// completed result is `id`.
    pub fn list<A: Artifact>(&self, id: &Id<A>, attributes: &A::Attributes, info: &RunInfo) -> Result<()> {
        if self.inner.find_completed(A::KIND, &id.key())?.is_none() {
            return Err(Error::NotFound { handle: id.key().to_hex() });
        }
        let entries = self.register(info, id.key(), id.key(), A::KIND, attributes, &Wanted::<A>::none(), Reuse::IfCompleted)?;
        info.shared.set(&entries.upstream, Outcome::Reused);
        Ok(())
    }

    /// Read-only: the content artifact holding `content`, when it is stored.
    /// Nothing is written, so a command that only plans can name it.
    pub fn get_content<C: ContentFiles>(&self, content: &C::Identity) -> Result<Option<Stored<C>>> {
        self.get(&Id::<C>::of(&self.inner.ambient_key, content))
    }

    /// Read-only: what exists for each identity. Nothing is claimed.
    pub fn plan<A: Primary>(&self, members: &[(A::Identity, A::Attributes)]) -> Result<Vec<Planned<A>>> {
        members
            .iter()
            .map(|(identity, attributes)| {
                let id = Id::<A>::of(&self.inner.ambient_key, identity);
                let placed = A::layout(identity, attributes);
                let state = match self.inner.classify(A::KIND, id.key(), Where::Placed(&placed))? {
                    State::Absent => PlanState::Absent,
                    State::Claim { owner, .. } | State::Running { owner, .. } | State::HalfMade { owner: Some(owner), .. } => {
                        if owner.is_this_host() && local_liveness(&owner) == Liveness::Dead {
                            PlanState::Interrupted
                        } else {
                            PlanState::Running(owner)
                        }
                    }
                    State::HalfMade { owner: None, .. } | State::Damaged { .. } => PlanState::Damaged,
                    State::Failed { .. } => PlanState::Failed,
                    State::Completed { .. } => PlanState::Completed,
                };
                Ok(Planned { id, state })
            })
            .collect()
    }

    /// Read-only, as [`Store::plan`], with the by-products wanted of each
    /// member: a completed member lacking a completed wanted by-product is
    /// [`PlanState::ByProductsMissing`], since beginning it would compute it
    /// again to produce them.
    pub fn plan_with<A: Primary>(&self, members: &[(A::Identity, A::Attributes, Wanted<A>)]) -> Result<Vec<Planned<A>>> {
        let mut out = Vec::with_capacity(members.len());
        for (identity, attributes, wanted) in members {
            let mut planned = self.plan::<A>(std::slice::from_ref(&(identity.clone(), attributes.clone())))?.remove(0);
            if planned.state == PlanState::Completed {
                for w in &wanted.items {
                    let bp = crate::artifact::run_id(w.kind, &self.inner.ambient_key, &w.identity_key);
                    if self.inner.find_completed(w.kind, &bp)?.is_none() {
                        planned.state = PlanState::ByProductsMissing;
                        break;
                    }
                }
            }
            out.push(planned);
        }
        Ok(out)
    }

    /// Declares the members a command will compute, each with its attributes and
    /// the by-products wanted of it, in the order it asks for them: each is
    /// registered in the invocation record as planned, the running record is
    /// written with all of them, and none is claimed. A repeated identity is
    /// listed once per time it is asked for and claimed once.
    pub fn declare<A: Primary>(
        &self,
        members: &[(A::Identity, A::Attributes, Wanted<A>)],
        info: &RunInfo,
        reuse: Reuse,
    ) -> Result<Declared<A>> {
        let mut slots: Vec<Slot<A>> = Vec::new();
        let mut position = Vec::with_capacity(members.len());
        let mut by_id: BTreeMap<Key, usize> = BTreeMap::new();
        for (identity, attributes, wanted) in members {
            let id = Id::<A>::of(&self.inner.ambient_key, identity);
            let e = self.register(info, id.key(), identity.key(), A::KIND, attributes, wanted, reuse)?;
            match by_id.get(&id.key()) {
                Some(&s) => {
                    let all = &mut slots[s].entries;
                    all.upstream.extend(e.upstream);
                    for (a, b) in all.by_products.iter_mut().zip(e.by_products) {
                        a.extend(b);
                    }
                    position.push(s);
                }
                None => {
                    by_id.insert(id.key(), slots.len());
                    position.push(slots.len());
                    slots.push(Slot {
                        key: id.key(),
                        identity: identity.clone(),
                        attributes: attributes.clone(),
                        wanted: wanted.clone(),
                        entries: e,
                        begun: false,
                    });
                }
            }
        }
        info.shared.write_running()?;
        Ok(Declared { inner: Arc::clone(&self.inner), info: Arc::clone(&info.shared), reuse, slots, position })
    }

    /// Whether `owner` is alive, waiting up to `wait` for an owner on another
    /// host.
    pub fn liveness(&self, owner: &Owner, wait: Duration) -> Result<Liveness> {
        Ok(self.inner.liveness(owner, wait))
    }

    /// The runs a configuration file has produced, newest first, from the
    /// invocation records that read a file with this digest.
    pub fn runs_of_configuration(&self, digest: Key) -> Vec<Key> {
        let mut out = Vec::new();
        for f in self.inner.invocations().iter().rev().map(|i| &i.record) {
            if config_digests(f).any(|d| d == digest) {
                for r in f.listings() {
                    if !out.contains(&r.run_id) {
                        out.push(r.run_id);
                    }
                }
            }
        }
        out
    }

    /// Every closed invocation record, and this store's own records, by
    /// sequence number: a snapshot read from the directory once, at the
    /// store's first lookup, to which each record this store writes is
    /// added. A record another process writes after the snapshot is not in
    /// it, nor is another process's running record.
    pub fn invocations(&self) -> Arc<[Invocation]> {
        self.inner.invocations()
    }

    /// How many times this store, and its clones, have read `.invocations/`:
    /// for a test that checks a command reads the records once.
    #[doc(hidden)]
    pub fn directory_reads(&self) -> usize {
        self.inner.record_reads.load(Ordering::Relaxed)
    }

    /// Whether `self` and `other` are one opening of a store.
    pub fn same_opening(&self, other: &Store) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// The invocation records this store's reads of `.invocations/` found and
    /// did not read ([`UnreadRecord`]), by path.
    pub fn unread_records(&self) -> Vec<UnreadRecord> {
        self.inner.records.lock().unwrap_or_else(|e| e.into_inner()).unread.values().cloned().collect()
    }

    fn register<A: Artifact>(
        &self,
        info: &RunInfo,
        run_id: Key,
        identity_key: Key,
        kind: &'static str,
        attributes: &A::Attributes,
        wanted: &Wanted<A>,
        reuse: Reuse,
    ) -> Result<Entries> {
        let mut list = vec![Listing { kind: kind.into(), run_id, attributes: encode_attributes(attributes)?, source: None, outcome: Pending::Planned }];
        for w in &wanted.items {
            assert_eq!(
                w.upstream_key, identity_key,
                "a wanted {}'s upstream record is not the identity being begun",
                w.kind
            );
            let bp_id = crate::artifact::run_id(w.kind, &self.inner.ambient_key, &w.identity_key);
            list.push(Listing { kind: w.kind.into(), run_id: bp_id, attributes: w.attributes.clone(), source: None, outcome: Pending::Planned });
        }
        let idx = info.shared.register(&self.inner, list, reuse)?;
        Ok(Entries { upstream: vec![idx[0]], by_products: idx[1..].iter().map(|i| vec![*i]).collect() })
    }
}

/// What exists for one identity, before anything is claimed.
pub struct Planned<A: Artifact> {
    pub id: Id<A>,
    pub state: PlanState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanState {
    Absent,
    /// The result is completed and a by-product wanted of it is not.
    ByProductsMissing,
    Running(Owner),
    Completed,
    Damaged,
    Failed,
    Interrupted,
}

/// The members a command declared ([`Store::declare`]): listed in its
/// invocation record as planned, and claimed one at a time through
/// [`Declared::begin`], or all together through [`Declared::begin_all`]. A
/// member never begun is recorded as failed when the command ends.
pub struct Declared<A: Primary> {
    inner: Arc<Inner>,
    info: Arc<Shared>,
    reuse: Reuse,
    /// One per distinct identity, in the order first asked for.
    slots: Vec<Slot<A>>,
    /// The slot of each member as the command listed it.
    position: Vec<usize>,
}

struct Slot<A: Primary> {
    key: Key,
    identity: A::Identity,
    attributes: A::Attributes,
    wanted: Wanted<A>,
    /// Its rows in the invocation record.
    entries: Entries,
    begun: bool,
}

impl<A: Primary> Declared<A> {
    /// Claims member `i`, in the order the members were declared, and no
    /// other: reused, busy or claimed, as [`Store::begin`] answers. `None`
    /// when its identity was begun already, under this index or another
    /// that names the same identity.
    pub fn begin(&mut self, i: usize) -> Result<Option<Begin<A>>> {
        let s = self.position[i];
        let slot = &mut self.slots[s];
        if slot.begun {
            return Ok(None);
        }
        slot.begun = true;
        let slot = &self.slots[s];
        self.inner
            .begin_inner(&slot.identity, &slot.attributes, &slot.wanted, &self.info, self.reuse, &slot.entries)
            .map(Some)
    }

    /// Claims every member not yet begun, in run-id order, so two commands
    /// declaring overlapping members cannot deadlock: an atomic id a live
    /// owner holds is waited for; an in-place one comes back as `Busy`.
    /// Returns one `Begin` per distinct identity claimed here, in the order
    /// first asked for.
    pub fn begin_all(&mut self) -> Result<Vec<Begin<A>>> {
        let mut order: Vec<usize> = (0..self.slots.len()).filter(|&s| !self.slots[s].begun).collect();
        order.sort_by_key(|&s| self.slots[s].key);
        let mut begun: BTreeMap<usize, Begin<A>> = BTreeMap::new();
        for s in order {
            self.slots[s].begun = true;
            let slot = &self.slots[s];
            let b = self.inner.begin_inner(&slot.identity, &slot.attributes, &slot.wanted, &self.info, self.reuse, &slot.entries)?;
            begun.insert(s, b);
        }
        Ok(begun.into_values().collect())
    }

    /// Replaces the command's progress snapshot, beside its invocation record.
    pub fn progress(&self, snapshot: &impl serde::Serialize) -> Result<()> {
        let v = serde_json::to_value(snapshot).map_err(|e| Error::Parse { what: "a progress snapshot".into(), why: e.to_string() })?;
        self.info.progress(v)
    }
}

impl Inner {
    /// The snapshot [`Store::invocations`] returns.
    pub(crate) fn invocations(&self) -> Arc<[Invocation]> {
        let mut r = self.records.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(m) = &r.merged {
            return Arc::clone(m);
        }
        if r.read.is_none() {
            drop(r);
            let closed: Vec<Record> = self.read_records().into_iter().filter(|f| !f.is_running()).collect();
            r = self.records.lock().unwrap_or_else(|e| e.into_inner());
            r.read.get_or_insert(closed);
        }
        let mut by_seq: BTreeMap<u64, &Record> = r.read.iter().flatten().map(|f| (f.seq(), f)).collect();
        by_seq.extend(r.own.iter().map(|(seq, f)| (*seq, f)));
        let merged: Arc<[Invocation]> =
            by_seq.into_values().map(|f| Invocation { record: f.clone(), took_label_from: None }).collect();
        r.merged = Some(Arc::clone(&merged));
        merged
    }

    /// Adds `file`, a record this store has just written for one of its own
    /// invocations, to the snapshot.
    pub(crate) fn wrote_own(&self, file: &Record) {
        let mut r = self.records.lock().unwrap_or_else(|e| e.into_inner());
        r.own.insert(file.seq(), file.clone());
        r.merged = None;
    }

    /// Drops the snapshot, so the next lookup reads the directory: after
    /// `clean` closes other processes' records.
    pub(crate) fn forget_records(&self) {
        let mut r = self.records.lock().unwrap_or_else(|e| e.into_inner());
        r.read = None;
        r.merged = None;
    }

    /// The running records under the root, read from the directory now. A
    /// record is replaced by renaming a whole new version over it, so each is
    /// read whole without the store-wide lock, and the read writes nothing:
    /// a store with no `.invocations/` has no running record.
    pub(crate) fn running(&self) -> Vec<Running> {
        self.read_records()
            .into_iter()
            .filter_map(|f| match f {
                Record::Running(r) => Some(r),
                Record::Ended(_) | Record::Interrupted(_) => None,
            })
            .collect()
    }

    /// Every invocation record under the root, read from the directory now.
    /// The records it finds and does not read are kept, and reported once
    /// per store to the reporter [`crate::report_unread_records_to`] set.
    pub(crate) fn read_records(&self) -> Vec<Record> {
        self.record_reads.fetch_add(1, Ordering::Relaxed);
        let (records, unread) = all_invocations(&self.root);
        if !unread.is_empty() {
            let mut r = self.records.lock().unwrap_or_else(|e| e.into_inner());
            for u in unread {
                r.unread.entry(u.path.clone()).or_insert(u);
            }
            if !r.reported {
                r.reported = true;
                let all: Vec<UnreadRecord> = r.unread.values().cloned().collect();
                drop(r);
                crate::invocation::report_unread(&self.root, &all);
            }
        }
        records
    }

    /// This process's beat in this root, started at the first call.
    pub(crate) fn beat(&self) -> Result<Arc<Beat>> {
        let mut b = self.beat.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(beat) = b.as_ref() {
            return Ok(Arc::clone(beat));
        }
        let beat = Arc::new(Beat::start(&self.root)?);
        *b = Some(Arc::clone(&beat));
        Ok(beat)
    }

    pub(crate) fn liveness(&self, owner: &Owner, wait: Duration) -> Liveness {
        if owner == this_owner() {
            Liveness::Alive
        } else if owner.is_this_host() {
            local_liveness(owner)
        } else {
            self.beats.remote_liveness(&self.root, owner, wait)
        }
    }

    pub(crate) fn ambient_key_of(&self, ambient: &serde_json::Value) -> Option<Key> {
        (self.ambient_check)(ambient).ok()
    }

    pub(crate) fn info_set(&self, info: &Shared, idx: &[usize], outcome: Outcome) {
        info.set(idx, outcome);
    }

    // ── Paths ──────────────────────────────────────────────────────────────

    fn build_suffix(&self) -> String {
        self.ambient_key.short12()
    }

    /// The directories holding kind directories that a read looks in: the root
    /// under `ByKind`, and this build's directories under `ByBuild`.
    fn own_tops(&self) -> Vec<PathBuf> {
        match &self.top {
            TopLevel::ByKind => vec![self.root.clone()],
            TopLevel::ByBuild { .. } => children_with_suffix(std::slice::from_ref(&self.root), &self.build_suffix()),
        }
    }

    /// Every directory holding kind directories, for every build.
    pub(crate) fn all_tops(&self) -> Vec<PathBuf> {
        match &self.top {
            TopLevel::ByKind => vec![self.root.clone()],
            TopLevel::ByBuild { .. } => visible_dirs(&self.root),
        }
    }

    fn own_top_for_write(&self) -> Result<PathBuf> {
        match &self.top {
            TopLevel::ByKind => Ok(self.root.clone()),
            TopLevel::ByBuild { label } => {
                if let Some(d) = self.own_tops().into_iter().next() {
                    return Ok(d);
                }
                let d = self.root.join(dir_name(label, &self.build_suffix()));
                create_dirs_durable(&self.root, &d)?;
                Ok(d)
            }
        }
    }

    fn base_dirs(&self, kind: &str, base: &Base) -> Result<Vec<PathBuf>> {
        Ok(match base {
            Base::InKind => self.own_tops().into_iter().map(|t| t.join(kind)).collect(),
            Base::Inside { kind: pk, parent, subdir } => match self.find_any(pk, parent)? {
                Some(d) => vec![d.join(subdir)],
                None => Vec::new(),
            },
        })
    }

    /// The directory a new result's folder goes in, with every group
    /// directory it needs created.
    fn parent_for_write(&self, kind: &str, placed: &Placed) -> Result<PathBuf> {
        let mut dir = match &placed.layout.base {
            Base::InKind => self.own_top_for_write()?.join(kind),
            Base::Inside { kind: pk, parent, subdir } => {
                let Some(p) = self.find_any(pk, parent)? else {
                    return Err(Error::NotFound { handle: format!("the {pk} {} this result is placed inside", parent.short12()) });
                };
                p.join(subdir)
            }
        };
        for (label, key) in &placed.layout.groups {
            let suffix = key.short12();
            dir = match children_with_suffix(&[dir.clone()], &suffix).into_iter().next() {
                Some(d) => d,
                None => dir.join(dir_name(label, &suffix)),
            };
        }
        create_dirs_durable(&self.root, &dir)?;
        Ok(dir)
    }

    // ── Finding folders ────────────────────────────────────────────────────

    fn index_folder(&self, run_id: &Key) -> Option<PathBuf> {
        let mut idx = self.index.lock().unwrap_or_else(|e| e.into_inner());
        let idx = idx.get_or_insert_with(|| Index::load(&self.root));
        idx.folder(run_id).map(|rel| self.root.join(rel))
    }

    /// The folder of `run_id`, found by its layout: every group directory
    /// with the group's suffix, then every folder with the run id's. A folder
    /// with no `identity.json` yet is returned as half-made.
    fn locate(&self, kind: &str, run_id: &Key, wh: &Where<'_>) -> Result<Option<(PathBuf, bool)>> {
        if let Some(d) = self.index_folder(run_id) {
            if identity_of(&d).is_some_and(|f| &f.run_id == run_id && f.kind == kind) {
                return Ok(Some((d, true)));
            }
        }
        let frontier = match wh {
            Where::Placed(placed) => {
                let mut f = self.base_dirs(kind, &placed.layout.base)?;
                for (_, key) in &placed.layout.groups {
                    f = children_with_suffix(&f, &key.short12());
                }
                f
            }
            Where::Under(up) => vec![up.join(kind)],
        };
        let mut cands = children_with_suffix(&frontier, &run_id.short12());
        cands.extend(children_with_suffix(&frontier, &run_id.to_hex()));
        // A folder with this id's suffix whose identity file is present and
        // does not read is classified, as damaged, rather than passed over:
        // passing over it would leave its name taken and the claim unable
        // to proceed.
        let mut half = None;
        let mut unreadable = None;
        for c in cands {
            match identity_of(&c) {
                Some(f) if &f.run_id == run_id => return Ok(Some((c, true))),
                Some(_) => {}
                None if !c.join(IDENTITY_FILE).exists() => {
                    half.get_or_insert(c);
                }
                None => {
                    unreadable.get_or_insert(c);
                }
            }
        }
        Ok(unreadable.map(|u| (u, true)).or(half.map(|h| (h, false))))
    }

    /// The folder of `run_id` in any status, found from the id alone: the
    /// folder the index names when its `identity.json` confirms it, else a
    /// walk of every result tree. A walk does not write the index, so a
    /// read leaves the store as it was.
    pub(crate) fn find_any(&self, kind: &str, run_id: &Key) -> Result<Option<PathBuf>> {
        if let Some(d) = self.index_folder(run_id) {
            if identity_of(&d).is_some_and(|f| &f.run_id == run_id && f.kind == kind) {
                return Ok(Some(d));
            }
        }
        let found = crate::maintenance::walk(self);
        Ok(found.into_iter().find(|(_, f)| &f.run_id == run_id && f.kind == kind).map(|(d, _)| d))
    }

    /// The folder of `run_id` in any status and of any kind, with its identity
    /// file, found as [`Inner::find_any`] finds one.
    pub(crate) fn find_any_kind(&self, run_id: &Key) -> Result<Option<(PathBuf, IdentityFile)>> {
        if let Some(d) = self.index_folder(run_id) {
            if let Some(f) = identity_of(&d).filter(|f| &f.run_id == run_id) {
                return Ok(Some((d, f)));
            }
        }
        let found = crate::maintenance::walk(self);
        Ok(found.into_iter().find(|(_, f)| &f.run_id == run_id))
    }

    /// The folder and record of a completed, undamaged result. A folder in
    /// any other state, damaged included, is `None`, so a caller that then
    /// begins the result moves the damaged folder to `.quarantine/` and
    /// computes it again.
    pub(crate) fn find_completed(&self, kind: &str, run_id: &Key) -> Result<Option<(PathBuf, RunFile)>> {
        let Some(dir) = self.find_any(kind, run_id)? else { return Ok(None) };
        match folder_state(&dir, run_id, &self.kinds) {
            State::Completed { dir, run } => Ok(Some((dir, *run))),
            _ => Ok(None),
        }
    }

    /// What is damaged in the folder of `run_id`, when one is found and its
    /// records or files no longer match; `None` for a folder in any other
    /// state, or none.
    pub(crate) fn damage_of(&self, kind: &str, run_id: &Key) -> Result<Option<String>> {
        let Some(dir) = self.find_any(kind, run_id)? else { return Ok(None) };
        match folder_state(&dir, run_id, &self.kinds) {
            State::Damaged { what, .. } => Ok(Some(what)),
            _ => Ok(None),
        }
    }

    // ── Classification ─────────────────────────────────────────────────────

    pub(crate) fn classify(&self, kind: &str, run_id: Key, wh: Where<'_>) -> Result<State> {
        let staging = match &wh {
            Where::Placed(_) => staging_dir(&self.root, &run_id),
            Where::Under(up) => up.join(STAGING_DIR).join(run_id.to_hex()),
        };
        if staging.exists() {
            match read_record::<RunFile>(&staging.join(RUN_FILE)) {
                Read::Ok(r) => return Ok(State::Claim { dir: staging, owner: r.owner }),
                // A claim folder is renamed into place whole, records first;
                // one without a readable record is debris.
                _ => {
                    quarantine(&self.root, &staging)?;
                }
            }
        }
        match self.locate(kind, &run_id, &wh)? {
            None => Ok(State::Absent),
            Some((dir, false)) => Ok(State::HalfMade { owner: lock_owner(&dir.join(LOCK_FILE)).ok().flatten(), dir }),
            Some((dir, true)) => Ok(folder_state(&dir, &run_id, &self.kinds)),
        }
    }

    pub(crate) fn entry<A: Artifact>(self: &Arc<Self>, run_id: Key) -> Result<Option<Entry<A>>> {
        let staging = staging_dir(&self.root, &run_id);
        let (dir, state) = if staging.join(RUN_FILE).is_file() {
            let r: RunFile = read_record(&staging.join(RUN_FILE)).required(&staging)?;
            (staging, State::Claim { dir: PathBuf::new(), owner: r.owner })
        } else {
            match self.find_any(A::KIND, &run_id)? {
                None => return Ok(None),
                Some(d) => {
                    let s = folder_state(&d, &run_id, &self.kinds);
                    (d, s)
                }
            }
        };
        let ident: IdentityFile = read_record(&dir.join(IDENTITY_FILE)).required(&dir)?;
        let record = serde_json::from_value(ident.identity)
            .map_err(|e| Error::Parse { what: dir.join(IDENTITY_FILE).display().to_string(), why: e.to_string() })?;
        let state = match state {
            State::Claim { owner, .. } | State::Running { owner, .. } | State::HalfMade { owner: Some(owner), .. } => {
                match self.liveness(&owner, Duration::ZERO) {
                    Liveness::Dead => EntryState::Interrupted { owner },
                    l => EntryState::Running { owner, liveness: l },
                }
            }
            State::HalfMade { owner: None, .. } => EntryState::Damaged { what: "a folder with no record".into() },
            State::Completed { .. } => EntryState::Completed,
            State::Damaged { what, .. } => EntryState::Damaged { what },
            State::Failed { failure, .. } => {
                let f = failure.unwrap_or_default();
                EntryState::Failed { operation: f.operation, reason: f.reason }
            }
            State::Absent => return Ok(None),
        };
        Ok(Some(Entry { id: Id::from_key(run_id), record, state, dir, store: Arc::clone(self) }))
    }

    // ── Begin ──────────────────────────────────────────────────────────────

    pub(crate) fn begin_inner<A: Primary>(
        self: &Arc<Self>,
        identity: &A::Identity,
        attributes: &A::Attributes,
        wanted: &Wanted<A>,
        info: &Arc<Shared>,
        reuse: Reuse,
        entries: &Entries,
    ) -> Result<Begin<A>> {
        let id = Id::<A>::of(&self.ambient_key, identity);
        let placed = A::layout(identity, attributes);
        let force = reuse == Reuse::Force || !self.known_build;
        let busy = |owner: Owner| {
            info.set(&entries.upstream, Outcome::Busy);
            for e in &entries.by_products {
                info.set(e, Outcome::Busy);
            }
            Ok(Begin::Busy(owner))
        };
        let mut forced = false;
        loop {
            let state = self.classify(A::KIND, id.key(), Where::Placed(&placed))?;
            match state {
                State::Absent => {
                    if let Some(run) = self.claim_fresh::<A>(identity, &placed, wanted, info, entries)? {
                        return Ok(Begin::Claimed(run));
                    }
                }
                State::Claim { dir, owner } | State::Running { dir, owner } | State::HalfMade { dir, owner: Some(owner) } => {
                    if &owner == this_owner() {
                        return busy(owner);
                    }
                    match self.liveness(&owner, STALE_AFTER) {
                        Liveness::Dead => {
                            if dir.starts_with(self.root.join(STAGING_DIR)) {
                                self.clear_claim(&dir, &owner, false)?;
                            } else if let Some(run) = self.reclaim::<A>(identity, &dir, wanted, info, entries)? {
                                return Ok(Begin::Claimed(run));
                            }
                        }
                        Liveness::Stale { .. } | Liveness::Unknown if !owner.is_this_host() => {
                            if !force {
                                return busy(owner);
                            }
                            if dir.starts_with(self.root.join(STAGING_DIR)) {
                                self.clear_claim(&dir, &owner, true)?;
                            } else {
                                quarantine(&self.root, &dir)?;
                            }
                        }
                        _ => {
                            if A::WRITE == WriteMode::InPlace {
                                return busy(owner);
                            }
                            self.wait_for(&dir, &owner);
                        }
                    }
                }
                State::HalfMade { dir, owner: None } => {
                    if let Some(run) = self.claim_half_made::<A>(identity, &placed, &dir, wanted, info, entries)? {
                        return Ok(Begin::Claimed(run));
                    }
                }
                State::Damaged { dir, .. } => {
                    self.quarantine_damaged(&dir, &id.key(), A::WRITE)?;
                }
                State::Failed { dir, .. } => {
                    if let Some(run) = self.reclaim::<A>(identity, &dir, wanted, info, entries)? {
                        return Ok(Begin::Claimed(run));
                    }
                }
                State::Completed { dir, run } => {
                    if force && !forced {
                        quarantine(&self.root, &dir)?;
                        forced = true;
                        continue;
                    }
                    match self.begin_by_products::<A>(identity, &dir, run, wanted, info, entries, force)? {
                        ByProducts::Done(b) => return Ok(b),
                        ByProducts::Again => {}
                    }
                }
            }
        }
    }

    /// Waits until the claim at `dir` ends or its owner is no longer alive,
    /// reading both once per beat interval.
    fn wait_for(&self, dir: &Path, owner: &Owner) {
        loop {
            std::thread::sleep(BEAT_INTERVAL);
            if !dir.exists() {
                return;
            }
            match self.liveness(owner, STALE_AFTER) {
                Liveness::Alive => {}
                Liveness::Unknown if owner.is_this_host() => {}
                _ => return,
            }
        }
    }

    fn identity_file<A: Artifact>(&self, identity: &A::Identity, parent: Option<Parent>) -> Result<IdentityFile> {
        Ok(IdentityFile {
            format: RECORD_FORMAT,
            kind: A::KIND.into(),
            run_id: Id::<A>::of(&self.ambient_key, identity).key(),
            hash_version: HASH_VERSION,
            ambient: self.ambient.clone(),
            identity: serde_json::to_value(identity.record())
                .map_err(|e| Error::Parse { what: format!("an identity of kind `{}`", A::KIND), why: e.to_string() })?,
            parent,
        })
    }

    fn parent_of(placed: &Placed) -> Option<Parent> {
        match &placed.layout.base {
            Base::InKind => None,
            Base::Inside { kind, parent, .. } => Some(Parent { relation: Relation::Inside, kind: kind.to_string(), run_id: *parent }),
        }
    }

    fn reserved_names(&self) -> Vec<&'static str> {
        self.kinds.names().collect()
    }

    fn core(
        self: &Arc<Self>,
        kind: &'static str,
        run_id: Key,
        place: Place,
        table_dir: PathBuf,
        run_file: RunFile,
        held: Vec<Key>,
        info: &Arc<Shared>,
        entries: &Entries,
        by_products: Vec<ByProductSlot>,
    ) -> Core {
        Core {
            store: Arc::clone(self),
            kind,
            run_id,
            place,
            table: Arc::new(FileTable::new(table_dir, self.reserved_names())),
            run_file,
            held,
            info: Arc::clone(info),
            entries: entries.upstream.clone(),
            by_products,
            dropped: Vec::new(),
            events: Default::default(),
            settled: false,
        }
    }

    fn by_product_slot(&self, w: &WantedItem, dir: PathBuf, upstream: Key, upstream_kind: &str, info: &Shared, entries: Vec<usize>) -> Result<ByProductSlot> {
        let run_id = crate::artifact::run_id(w.kind, &self.ambient_key, &w.identity_key);
        let identity_file = IdentityFile {
            format: RECORD_FORMAT,
            kind: w.kind.into(),
            run_id,
            hash_version: HASH_VERSION,
            ambient: self.ambient.clone(),
            identity: w.record.clone(),
            parent: Some(Parent { relation: Relation::Upstream, kind: upstream_kind.into(), run_id: upstream }),
        };
        let run_file = running_record(info, info.seq(), w.deps.clone());
        std::fs::create_dir_all(&dir).at("create directory", &dir)?;
        write_claim_records(&dir, &identity_file, &run_file)?;
        Ok(ByProductSlot {
            kind: w.kind,
            run_id,
            label: w.label.clone(),
            table: Arc::new(FileTable::new(dir.clone(), self.reserved_names())),
            dir,
            entries,
            identity: Arc::clone(&w.identity),
            run_file,
            held: w.held.clone(),
            taken: false,
        })
    }

    /// Claims an absent run id. `None` when another process won the claim,
    /// so the caller classifies again.
    fn claim_fresh<A: Primary>(
        self: &Arc<Self>,
        identity: &A::Identity,
        placed: &Placed,
        wanted: &Wanted<A>,
        info: &Arc<Shared>,
        entries: &Entries,
    ) -> Result<Option<Run<A>>> {
        let run_id = Id::<A>::of(&self.ambient_key, identity).key();
        let ident = self.identity_file::<A>(identity, Self::parent_of(placed))?;
        let run_file = running_record(info, info.seq(), identity.deps());
        match A::WRITE {
            WriteMode::Atomic => {
                let staging_root = self.root.join(STAGING_DIR);
                std::fs::create_dir_all(&staging_root).at("create directory", &staging_root)?;
                let tmp = staging_root.join(claim_draft_name(this_owner(), &run_id.to_hex()));
                std::fs::create_dir(&tmp).at("create directory", &tmp)?;
                write_claim_records(&tmp, &ident, &run_file)?;
                let mut rel_dirs = Vec::new();
                for w in &wanted.items {
                    let kind_dir = tmp.join(w.kind);
                    let bp_id = crate::artifact::run_id(w.kind, &self.ambient_key, &w.identity_key);
                    let mut d = kind_dir.join(dir_name(&w.label, &bp_id.short12()));
                    if d.exists() {
                        d = kind_dir.join(dir_name(&w.label, &bp_id.to_hex()));
                    }
                    rel_dirs.push(d.strip_prefix(&tmp).expect("inside").to_path_buf());
                    std::fs::create_dir_all(&d).at("create directory", &d)?;
                }
                fsync_dir(&tmp)?;
                let staging = staging_dir(&self.root, &run_id);
                let placed_claim = self.under_claim_lock(&staging_root, &run_id.to_hex(), || {
                    Ok(!staging.exists() && std::fs::rename(&tmp, &staging).is_ok())
                })?;
                if placed_claim != Some(true) {
                    remove_all(&tmp)?;
                    return Ok(None);
                }
                fsync_dir(&staging_root)?;
                // A completed result that arrived meanwhile is reused.
                if let Some((_, true)) = self.locate(A::KIND, &run_id, &Where::Placed(placed))? {
                    remove_all(&staging)?;
                    return Ok(None);
                }
                let mut slots = Vec::new();
                for ((w, rel), e) in wanted.items.iter().zip(rel_dirs).zip(&entries.by_products) {
                    slots.push(self.by_product_slot(w, staging.join(rel), run_id, A::KIND, info, e.clone())?);
                }
                let core = self.core(
                    A::KIND,
                    run_id,
                    Place::Staging { dir: staging.clone(), placed: placed.clone() },
                    staging,
                    run_file,
                    identity.held_content(),
                    info,
                    entries,
                    slots,
                );
                Ok(Some(Run { core, identity: identity.clone(), _kind: PhantomData }))
            }
            WriteMode::InPlace => {
                let parent = self.parent_for_write(A::KIND, placed)?;
                let mut dir = parent.join(dir_name(&placed.label, &run_id.short12()));
                if let Err(e) = std::fs::create_dir(&dir) {
                    // `clean` removed an empty group directory a moment before.
                    if e.kind() == std::io::ErrorKind::NotFound {
                        return Ok(None);
                    }
                    if e.kind() != std::io::ErrorKind::AlreadyExists {
                        return Err(e).at("create directory", &dir);
                    }
                    match identity_of(&dir) {
                        Some(f) if f.run_id != run_id => {
                            dir = parent.join(dir_name(&placed.label, &run_id.to_hex()));
                            if std::fs::create_dir(&dir).is_err() {
                                return Ok(None);
                            }
                        }
                        _ => return Ok(None),
                    }
                }
                let live = |o: &Owner| self.liveness(o, STALE_AFTER);
                let lock = match try_lock(&dir.join(LOCK_FILE), &live) {
                    Ok(LockAttempt::Acquired(g)) => g,
                    Ok(LockAttempt::Held(..)) => return Ok(None),
                    // `clean` removed the folder, empty, before its lock existed.
                    Err(_) if !dir.exists() => return Ok(None),
                    Err(e) => return Err(e),
                };
                write_claim_records(&dir, &ident, &run_file)?;
                fsync_dir(&dir)?;
                fsync_dir(&parent)?;
                self.in_place_run(identity, dir, lock, ident, run_file, wanted, info, entries).map(Some)
            }
        }
    }

    fn in_place_run<A: Primary>(
        self: &Arc<Self>,
        identity: &A::Identity,
        dir: PathBuf,
        lock: LockGuard,
        ident: IdentityFile,
        run_file: RunFile,
        wanted: &Wanted<A>,
        info: &Arc<Shared>,
        entries: &Entries,
    ) -> Result<Run<A>> {
        let run_id = ident.run_id;
        let mut slots = Vec::new();
        for (w, e) in wanted.items.iter().zip(&entries.by_products) {
            let bp_id = crate::artifact::run_id(w.kind, &self.ambient_key, &w.identity_key);
            let d = dir.join(STAGING_DIR).join(bp_id.to_hex());
            slots.push(self.by_product_slot(w, d, run_id, A::KIND, info, e.clone())?);
        }
        let core = self.core(
            A::KIND,
            run_id,
            Place::Final { dir: dir.clone(), lock: Some(lock) },
            dir,
            run_file,
            identity.held_content(),
            info,
            entries,
            slots,
        );
        Ok(Run { core, identity: identity.clone(), _kind: PhantomData })
    }

    /// Runs `f` under `<dir>/<name>.lock`. Every creation of the claim folder
    /// `<dir>/<name>`, and every removal of one, takes this lock, so a removal
    /// decided from an earlier read is checked again, and cannot remove a claim
    /// another process made after that read. `None` when another claimant holds
    /// the lock; the caller classifies again.
    fn under_claim_lock<T>(&self, dir: &Path, name: &str, f: impl FnOnce() -> Result<T>) -> Result<Option<T>> {
        std::fs::create_dir_all(dir).at("create directory", dir)?;
        let live = |o: &Owner| self.liveness(o, STALE_AFTER);
        match try_lock(&dir.join(format!("{name}.lock")), &live)? {
            LockAttempt::Acquired(_guard) => f().map(Some),
            LockAttempt::Held(..) => {
                std::thread::sleep(CLAIM_LOCK_RETRY);
                Ok(None)
            }
        }
    }

    /// Removes the claim folder `dir`, or moves it to quarantine, if its
    /// record still names `owner`.
    pub(crate) fn clear_claim(&self, dir: &Path, owner: &Owner, to_quarantine: bool) -> Result<bool> {
        let (Some(parent), Some(name)) = (dir.parent(), dir.file_name()) else { return Ok(false) };
        let cleared = self.under_claim_lock(parent, &name.to_string_lossy(), || match read_record::<RunFile>(&dir.join(RUN_FILE)) {
            Read::Ok(r) if &r.owner == owner => {
                if to_quarantine {
                    quarantine(&self.root, dir)?;
                } else {
                    remove_all(dir)?;
                }
                Ok(true)
            }
            _ => Ok(false),
        })?;
        Ok(cleared == Some(true))
    }

    /// Moves a damaged folder to quarantine. An in-place folder is moved only
    /// under its `.lock`, after the damage is seen again there, so a claimant
    /// that has just reclaimed it keeps it.
    fn quarantine_damaged(&self, dir: &Path, run_id: &Key, write: WriteMode) -> Result<()> {
        if write == WriteMode::Atomic {
            return quarantine(&self.root, dir).map(|_| ());
        }
        let live = |o: &Owner| self.liveness(o, STALE_AFTER);
        let lock = match try_lock(&dir.join(LOCK_FILE), &live) {
            Ok(LockAttempt::Acquired(g)) => g,
            Ok(LockAttempt::Held(..)) => {
                std::thread::sleep(CLAIM_LOCK_RETRY);
                return Ok(());
            }
            Err(_) if !dir.exists() => return Ok(()),
            Err(e) => return Err(e),
        };
        if let State::Damaged { .. } = folder_state(dir, run_id, &self.kinds) {
            quarantine(&self.root, dir)?;
            lock.forget();
        }
        Ok(())
    }

    /// A folder with no record and no lock: an in-place claim between creating
    /// the folder and writing its `.lock`, or debris. Decided under the
    /// folder's lock, never from outside it: a folder that is empty beside the
    /// lock is claimed, one holding anything else is moved to quarantine, and
    /// one whose lock another claimant took first is looked at again. `None`
    /// when the caller should classify again.
    fn claim_half_made<A: Primary>(
        self: &Arc<Self>,
        identity: &A::Identity,
        placed: &Placed,
        dir: &Path,
        wanted: &Wanted<A>,
        info: &Arc<Shared>,
        entries: &Entries,
    ) -> Result<Option<Run<A>>> {
        let live = |o: &Owner| self.liveness(o, STALE_AFTER);
        let lock = match try_lock(&dir.join(LOCK_FILE), &live) {
            Ok(LockAttempt::Acquired(g)) => g,
            Ok(LockAttempt::Held(..)) => return Ok(None),
            // The folder went away: classify again.
            Err(_) if !dir.exists() => return Ok(None),
            Err(e) => return Err(e),
        };
        let debris = std::fs::read_dir(dir)
            .at("list", dir)?
            .flatten()
            .any(|e| !is_lock_entry(&e.file_name().to_string_lossy()));
        if dir.join(IDENTITY_FILE).exists() {
            return Ok(None);
        }
        if debris || A::WRITE != WriteMode::InPlace {
            quarantine(&self.root, dir)?;
            lock.forget();
            return Ok(None);
        }
        let ident = self.identity_file::<A>(identity, Self::parent_of(placed))?;
        let run_file = running_record(info, info.seq(), identity.deps());
        write_claim_records(dir, &ident, &run_file)?;
        fsync_dir(dir)?;
        self.in_place_run(identity, dir.to_path_buf(), lock, ident, run_file, wanted, info, entries).map(Some)
    }

    /// Reclaims an in-place folder whose run failed or whose owner is dead on
    /// this host: its root-level files move to `.attempts/<n>/`, `run.json`
    /// as `attempt.json`, its subdirectories are cleared, and the last three
    /// attempts are kept. `None` when another process holds the folder, or a
    /// completed result is found under the lock.
    fn reclaim<A: Primary>(
        self: &Arc<Self>,
        identity: &A::Identity,
        dir: &Path,
        wanted: &Wanted<A>,
        info: &Arc<Shared>,
        entries: &Entries,
    ) -> Result<Option<Run<A>>> {
        let live = |o: &Owner| self.liveness(o, STALE_AFTER);
        let lock = match try_lock(&dir.join(LOCK_FILE), &live)? {
            LockAttempt::Acquired(g) => g,
            LockAttempt::Held(..) => return Ok(None),
        };
        let run_id = Id::<A>::of(&self.ambient_key, identity).key();
        if let State::Completed { .. } = folder_state(dir, &run_id, &self.kinds) {
            return Ok(None);
        }
        preserve_attempt(dir)?;
        let ident = match read_record::<IdentityFile>(&dir.join(IDENTITY_FILE)) {
            Read::Ok(f) => f,
            _ => self.identity_file::<A>(identity, None)?,
        };
        let run_file = running_record(info, info.seq(), identity.deps());
        write_claim_records(dir, &ident, &run_file)?;
        fsync_dir(dir)?;
        self.in_place_run(identity, dir.to_path_buf(), lock, ident, run_file, wanted, info, entries).map(Some)
    }

    // ── By-products of a stored result ─────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    fn begin_by_products<A: Primary>(
        self: &Arc<Self>,
        identity: &A::Identity,
        dir: &Path,
        run: Box<RunFile>,
        wanted: &Wanted<A>,
        info: &Arc<Shared>,
        entries: &Entries,
        force: bool,
    ) -> Result<ByProducts<A>> {
        let run_id = Id::<A>::of(&self.ambient_key, identity).key();
        let mut missing = Vec::new();
        for (i, w) in wanted.items.iter().enumerate() {
            let bp_id = crate::artifact::run_id(w.kind, &self.ambient_key, &w.identity_key);
            match self.classify(w.kind, bp_id, Where::Under(dir))? {
                State::Completed { .. } => {}
                State::Absent => missing.push(i),
                State::Damaged { dir: d, .. } | State::HalfMade { dir: d, owner: None } | State::Failed { dir: d, .. } => {
                    quarantine(&self.root, &d)?;
                    missing.push(i);
                }
                State::Claim { dir: d, owner } | State::Running { dir: d, owner } | State::HalfMade { dir: d, owner: Some(owner) } => {
                    if &owner == this_owner() {
                        return self.all_busy(owner, info, entries);
                    }
                    match self.liveness(&owner, STALE_AFTER) {
                        Liveness::Dead => {
                            self.clear_claim(&d, &owner, false)?;
                            return Ok(ByProducts::Again);
                        }
                        Liveness::Stale { .. } | Liveness::Unknown if !owner.is_this_host() => {
                            if !force {
                                return self.all_busy(owner, info, entries);
                            }
                            self.clear_claim(&d, &owner, true)?;
                            return Ok(ByProducts::Again);
                        }
                        _ => {
                            self.wait_for(&d, &owner);
                            return Ok(ByProducts::Again);
                        }
                    }
                }
            }
        }
        if missing.is_empty() {
            info.set(&entries.upstream, Outcome::Reused);
            for e in &entries.by_products {
                info.set(e, Outcome::Reused);
            }
            return Ok(ByProducts::Done(Begin::Reused(Stored::open_at(self, run_id, dir)?)));
        }
        for (i, e) in entries.by_products.iter().enumerate() {
            if !missing.contains(&i) {
                info.set(e, Outcome::Reused);
            }
        }
        // Claim each missing by-product: a staging folder created whole by a rename.
        let staging_root = dir.join(STAGING_DIR);
        std::fs::create_dir_all(&staging_root).at("create directory", &staging_root)?;
        let mut slots = Vec::new();
        for &i in &missing {
            let w = &wanted.items[i];
            let bp_id = crate::artifact::run_id(w.kind, &self.ambient_key, &w.identity_key);
            let tmp = staging_root.join(claim_draft_name(this_owner(), &bp_id.to_hex()));
            let slot = self.by_product_slot(w, tmp.clone(), run_id, A::KIND, info, entries.by_products[i].clone())?;
            let dest = staging_root.join(bp_id.to_hex());
            let claimed = self.under_claim_lock(&staging_root, &bp_id.to_hex(), || {
                Ok(!dest.exists() && std::fs::rename(&tmp, &dest).is_ok())
            })?;
            if claimed != Some(true) {
                remove_all(&tmp)?;
                for s in &slots {
                    let s: &ByProductSlot = s;
                    remove_all(&s.dir)?;
                }
                return Ok(ByProducts::Again);
            }
            slots.push(ByProductSlot { table: Arc::new(FileTable::new(dest.clone(), self.reserved_names())), dir: dest, ..slot });
        }
        let files_dir = reproduce_dir(&slots[0].dir);
        std::fs::create_dir_all(&files_dir).at("create directory", &files_dir)?;
        // The reproducing run records the reads it makes itself, not the
        // stored run's, which its by-products would otherwise list twice.
        let mut record = (*run).clone();
        record.reads.clear();
        let core = self.core(
            A::KIND,
            run_id,
            Place::Reproducing { stored_dir: dir.to_path_buf(), stored: run.clone(), files_dir: files_dir.clone() },
            files_dir,
            record,
            identity.held_content(),
            info,
            entries,
            slots,
        );
        Ok(ByProducts::Done(Begin::Claimed(Run { core, identity: identity.clone(), _kind: PhantomData })))
    }

    fn all_busy<A: Artifact>(&self, owner: Owner, info: &Shared, entries: &Entries) -> Result<ByProducts<A>> {
        info.set(&entries.upstream, Outcome::Busy);
        for e in &entries.by_products {
            info.set(e, Outcome::Busy);
        }
        Ok(ByProducts::Done(Begin::Busy(owner)))
    }

    // ── Commit ─────────────────────────────────────────────────────────────

    /// Renames an atomic result's staging folder into its final folder,
    /// creating the group directories it needs, and returns the folder.
    /// `true` with the folder when this call moved it into place, `false`
    /// with the folder of a result of the same id that was already there, in
    /// which case `staging` is left for the caller to compare and remove.
    pub(crate) fn place_staged(&self, kind: &str, run_id: Key, placed: &Placed, staging: &Path) -> Result<(PathBuf, bool)> {
        for _ in 0..8 {
            let parent = self.parent_for_write(kind, placed)?;
            let short = parent.join(dir_name(&placed.label, &run_id.short12()));
            let full = parent.join(dir_name(&placed.label, &run_id.to_hex()));
            let mut dest = short.clone();
            for cand in [&short, &full] {
                if cand.exists() {
                    match identity_of(cand) {
                        Some(f) if f.run_id == run_id => return Ok((cand.clone(), false)),
                        _ => dest = full.clone(),
                    }
                }
            }
            match std::fs::rename(staging, &dest) {
                Ok(()) => {
                    fsync_dir(&parent)?;
                    return Ok((dest, true));
                }
                // A directory of the chain removed by `clean` a moment before
                // is made again; a destination another writer took is looked
                // at again.
                Err(_) if !parent.exists() || dest.exists() => continue,
                Err(e) => return Err(e).at("move the result into place", &dest),
            }
        }
        Err(Error::Io { op: "place", path: staging.to_path_buf(), source: std::io::Error::other("the destination kept changing") })
    }

    pub(crate) fn index_completed(&self, kind: &str, run_id: Key, dir: &Path, run: &RunFile, upstream: Option<Key>) -> Result<()> {
        let folder = crate::fsutil::rel_key(&self.root, dir);
        let line = IndexLine::Completed {
            format: RECORD_FORMAT,
            kind: kind.into(),
            run_id,
            folder,
            invocation: run.invocation,
            deps: run.deps.iter().map(|d| d.run_id).collect(),
            lineage: run
                .lineage
                .iter()
                .filter_map(|l| match l {
                    crate::record::Lineage::Result { run_id, .. } => Some(*run_id),
                    crate::record::Lineage::Input { .. } => None,
                })
                .collect(),
            upstream,
        };
        self.append_index(&line)
    }

    pub(crate) fn append_index(&self, line: &IndexLine) -> Result<()> {
        if self.scratch {
            return Ok(());
        }
        let mut idx = self.index.lock().unwrap_or_else(|e| e.into_inner());
        let idx = idx.get_or_insert_with(|| Index::load(&self.root));
        idx.append(&self.root, line)
    }

    // ── Content artifacts ──────────────────────────────────────────────────

    /// Writes a content artifact if it is not stored, and returns it with
    /// whether this call wrote it.
    fn put_inner<C: ContentFiles>(
        self: &Arc<Self>,
        content: &C::Identity,
        attributes: &C::Attributes,
        provenance: &Provenance,
        info: &Arc<Shared>,
        run_id: Key,
    ) -> Result<(Stored<C>, bool)> {
        let placed = C::layout(content, attributes);
        self.beat()?;
        loop {
            match self.locate(C::KIND, &run_id, &Where::Placed(&placed))? {
                Some((dir, true)) => match folder_state(&dir, &run_id, &self.kinds) {
                    State::Completed { .. } => return Ok((Stored::open_at(self, run_id, &dir)?, false)),
                    _ => {
                        quarantine(&self.root, &dir)?;
                        continue;
                    }
                },
                Some((dir, false)) => {
                    quarantine(&self.root, &dir)?;
                    continue;
                }
                None => {}
            }
            let staging_root = self.root.join(STAGING_DIR);
            std::fs::create_dir_all(&staging_root).at("create directory", &staging_root)?;
            let tmp = staging_root.join(claim_draft_name(this_owner(), &run_id.to_hex()));
            std::fs::create_dir(&tmp).at("create directory", &tmp)?;
            let ident = self.identity_file::<C>(content, Self::parent_of(&placed))?;
            let mut run = running_record(info, info.seq(), content.deps());
            write_json_synced(&tmp.join(IDENTITY_FILE), &ident)?;
            let table = FileTable::new(tmp.clone(), self.reserved_names());
            let held = content.held_content();
            let results = C::files(content);
            if let Some((name, _)) = results.iter().find(|(_, bytes)| !held.contains(&Key::digest_bytes(bytes))) {
                remove_all(&tmp)?;
                return Err(Error::ContentUnheld { file: name.clone() });
            }
            table.expect(crate::run::ExpectedFiles {
                result: results.iter().map(|(name, _)| name.clone()).collect(),
                provenance: provenance.files.keys().cloned().collect(),
            });
            for (name, bytes) in &results {
                table.open(name, Role::Result)?.write_all(bytes)?;
            }
            for (name, bytes) in &provenance.files {
                table.open(name, Role::Provenance)?.write_all(bytes)?;
            }
            let files = table.close_all()?;
            table.check_expected(run_id, &files)?;
            Core::check_held(&held, &files)?;
            run.status = Status::Completed;
            run.finished = Some(crate::owner::Timestamp::now());
            run.files = files;
            run.recorded = provenance.values.clone();
            write_json_synced(&tmp.join(RUN_FILE), &run)?;
            fsync_dir(&tmp)?;
            let (dest, moved) = self.place_staged(C::KIND, run_id, &placed, &tmp)?;
            if !moved {
                // Content is keyed by its own bytes: the stored copy is this one.
                remove_all(&tmp)?;
            }
            if moved {
                self.index_completed(C::KIND, run_id, &dest, &run, None)?;
            } else if !matches!(folder_state(&dest, &run_id, &self.kinds), State::Completed { .. }) {
                continue;
            }
            return Ok((Stored::open_at(self, run_id, &dest)?, moved));
        }
    }
}

enum ByProducts<A: Artifact> {
    Done(Begin<A>),
    Again,
}

/// The `identity.json` of `dir`, when it reads.
pub(crate) fn identity_of(dir: &Path) -> Option<IdentityFile> {
    match read_record::<IdentityFile>(&dir.join(IDENTITY_FILE)) {
        Read::Ok(f) => Some(f),
        _ => None,
    }
}

/// The state of a result folder known to hold `run_id`.
pub(crate) fn folder_state(dir: &Path, run_id: &Key, kinds: &Kinds) -> State {
    let dir = dir.to_path_buf();
    match read_record::<IdentityFile>(&dir.join(IDENTITY_FILE)) {
        Read::Absent => return State::HalfMade { owner: lock_owner(&dir.join(LOCK_FILE)).ok().flatten(), dir },
        Read::Bad(e) => return State::Damaged { dir, what: e.to_string() },
        Read::Ok(f) if &f.run_id != run_id => {
            return State::Damaged { dir, what: format!("its identity.json names {}", f.run_id) }
        }
        Read::Ok(_) => {}
    }
    match read_record::<RunFile>(&dir.join(RUN_FILE)) {
        Read::Absent => State::HalfMade { owner: lock_owner(&dir.join(LOCK_FILE)).ok().flatten(), dir },
        Read::Bad(e) => State::Damaged { dir, what: e.to_string() },
        Read::Ok(run) => match run.status {
            Status::Running => State::Running { owner: run.owner, dir },
            Status::Failed => State::Failed { failure: run.failure, dir },
            Status::Completed => match damage(&dir, &run, run_id, kinds) {
                Some(what) => State::Damaged { dir, what },
                None => State::Completed { dir, run: Box::new(run) },
            },
        },
    }
}

/// The visible (not dot-named) subdirectories of `dir`, in name order.
pub(crate) fn visible_dirs(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut out: Vec<PathBuf> = entries
        .flatten()
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .filter(|e| !e.file_name().to_string_lossy().starts_with('.'))
        .map(|e| e.path())
        .collect();
    out.sort();
    out
}

/// Every visible subdirectory of each of `dirs` whose name ends in
/// `-<suffix>`, in name order.
pub(crate) fn children_with_suffix(dirs: &[PathBuf], suffix: &str) -> Vec<PathBuf> {
    let tail = format!("-{suffix}");
    dirs.iter()
        .flat_map(|d| visible_dirs(d))
        .filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().ends_with(&tail)))
        .collect()
}

/// Moves an in-place folder's earlier attempt into `.attempts/<n>/`: every
/// file at its root except `identity.json` and the lock files, `run.json`
/// kept as `attempt.json`; then clears its subdirectories. The last
/// [`MAX_PRESERVED_ATTEMPTS`] attempts are kept.
fn preserve_attempt(dir: &Path) -> Result<()> {
    let attempts = dir.join(ATTEMPTS_DIR);
    let mut kept: Vec<u64> = std::fs::read_dir(&attempts)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| e.file_name().to_string_lossy().parse().ok())
        .collect();
    kept.sort_unstable();
    let has_record = dir.join(RUN_FILE).is_file();
    if has_record {
        let next = kept.last().map_or(1, |n| n + 1);
        let dest = attempts.join(next.to_string());
        std::fs::create_dir_all(&dest).at("create directory", &dest)?;
        for entry in std::fs::read_dir(dir).at("list", dir)?.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name == IDENTITY_FILE || is_lock_entry(&name) || !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                continue;
            }
            let to = if name == RUN_FILE { dest.join(ATTEMPT_RECORD) } else { dest.join(&name) };
            std::fs::rename(entry.path(), &to).at("move into the earlier attempts", entry.path())?;
        }
        fsync_dir(&dest)?;
        kept.push(next);
        while kept.len() > MAX_PRESERVED_ATTEMPTS {
            let oldest = kept.remove(0);
            remove_all(&attempts.join(oldest.to_string()))?;
        }
        fsync_dir(&attempts)?;
    }
    for entry in std::fs::read_dir(dir).at("list", dir)?.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name == ATTEMPTS_DIR || !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
        remove_all(&entry.path())?;
    }
    fsync_dir(dir)
}

/// The ids of every invocation that may hold `label`, newest first.
pub(crate) fn label_holders(inner: &Inner, label: &str) -> Vec<Invocation> {
    inner.invocations().iter().rev().filter(|i| i.record.label() == Some(label) && i.record.all_completed()).cloned().collect()
}

#[cfg(test)]
mod tests;
