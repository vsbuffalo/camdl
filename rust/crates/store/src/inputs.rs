//! Declared inputs: arguments, settings files and environment variables, each
//! field with a class.
//!
//! The derives (`#[derive(Inputs)]`, `#[derive(FileInputs)]`,
//! `#[derive(EnvInputs)]`) implement the traits here. Each refuses a field
//! with no `#[input(..)]` class, and splits the struct by class into one
//! generated struct per destination. The declared part, the identity,
//! `expands` and `adds` fields, has private fields and one `into_parts` that
//! returns them all, so a new input breaks every caller until it is handled.
//! The split reads no file and cannot fail.

use std::cell::RefCell;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::{At, Error, Result};
use crate::fsutil::write_file_synced;
use crate::hash::Key;
use crate::identity::{Content, Hashed, Identity};
use crate::record::ConfigFile;
use crate::run::FileHandle;

/// What an input may do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    /// Can change a result; hashed into the run id.
    Identity,
    /// Decides which results a batch has.
    Expands,
    /// Requests a by-product.
    Adds,
    /// Used by the engine; must not change a result.
    Execution,
    /// A path to write a copy or a rendering of stored files to.
    Export,
    /// How an export is rendered.
    Render,
    /// A request to the store.
    Store,
    /// Kept on record; nothing reads it.
    Recorded,
    /// Affects only what is printed; recorded nowhere.
    Skip,
}

/// One field, for the class checks. A field of a group is named `group.field`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldInfo {
    /// The clap id, the settings key, or the variable.
    pub name: &'static str,
    pub class: Class,
    /// The path given as `vary = ...`; `None` uses [`Vary`].
    pub vary: Option<&'static str>,
    /// The path given as `exercised_by = ...`.
    pub exercised_by: Option<&'static str>,
}

/// The fields of a group, each renamed `group.field`, for a parent's field
/// list. The names are leaked once per field list, which is built once.
#[doc(hidden)]
pub fn __prefixed(group: &str, fields: &[FieldInfo]) -> Vec<FieldInfo> {
    fields
        .iter()
        .map(|f| FieldInfo { name: Box::leak(format!("{group}.{}", f.name).into_boxed_str()), ..f.clone() })
        .collect()
}

/// Implemented by `#[derive(Inputs)]` on a clap struct.
pub trait Inputs: Sized {
    /// The identity, expands and adds fields; a group as its own `Declared`.
    type Declared;
    /// The execution fields.
    type Execution: Serialize;
    /// The export and render fields; `()` when there are none.
    type Exports;
    /// The store fields; `()` when there are none.
    type Request;
    /// Every field with its class, groups' fields included.
    fn fields() -> &'static [FieldInfo];
    fn split(self) -> Split<Self>;
    /// These arguments with one identity or render field changed to a second
    /// value: by [`Vary`] on the field's type, or by the function its
    /// `vary = ...` names. Panics naming the field for any other class.
    fn varied(&self, field: &str) -> Self;
    /// The arguments of the fixture this struct's own execution field's
    /// `exercised_by = ...` names, at two values of the field. A field of a
    /// group is reached through the group's own `exercised`.
    fn exercised(field: &str) -> (Self, Self);
}

/// The split arguments. Taken apart once, in full.
pub struct Split<I: Inputs> {
    declared: I::Declared,
    execution: I::Execution,
    exports: I::Exports,
    request: I::Request,
    given: Given,
}

impl<I: Inputs> Split<I> {
    #[doc(hidden)]
    pub fn __new(declared: I::Declared, execution: I::Execution, exports: I::Exports, request: I::Request, given: Given) -> Self {
        Split { declared, execution, exports, request, given }
    }

    /// The declared inputs, the execution inputs, the exports, the store
    /// request, and the inputs as given, for the records.
    pub fn into_parts(self) -> (I::Declared, I::Execution, I::Exports, I::Request, Given) {
        (self.declared, self.execution, self.exports, self.request, self.given)
    }
}

/// The inputs as given, for the records: the command line, and each field's
/// value as typed, by name, with store and skip fields left out.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Given {
    pub(crate) argv: Vec<String>,
    pub(crate) fields: serde_json::Map<String, serde_json::Value>,
    pub(crate) config_files: Vec<ConfigFile>,
}

impl Given {
    /// The given fields of one struct, with this process's command line.
    #[doc(hidden)]
    pub fn __from_fields(fields: serde_json::Map<String, serde_json::Value>) -> Self {
        Given { argv: std::env::args().collect(), fields, config_files: Vec::new() }
    }

    /// The same, without a command line: an environment's or a file's own fields.
    #[doc(hidden)]
    pub fn __without_argv(mut self) -> Self {
        self.argv.clear();
        self
    }

    /// One field's value as typed. A value that serde cannot write as JSON is
    /// recorded as a string saying so, since the split cannot fail.
    #[doc(hidden)]
    pub fn __value<T: Serialize + ?Sized>(value: &T) -> serde_json::Value {
        serde_json::to_value(value).unwrap_or_else(|e| serde_json::Value::String(format!("<not recordable: {e}>")))
    }

    /// Nests a group's given fields under its name.
    #[doc(hidden)]
    pub fn __nest(&mut self, name: &str, group: Given) {
        self.fields.insert(name.to_string(), serde_json::Value::Object(group.fields));
        self.config_files.extend(group.config_files);
    }

    /// Adds a parsed settings file's own `Given` under the field that named
    /// it, as `{ "path", "sha256", "keys" }`, with the digest of its raw bytes.
    pub fn with_file(mut self, field: &str, file: Given, source: &SourceBytes) -> Given {
        let path = source.typed.display().to_string();
        self.fields.insert(
            field.to_string(),
            serde_json::json!({ "path": path, "sha256": source.digest, "keys": file.fields }),
        );
        self.config_files.push(ConfigFile { field: field.to_string(), path, sha256: source.digest });
        self.config_files.extend(file.config_files);
        self
    }

    /// The command line as given.
    pub fn argv(&self) -> &[String] {
        &self.argv
    }

    /// The fields as `run.json` and the invocation record write them.
    pub fn fields(&self) -> &serde_json::Map<String, serde_json::Value> {
        &self.fields
    }
}

thread_local! {
    /// The directory of the settings file being parsed, against which a
    /// relative path inside it is read.
    static PARSE_BASE: RefCell<Option<PathBuf>> = const { RefCell::new(None) };
}

/// A file argument. `T` says what the file holds. A relative path inside a
/// settings file is read relative to that file's directory; a path given on
/// the command line is read as given. The path is recorded as typed.
pub struct InputFile<T> {
    typed: PathBuf,
    path: PathBuf,
    _holds: PhantomData<fn() -> T>,
}

impl<T> InputFile<T> {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        let typed = path.into();
        InputFile { path: typed.clone(), typed, _holds: PhantomData }
    }

    /// The path as typed.
    pub fn typed(&self) -> &Path {
        &self.typed
    }

    fn read(&self) -> Result<Vec<u8>> {
        std::fs::read(&self.path).at("read", &self.path)
    }

    fn source(&self) -> Result<SourceBytes> {
        let bytes = self.read()?;
        Ok(SourceBytes { typed: self.typed.clone(), digest: Key::digest_bytes(&bytes), bytes })
    }
}

impl<T> Clone for InputFile<T> {
    fn clone(&self) -> Self {
        InputFile { typed: self.typed.clone(), path: self.path.clone(), _holds: PhantomData }
    }
}

impl<T> std::fmt::Debug for InputFile<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "InputFile({})", self.typed.display())
    }
}

impl<T> PartialEq for InputFile<T> {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
    }
}

impl<T> std::str::FromStr for InputFile<T> {
    type Err = std::convert::Infallible;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(InputFile::new(s))
    }
}

impl<T> From<PathBuf> for InputFile<T> {
    fn from(p: PathBuf) -> Self {
        InputFile::new(p)
    }
}

impl<T> Serialize for InputFile<T> {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        self.typed.serialize(s)
    }
}

impl<'de, T> Deserialize<'de> for InputFile<T> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let typed = PathBuf::deserialize(d)?;
        let path = PARSE_BASE.with(|b| match b.borrow().as_ref() {
            Some(base) if typed.is_relative() => base.join(&typed),
            _ => typed.clone(),
        });
        Ok(InputFile { typed, path, _holds: PhantomData })
    }
}

/// Implemented by `#[derive(FileInputs)]` on a settings file. The same as
/// [`Inputs`], without exports: the derive refuses `export` and `render` on a
/// settings key. A settings file is TOML; an unknown key is a parse error.
pub trait FileInputs: DeserializeOwned + Serialize + Sized {
    type Declared;
    /// `()` when the file has no execution keys.
    type Execution: Serialize;
    /// `()` when the file has no store keys.
    type Request;
    fn fields() -> &'static [FieldInfo];
    /// The parts of this table, without its bytes; a nested table's parent
    /// calls it.
    #[doc(hidden)]
    fn __split_table(self) -> (Self::Declared, Self::Execution, Self::Request, Given);
    /// Called by [`InputFile::parse`] with the bytes it read.
    fn split(self, source: SourceBytes) -> Parsed<Self> {
        let (declared, execution, request, mut given) = self.__split_table();
        given.argv.clear();
        Parsed { declared, execution, request, given, source }
    }
    /// As [`Inputs::varied`], for one key.
    fn varied(&self, key: &str) -> Self;
}

impl<T: FileInputs> InputFile<T> {
    /// Reads, parses and splits the file, as for arguments.
    pub fn parse(self) -> Result<Parsed<T>> {
        let source = self.source()?;
        let text = std::str::from_utf8(&source.bytes)
            .map_err(|e| Error::Parse { what: self.typed.display().to_string(), why: e.to_string() })?;
        let base = self.path.parent().map(Path::to_path_buf);
        let prior = PARSE_BASE.with(|b| b.replace(base));
        let parsed: std::result::Result<T, _> = toml::from_str(text);
        PARSE_BASE.with(|b| *b.borrow_mut() = prior);
        let value = parsed.map_err(|e| Error::Parse { what: self.typed.display().to_string(), why: e.to_string() })?;
        Ok(value.split(source))
    }
}

impl<T: Content> InputFile<T> {
    /// Reads a file that is hashed whole. The path is recorded; only the
    /// content is hashed.
    pub fn content(self) -> Result<Hashed<T>> {
        let bytes = self.read()?;
        let value = T::from_bytes(&bytes)?;
        Ok(Hashed::new(value))
    }
}

/// A parsed and split settings file.
pub struct Parsed<T: FileInputs> {
    declared: T::Declared,
    execution: T::Execution,
    request: T::Request,
    given: Given,
    source: SourceBytes,
}

impl<T: FileInputs> Parsed<T> {
    /// Every part of the file: declared inputs, execution inputs, store
    /// request, the keys as given (recorded keys included), and the raw bytes.
    pub fn into_parts(self) -> (T::Declared, T::Execution, T::Request, Given, SourceBytes) {
        (self.declared, self.execution, self.request, self.given, self.source)
    }
}

/// A settings file's raw bytes, its path as given, and their digest.
#[derive(Debug, Clone)]
pub struct SourceBytes {
    typed: PathBuf,
    bytes: Vec<u8>,
    digest: Key,
}

impl SourceBytes {
    pub fn digest(&self) -> Key {
        self.digest
    }

    pub fn path(&self) -> &Path {
        &self.typed
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Writes the bytes as they were read, to a file of a run (a provenance
    /// file).
    pub fn write(&self, to: &FileHandle) -> Result<()> {
        to.write_all(&self.bytes)
    }

    /// Writes the bytes to a plain path, for a test that needs a settings file
    /// on disk.
    #[doc(hidden)]
    pub fn __write_to(&self, path: &Path) -> Result<()> {
        write_file_synced(path, &self.bytes)
    }
}

/// What [`EnvInputs::read`] returns: the identity, execution and store
/// parts, and the variables as given.
pub type EnvParts<E> = (<E as EnvInputs>::Identity, <E as EnvInputs>::Execution, <E as EnvInputs>::Request, Given);

/// Implemented by `#[derive(EnvInputs)]`. The project lists its variables once;
/// nothing else reads the environment.
pub trait EnvInputs: Sized {
    /// The identity-class variables, hashed as read.
    type Identity: Identity;
    type Execution;
    type Request;
    fn fields() -> &'static [FieldInfo];
    /// Reads every declared variable once, and no other.
    fn read() -> Result<EnvParts<Self>>;
}

/// Reads one environment variable by the rule its field's type sets.
#[doc(hidden)]
pub mod env {
    use super::*;
    use std::collections::BTreeMap;

    fn text(var: &str) -> Result<Option<String>> {
        match std::env::var(var) {
            Ok(v) => Ok(Some(v)),
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(std::env::VarError::NotUnicode(_)) => {
                Err(Error::Parse { what: format!("the environment variable {var}"), why: "not UTF-8".into() })
            }
        }
    }

    /// The raw value, for a field with its own `parse = ...`.
    pub fn raw(var: &str) -> Result<Option<String>> {
        text(var)
    }

    /// True when the variable is set, whatever its value.
    pub fn flag(var: &str) -> Result<bool> {
        Ok(std::env::var_os(var).is_some())
    }

    /// Every set variable whose name starts with `prefix`, keyed by its full name.
    pub fn prefixed(prefix: &str) -> Result<BTreeMap<String, String>> {
        let mut out = BTreeMap::new();
        for (k, v) in std::env::vars_os() {
            let Some(k) = k.to_str() else { continue };
            if k.starts_with(prefix) {
                let v = v.into_string().map_err(|_| Error::Parse {
                    what: format!("the environment variable {k}"),
                    why: "not UTF-8".into(),
                })?;
                out.insert(k.to_string(), v);
            }
        }
        Ok(out)
    }

    /// The variable's text, `None` when unset.
    pub fn optional<T: From<String>>(var: &str) -> Result<Option<T>> {
        Ok(text(var)?.map(T::from))
    }

    /// The variable's text; an unset variable is refused.
    pub fn required<T: From<String>>(var: &str) -> Result<T> {
        text(var)?.map(T::from).ok_or_else(|| Error::Parse {
            what: format!("the environment variable {var}"),
            why: "it is not set".into(),
        })
    }

    /// The result of a field's own parser, with the variable named in its error.
    pub fn parsed<T>(var: &str, r: std::result::Result<T, String>) -> Result<T> {
        r.map_err(|why| Error::Parse { what: format!("the environment variable {var}"), why })
    }
}

/// Consumes a declared input a command does not use, with the reason.
pub fn release<T>(value: T, reason: &'static str) {
    // The reason is for the reader of the call; nothing records it.
    let _ = reason;
    drop(value);
}

/// Another valid value of the same type, for the generated identity tests.
pub trait Vary {
    fn vary(&self) -> Self;
}

macro_rules! vary_int {
    ($($t:ty),*) => {$(
        impl Vary for $t { fn vary(&self) -> Self { self.wrapping_add(1) } }
    )*};
}
vary_int!(u8, u16, u32, u64, usize, i32, i64);

impl Vary for f64 {
    fn vary(&self) -> Self {
        if *self == 0.0 {
            1.0
        } else {
            self * 2.0
        }
    }
}

impl Vary for bool {
    fn vary(&self) -> Self {
        !self
    }
}

impl Vary for String {
    fn vary(&self) -> Self {
        format!("{self}_varied")
    }
}

impl<T: Vary + Default> Vary for Option<T> {
    fn vary(&self) -> Self {
        match self {
            Some(v) => Some(v.vary()),
            None => Some(T::default()),
        }
    }
}

/// Each value varied; an empty map gains one entry.
impl<K: Clone + Ord + Default, V: Vary + Default> Vary for std::collections::BTreeMap<K, V> {
    fn vary(&self) -> Self {
        if self.is_empty() {
            [(K::default(), V::default())].into()
        } else {
            self.iter().map(|(k, v)| (k.clone(), v.vary())).collect()
        }
    }
}

/// Each element varied; an empty list gains one element.
impl<T: Vary + Default> Vary for Vec<T> {
    fn vary(&self) -> Self {
        if self.is_empty() {
            vec![T::default()]
        } else {
            self.iter().map(Vary::vary).collect()
        }
    }
}
