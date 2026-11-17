//! The Rust extractor: turns rustdoc's JSON output for a set of crates into
//! the renderer's input ([`crate::model::Input`]).
//!
//! rustdoc resolves every path the compiler resolves, so the links this
//! extractor writes are exact, as the OCaml extractor's are from the `.cmt`
//! files: a name in a declaration links only when rustdoc says which item it
//! names. What rustdoc does not record — the uses of a type inside a function
//! body — is left unlinked rather than guessed.
//!
//! Each crate's modules become [`Module`]s named by their Rust path, the
//! crate first (`sim::inference::pgas`); its structs, enums, unions, type
//! aliases and traits become [`TypeDecl`]s. Functions, constants and impl
//! blocks are not documented on the pages, but their signatures are linked
//! on the source pages.
//!
//! rustdoc's JSON format is unstable and versioned by `format_version`. The
//! extractor reads exactly [`rustdoc_types::FORMAT_VERSION`] and refuses any
//! other, so a toolchain bump fails the docs build instead of rendering
//! silently wrong links.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use rustdoc_types::{
    Attribute, Crate, GenericArg, GenericArgs, GenericBound, GenericParamDefKind, Generics, Id,
    Item, ItemEnum, ItemKind, StructKind, Term, Type, VariantKind, WherePredicate,
};
use serde::Deserialize;

use crate::model::{
    DocLink, Impl, Input, Kind, Language, LinkTo, Member, Module, SourceFile, Span, SpanClass,
    Target, TypeDecl,
};
use crate::rustlex::{self, Lines, PathRun, Tok, Token};

/// One crate to extract: the name its modules are listed under, and the
/// rustdoc JSON file written for it.
#[derive(Debug, Clone)]
pub struct CrateSpec {
    /// The crate's name on the pages, which prefixes its module paths. For a
    /// library this is the crate's own name; a binary crate may be listed
    /// under its package's name instead (`cli` for the `camdl` binary).
    pub label: String,
    pub json: PathBuf,
}

impl CrateSpec {
    /// Parses `label=path`.
    pub fn parse(s: &str) -> Result<CrateSpec, String> {
        match s.split_once('=') {
            Some((label, json)) if !label.is_empty() && !json.is_empty() => Ok(CrateSpec {
                label: label.to_string(),
                json: PathBuf::from(json),
            }),
            _ => Err(format!("--crate {s:?} is not label=path/to/rustdoc.json")),
        }
    }
}

/// Refuses rustdoc JSON of any format version but the one this extractor
/// was written for.
pub fn check_format_version(json: &str) -> Result<(), String> {
    #[derive(Deserialize)]
    struct Head {
        format_version: u32,
    }
    let head: Head = serde_json::from_str(json)
        .map_err(|e| format!("not rustdoc JSON (no format_version): {e}"))?;
    if head.format_version != rustdoc_types::FORMAT_VERSION {
        return Err(format!(
            "rustdoc JSON format_version {} is not {}, the version this extractor reads. \
             The toolchain in rust/rust-toolchain.toml writes a different format than \
             rust/crates/typedoc was written for: pin `rustdoc-types` in \
             rust/crates/typedoc/Cargo.toml to the release for format {}, then fix what \
             no longer compiles in src/rustdoc.rs",
            head.format_version,
            rustdoc_types::FORMAT_VERSION,
            head.format_version
        ));
    }
    Ok(())
}

/// Reads one crate's rustdoc JSON, checking its format version first.
fn load(spec: &CrateSpec) -> Result<Crate, String> {
    let at = |e: String| format!("{}: {e}", spec.json.display());
    let text = fs::read_to_string(&spec.json).map_err(|e| at(e.to_string()))?;
    check_format_version(&text).map_err(at)?;
    let krate: Crate = serde_json::from_str(&text).map_err(|e| at(e.to_string()))?;
    if !krate.includes_private {
        return Err(at(
            "written without --document-private-items; the internals pages document private \
             items too"
                .to_string(),
        ));
    }
    Ok(krate)
}

/// A crate's module, as the walk from the crate root finds it.
struct ModInfo {
    id: Id,
    name: String,
    /// The source file, relative to the repository root.
    file: String,
    /// The types the module declares, in rustdoc's order.
    types: Vec<Id>,
}

/// One loaded crate and what the walk over its modules found.
struct Loaded {
    label: String,
    krate: Crate,
    mods: Vec<ModInfo>,
    /// Each documented type: its module's name and its own name.
    type_at: HashMap<Id, (String, String)>,
    /// Each module, by id: its name.
    mod_at: HashMap<Id, String>,
    /// For a variant, field or trait item: the documented type it belongs to.
    parent: HashMap<Id, Id>,
}

/// What every crate declares, for resolving names across crates.
struct World {
    types: BTreeSet<(String, String)>,
    modules: BTreeSet<String>,
    /// rustdoc's crate name (`camdl`) to the label it is listed under (`cli`).
    labels: HashMap<String, String>,
    /// The source files of every module, relative to the repository root.
    files: BTreeSet<String>,
}

/// Where a crate's source paths are rooted: rustdoc writes them relative to
/// the directory cargo ran in, the workspace root, which is `prefix` below
/// the repository root.
fn repo_path(prefix: &str, filename: &Path) -> Option<String> {
    if filename.is_absolute() {
        return None;
    }
    let f = filename.to_string_lossy().replace('\\', "/");
    Some(if prefix.is_empty() || prefix == "." {
        f
    } else {
        format!("{}/{f}", prefix.trim_end_matches('/'))
    })
}

fn is_type_item(inner: &ItemEnum) -> bool {
    matches!(
        inner,
        ItemEnum::Struct(_)
            | ItemEnum::Enum(_)
            | ItemEnum::Union(_)
            | ItemEnum::Trait(_)
            | ItemEnum::TypeAlias(_)
    )
}

/// Walks the module tree from the crate root, in rustdoc's item order.
fn walk(label: &str, krate: Crate, prefix: &str) -> Result<Loaded, String> {
    let mut out = Loaded {
        label: label.to_string(),
        mods: Vec::new(),
        type_at: HashMap::new(),
        mod_at: HashMap::new(),
        parent: HashMap::new(),
        krate,
    };
    let mut stack = vec![(out.krate.root, label.to_string())];
    while let Some((id, name)) = stack.pop() {
        let item = &out.krate.index[&id];
        let ItemEnum::Module(m) = &item.inner else {
            return Err(format!("{name}: expected a module"));
        };
        let file = item
            .span
            .as_ref()
            .and_then(|s| repo_path(prefix, &s.filename))
            .ok_or_else(|| format!("module {name} has no source file in the workspace"))?;
        let mut types = Vec::new();
        let mut children = Vec::new();
        for child in &m.items {
            let Some(it) = out.krate.index.get(child) else {
                continue;
            };
            match &it.inner {
                ItemEnum::Module(_) => {
                    let n = it.name.as_deref().unwrap_or("_");
                    children.push((*child, format!("{name}::{n}")));
                }
                inner if is_type_item(inner) => {
                    let in_file = it
                        .span
                        .as_ref()
                        .and_then(|s| repo_path(prefix, &s.filename))
                        .is_some_and(|f| f == file);
                    if !in_file {
                        return Err(format!(
                            "{name}::{}: declared outside the module's file {file} (by a macro?); \
                             the pages cannot show it",
                            it.name.as_deref().unwrap_or("?")
                        ));
                    }
                    types.push(*child);
                    out.type_at
                        .insert(*child, (name.clone(), it.name.clone().unwrap_or_default()));
                }
                _ => {}
            }
        }
        types.sort_by_key(|t| out.krate.index[t].span.as_ref().map(|s| s.begin));
        out.mod_at.insert(id, name.clone());
        out.mods.push(ModInfo {
            id,
            name,
            file,
            types,
        });
        // Depth first, children in item order.
        children.reverse();
        stack.extend(children);
    }
    for (&id, _) in out.type_at.clone().iter() {
        let item = &out.krate.index[&id];
        let mut own = |c: &Id| {
            out.parent.insert(*c, id);
        };
        match &item.inner {
            ItemEnum::Struct(s) => match &s.kind {
                StructKind::Plain { fields, .. } => fields.iter().for_each(&mut own),
                StructKind::Tuple(fields) => fields.iter().flatten().for_each(&mut own),
                StructKind::Unit => {}
            },
            ItemEnum::Union(u) => u.fields.iter().for_each(&mut own),
            ItemEnum::Trait(t) => t.items.iter().for_each(&mut own),
            ItemEnum::Enum(e) => {
                for v in &e.variants {
                    own(v);
                    if let Some(ItemEnum::Variant(var)) = out.krate.index.get(v).map(|i| &i.inner) {
                        match &var.kind {
                            VariantKind::Plain => {}
                            VariantKind::Tuple(f) => f.iter().flatten().for_each(&mut own),
                            VariantKind::Struct { fields, .. } => fields.iter().for_each(&mut own),
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Ok(out)
}

/// Resolves rustdoc ids of one crate to the pages' targets.
struct Resolver<'a> {
    c: &'a Loaded,
    world: &'a World,
    prefix: &'a str,
}

impl Resolver<'_> {
    /// The documented type an id names, in this crate or another input crate.
    fn target(&self, id: &Id) -> Option<Target> {
        if let Some((module, ty)) = self.c.type_at.get(id) {
            return Some(Target {
                module: module.clone(),
                ty: ty.clone(),
            });
        }
        let s = self.c.krate.paths.get(id)?;
        if s.crate_id == 0 {
            return None;
        }
        if !matches!(
            s.kind,
            ItemKind::Struct
                | ItemKind::Enum
                | ItemKind::Union
                | ItemKind::Trait
                | ItemKind::TypeAlias
        ) {
            return None;
        }
        self.external(&s.crate_id, &s.path)
    }

    /// The documented type at a path of another crate: its crate's label,
    /// its modules, and its name.
    fn external(&self, crate_id: &u32, path: &[String]) -> Option<Target> {
        let (name, mods) = path.split_last()?;
        let module = self.external_module(crate_id, mods)?;
        let key = (module, name.clone());
        self.world.types.contains(&key).then_some(Target {
            module: key.0,
            ty: key.1,
        })
    }

    fn external_module(&self, crate_id: &u32, mods: &[String]) -> Option<String> {
        let krate = &self.c.krate.external_crates.get(crate_id)?.name;
        let label = self.world.labels.get(krate)?;
        let mut module = label.clone();
        for m in mods.iter().skip(1) {
            module = format!("{module}::{m}");
        }
        Some(module)
    }

    /// What a doc comment's link to an id points to on the pages.
    fn link(&self, id: &Id) -> Option<LinkTo> {
        let as_type = |t: Target| LinkTo::Type {
            module: t.module,
            ty: t.ty,
        };
        if let Some(t) = self.target(id) {
            return Some(as_type(t));
        }
        if let Some(p) = self.c.parent.get(id) {
            return self.target(p).map(as_type);
        }
        if let Some(m) = self.c.mod_at.get(id) {
            return Some(LinkTo::Module { module: m.clone() });
        }
        if let Some(item) = self.c.krate.index.get(id).filter(|i| i.crate_id == 0) {
            let span = item.span.as_ref()?;
            let path = repo_path(self.prefix, &span.filename)?;
            return self.world.files.contains(&path).then_some(LinkTo::Source {
                path,
                line: span.begin.0 as u32,
            });
        }
        let s = self.c.krate.paths.get(id)?;
        if s.crate_id == 0 {
            return None;
        }
        match s.kind {
            ItemKind::Module => {
                let m = self.external_module(&s.crate_id, &s.path)?;
                self.world
                    .modules
                    .contains(&m)
                    .then_some(LinkTo::Module { module: m })
            }
            ItemKind::Variant => {
                let (_, ty) = s.path.split_last()?;
                self.external(&s.crate_id, ty).map(as_type)
            }
            _ => None,
        }
    }

    /// An item's doc links, sorted by text, for those that point somewhere on
    /// the pages.
    fn doc_links(&self, item: &Item) -> Vec<DocLink> {
        let mut out: Vec<DocLink> = item
            .links
            .iter()
            .filter_map(|(text, id)| {
                Some(DocLink {
                    text: text.trim_matches('`').to_string(),
                    to: self.link(id)?,
                })
            })
            .collect();
        out.sort_by(|a, b| a.text.cmp(&b.text));
        out.dedup_by(|a, b| a.text == b.text);
        out
    }
}

/// The type paths in an item's signature, each with the id rustdoc resolved
/// it to, for finding them in the item's source text.
///
/// rustdoc does not record where in the text a path is written, nor always
/// the path as written: an imported type comes back by its full path
/// (`other::shapes::Shape` for a field written `Shape`) and a trait in an
/// impl header by its last segment (`Area` for `other::Area`). So a path in
/// the text is matched by its last segment, and only when one of the two
/// paths ends with the other. Within one item a name is unambiguous unless
/// the item names two types of that name; then the name is dropped, and
/// neither is linked rather than one wrongly. A path starting with `Self` or
/// with one of the item's generic parameters (`T::Item`) never matches: it
/// names an associated item, which rustdoc gives no path for.
#[derive(Default)]
struct PathMap {
    /// By last segment: the full path rustdoc gave and its id, or `None`
    /// when the item names two different items with that last segment.
    map: HashMap<String, Option<(String, Id)>>,
    /// The generic parameters in scope.
    generic: BTreeSet<String>,
}

impl PathMap {
    fn add(&mut self, path: &str, id: Id) {
        let last = path.rsplit("::").next().unwrap_or(path).to_string();
        self.map
            .entry(last)
            .and_modify(|e| {
                if e.as_ref().is_some_and(|(_, i)| *i != id) {
                    *e = None;
                }
            })
            .or_insert(Some((path.to_string(), id)));
    }

    /// The item a path written in the text names, if the signature has it.
    fn get(&self, written: &str) -> Option<Id> {
        let mut segs = written.trim_start_matches("::").split("::");
        let first = segs.next()?;
        let qualified = written.contains("::");
        if qualified && (first == "Self" || self.generic.contains(first)) {
            return None;
        }
        let last = written.rsplit("::").next().unwrap_or(written);
        let (path, id) = self.map.get(last)?.as_ref()?;
        let suffix =
            |long: &str, short: &str| long == short || long.ends_with(&format!("::{short}"));
        (suffix(path, written) || suffix(written, path)).then_some(*id)
    }

    fn generic_names(&mut self, g: &Generics) {
        self.generic.extend(g.params.iter().map(|p| p.name.clone()));
    }

    fn path(&mut self, p: &rustdoc_types::Path) {
        self.add(&p.path, p.id);
        if let Some(a) = &p.args {
            self.args(a);
        }
    }

    fn args(&mut self, a: &GenericArgs) {
        match a {
            GenericArgs::AngleBracketed { args, constraints } => {
                for g in args {
                    if let GenericArg::Type(t) = g {
                        self.ty(t);
                    }
                }
                for c in constraints {
                    if let Some(a) = &c.args {
                        self.args(a);
                    }
                    match &c.binding {
                        rustdoc_types::AssocItemConstraintKind::Equality(Term::Type(t)) => {
                            self.ty(t)
                        }
                        rustdoc_types::AssocItemConstraintKind::Equality(Term::Constant(_)) => {}
                        rustdoc_types::AssocItemConstraintKind::Constraint(b) => self.bounds(b),
                    }
                }
            }
            GenericArgs::Parenthesized { inputs, output } => {
                inputs.iter().for_each(|t| self.ty(t));
                if let Some(o) = output {
                    self.ty(o);
                }
            }
            GenericArgs::ReturnTypeNotation => {}
        }
    }

    fn bounds(&mut self, bounds: &[GenericBound]) {
        for b in bounds {
            if let GenericBound::TraitBound { trait_, .. } = b {
                self.path(trait_);
            }
        }
    }

    fn ty(&mut self, t: &Type) {
        match t {
            Type::ResolvedPath(p) => self.path(p),
            Type::DynTrait(d) => d.traits.iter().for_each(|p| self.path(&p.trait_)),
            Type::FunctionPointer(f) => {
                f.sig.inputs.iter().for_each(|(_, t)| self.ty(t));
                if let Some(o) = &f.sig.output {
                    self.ty(o);
                }
            }
            Type::Tuple(ts) => ts.iter().for_each(|t| self.ty(t)),
            Type::Slice(t) => self.ty(t),
            Type::Array { type_, .. }
            | Type::Pat { type_, .. }
            | Type::RawPointer { type_, .. }
            | Type::BorrowedRef { type_, .. } => self.ty(type_),
            Type::ImplTrait(b) => self.bounds(b),
            Type::QualifiedPath {
                args,
                self_type,
                trait_,
                ..
            } => {
                self.ty(self_type);
                if let Some(p) = trait_ {
                    self.path(p);
                }
                if let Some(a) = args {
                    self.args(a);
                }
            }
            Type::Generic(_) | Type::Primitive(_) | Type::Infer => {}
        }
    }

    fn generics(&mut self, g: &Generics) {
        self.generic_names(g);
        for p in &g.params {
            match &p.kind {
                GenericParamDefKind::Type {
                    bounds, default, ..
                } => {
                    self.bounds(bounds);
                    if let Some(d) = default {
                        self.ty(d);
                    }
                }
                GenericParamDefKind::Const { type_, .. } => self.ty(type_),
                GenericParamDefKind::Lifetime { .. } => {}
            }
        }
        for w in &g.where_predicates {
            match w {
                WherePredicate::BoundPredicate { type_, bounds, .. } => {
                    self.ty(type_);
                    self.bounds(bounds);
                }
                WherePredicate::EqPredicate { lhs, rhs } => {
                    self.ty(lhs);
                    if let Term::Type(t) = rhs {
                        self.ty(t);
                    }
                }
                WherePredicate::LifetimePredicate { .. } => {}
            }
        }
    }

    /// The paths of a field or variant's types, by its item id.
    fn field(&mut self, krate: &Crate, id: &Id) {
        if let Some(ItemEnum::StructField(t)) = krate.index.get(id).map(|i| &i.inner) {
            self.ty(t);
        }
    }

    /// Every path an item's own signature writes: for a type, its generics
    /// and the types of its fields or variants; for a trait, its generics,
    /// supertraits and its items' signatures, which its declaration shows;
    /// for a function, its generics, parameters and result; for an impl, its
    /// header.
    fn of_item(krate: &Crate, item: &Item) -> PathMap {
        let mut m = PathMap::default();
        m.collect(krate, item);
        m
    }

    fn collect(&mut self, krate: &Crate, item: &Item) {
        let m = self;
        match &item.inner {
            ItemEnum::Struct(s) => {
                m.generics(&s.generics);
                match &s.kind {
                    StructKind::Plain { fields, .. } => {
                        fields.iter().for_each(|f| m.field(krate, f))
                    }
                    StructKind::Tuple(f) => f.iter().flatten().for_each(|f| m.field(krate, f)),
                    StructKind::Unit => {}
                }
            }
            ItemEnum::Union(u) => {
                m.generics(&u.generics);
                u.fields.iter().for_each(|f| m.field(krate, f));
            }
            ItemEnum::Enum(e) => {
                m.generics(&e.generics);
                for v in &e.variants {
                    if let Some(ItemEnum::Variant(var)) = krate.index.get(v).map(|i| &i.inner) {
                        match &var.kind {
                            VariantKind::Plain => {}
                            VariantKind::Tuple(f) => {
                                f.iter().flatten().for_each(|f| m.field(krate, f))
                            }
                            VariantKind::Struct { fields, .. } => {
                                fields.iter().for_each(|f| m.field(krate, f))
                            }
                        }
                    }
                }
            }
            ItemEnum::Trait(t) => {
                m.generics(&t.generics);
                m.bounds(&t.bounds);
                for child in t.items.iter().filter_map(|c| krate.index.get(c)) {
                    m.collect(krate, child);
                }
            }
            ItemEnum::TypeAlias(t) => {
                m.generics(&t.generics);
                m.ty(&t.type_);
            }
            ItemEnum::Function(f) => {
                m.generics(&f.generics);
                f.sig.inputs.iter().for_each(|(_, t)| m.ty(t));
                if let Some(o) = &f.sig.output {
                    m.ty(o);
                }
            }
            ItemEnum::Impl(i) => {
                m.generics(&i.generics);
                if let Some(p) = &i.trait_ {
                    m.path(p);
                }
                m.ty(&i.for_);
            }
            ItemEnum::AssocType {
                generics,
                bounds,
                type_,
            } => {
                m.generics(generics);
                m.bounds(bounds);
                if let Some(t) = type_ {
                    m.ty(t);
                }
            }
            ItemEnum::Constant { type_, .. } | ItemEnum::AssocConst { type_, .. } => m.ty(type_),
            ItemEnum::Static(s) => m.ty(&s.type_),
            ItemEnum::StructField(t) => m.ty(t),
            _ => {}
        }
    }
}

/// A source file's text, its line starts and its tokens.
struct Text {
    src: String,
    lines: Lines,
    toks: Vec<Token>,
}

impl Text {
    fn new(src: String) -> Text {
        let lines = Lines::new(&src);
        let toks = rustlex::lex(&src);
        Text { src, lines, toks }
    }

    /// The byte offset of a rustdoc position: a line from 1 and a column in
    /// characters from 1.
    fn offset(&self, line: usize, col: usize) -> Option<usize> {
        let start = self.lines.start(line)?;
        let rest = &self.src[start..];
        let line_text = &rest[..rest.find('\n').unwrap_or(rest.len())];
        let k = col.checked_sub(1)?;
        match line_text.char_indices().nth(k) {
            Some((b, _)) => Some(start + b),
            None if k == line_text.chars().count() => Some(start + line_text.len()),
            None => None,
        }
    }

    /// The byte range of a rustdoc span, its end column inclusive.
    fn range(&self, span: &rustdoc_types::Span) -> Option<(usize, usize)> {
        let start = self.offset(span.begin.0, span.begin.1)?;
        let last = self.offset(span.end.0, span.end.1)?;
        let end = last + self.src[last..].chars().next().map_or(0, char::len_utf8);
        (start <= end).then_some((start, end))
    }

    /// The tokens wholly inside a byte range.
    fn toks_in(&self, start: usize, end: usize) -> &[Token] {
        let a = self.toks.partition_point(|t| t.start < start);
        let b = self.toks.partition_point(|t| t.start < end);
        &self.toks[a..b]
    }
}

fn tok_text<'s>(src: &'s str, t: &Token) -> &'s str {
    &src[t.start..t.end]
}

/// The first `{`, or the first of `stop`, outside parentheses and brackets,
/// in tokens: where a signature's body or initialiser starts.
fn first_at_depth0(src: &str, toks: &[Token], stop: &[&str]) -> Option<usize> {
    let mut depth = 0i32;
    for t in toks.iter().filter(|t| t.kind == Tok::Punct) {
        match tok_text(src, t) {
            "(" | "[" => depth += 1,
            ")" | "]" => depth -= 1,
            s if depth == 0 && (s == "{" || stop.contains(&s)) => return Some(t.start),
            _ => {}
        }
    }
    None
}

/// The end of the brace group opened at byte `open` (the `{` itself).
fn brace_end(src: &str, toks: &[Token], open: usize) -> Option<usize> {
    let mut depth = 0i32;
    for t in toks
        .iter()
        .filter(|t| t.start >= open && t.kind == Tok::Punct)
    {
        match tok_text(src, t) {
            "{" => depth += 1,
            "}" => {
                depth -= 1;
                if depth == 0 {
                    return Some(t.end);
                }
            }
            _ => {}
        }
    }
    None
}

/// Source text with comments removed: a line left blank by removing its
/// comment is dropped, trailing space is trimmed, and `indent` bytes of
/// leading space are removed from every line after the first.
fn strip_comments(text: &str, indent: usize) -> String {
    const MARK: char = '\u{1}';
    let mut s = String::with_capacity(text.len());
    for t in rustlex::lex(text) {
        match t.kind {
            Tok::Comment | Tok::Doc => {
                s.push(MARK);
                // A block comment's line ends are kept, so the lines after
                // it stay where they were.
                for _ in text[t.start..t.end].matches('\n') {
                    s.push('\n');
                    s.push(MARK);
                }
            }
            _ => s.push_str(&text[t.start..t.end]),
        }
    }
    let mut out = Vec::new();
    for (i, line) in s.split('\n').enumerate() {
        let had = line.contains(MARK);
        let clean: String = line.chars().filter(|c| *c != MARK).collect();
        if had && clean.trim().is_empty() {
            continue;
        }
        let clean = clean.trim_end();
        let lead = clean.len() - clean.trim_start().len();
        let clean = if i == 0 {
            clean
        } else {
            &clean[lead.min(indent)..]
        };
        out.push(clean.to_string());
    }
    while out.last().is_some_and(|l| l.is_empty()) {
        out.pop();
    }
    out.join("\n")
}

/// Text on one line: every run of whitespace becomes one space.
fn one_line(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Source written over several lines, on one: as [`one_line`], without the
/// space a line break left after an opening or before a closing delimiter,
/// or a trailing comma before one.
fn joined(s: &str) -> String {
    one_line(s)
        .replace(", )", ")")
        .replace(", }", " }")
        .replace(", ]", "]")
        .replace("( ", "(")
        .replace(" )", ")")
        .replace("[ ", "[")
        .replace(" ]", "]")
}

/// Replaces the bodies of the functions declared directly inside a brace
/// block (a trait's provided methods) with `{ … }`, so a trait's declaration
/// shows its signatures, not its default implementations.
fn elide_method_bodies(text: &str) -> String {
    let toks = rustlex::lex(text);
    let mut cuts: Vec<(usize, usize)> = Vec::new();
    let mut depth = 0i32;
    let mut k = 0;
    while k < toks.len() {
        let t = &toks[k];
        let s = tok_text(text, t);
        if t.kind == Tok::Punct && s == "{" {
            depth += 1;
        } else if t.kind == Tok::Punct && s == "}" {
            depth -= 1;
        } else if t.kind == Tok::Keyword && s == "fn" && depth == 1 {
            let rest = &toks[k..];
            if let Some(open) = first_at_depth0(text, rest, &[";"]) {
                if &text[open..open + 1] == "{" {
                    if let Some(end) = brace_end(text, rest, open) {
                        cuts.push((open, end));
                        k = toks.partition_point(|t| t.start < end);
                        continue;
                    }
                }
            }
        }
        k += 1;
    }
    let mut out = String::new();
    let mut at = 0;
    for (a, b) in cuts {
        out.push_str(&text[at..a]);
        out.push_str("{ … }");
        at = b;
    }
    out.push_str(&text[at..]);
    out
}

/// The highlight class of a token, or `None` for one shown plain.
fn class_of(kind: Tok) -> Option<SpanClass> {
    match kind {
        Tok::Keyword => Some(SpanClass::Keyword),
        Tok::Primitive => Some(SpanClass::BuiltinType),
        Tok::Comment => Some(SpanClass::Comment),
        Tok::Doc => Some(SpanClass::Doc),
        Tok::Str => Some(SpanClass::String),
        Tok::Char => Some(SpanClass::Char),
        Tok::Number => Some(SpanClass::Number),
        Tok::Ident | Tok::Lifetime | Tok::Punct | Tok::Space => None,
    }
}

/// Pushes a span over `start..end` of `src`, split at line ends.
fn push_span(
    out: &mut Vec<Span>,
    src: &str,
    lines: &Lines,
    (start, end): (usize, usize),
    class: SpanClass,
    target: Option<Target>,
) {
    let mut a = start;
    while a < end {
        let line_end = src[a..end].find('\n').map_or(end, |k| a + k);
        if line_end > a {
            let (line, col) = lines.at(a);
            out.push(Span {
                line,
                col,
                len: line_end - a,
                class,
                target: target.clone(),
            });
        }
        a = line_end + 1;
    }
}

/// What a path at a byte offset links to, given the path's text and whether
/// it is a name being bound.
type LinkFn<'a> = dyn Fn(usize, &str, bool) -> Option<Target> + 'a;

/// Whether the path run is a name being bound rather than a path: followed
/// by a single `:`, as a field, parameter or binding name is.
fn is_binding(src: &str, toks: &[Token], run: &PathRun) -> bool {
    toks[run.last + 1..]
        .iter()
        .find(|t| !matches!(t.kind, Tok::Space | Tok::Comment | Tok::Doc))
        .is_some_and(|t| t.kind == Tok::Punct && tok_text(src, t) == ":")
}

/// Spans for a text: one per highlighted token, and one per linked path,
/// covering the whole path. `link` gives the target of a path written at a
/// byte offset, if any, told whether the path is a name being bound
/// ([`is_binding`]).
fn spans_for(src: &str, toks: &[Token], link: &LinkFn) -> Vec<Span> {
    let lines = Lines::new(src);
    let mut linked: BTreeMap<usize, (PathRun, Target)> = BTreeMap::new();
    for run in rustlex::paths(src, toks) {
        let binding = is_binding(src, toks, &run);
        let (a, b) = (toks[run.first].start, toks[run.last].end);
        if let Some(t) = link(a, &src[a..b], binding) {
            linked.insert(run.first, (run, t));
        }
    }
    let mut out = Vec::new();
    let mut k = 0;
    while k < toks.len() {
        if let Some((run, t)) = linked.get(&k) {
            let range = (toks[run.first].start, toks[run.last].end);
            push_span(
                &mut out,
                src,
                &lines,
                range,
                SpanClass::Type,
                Some(t.clone()),
            );
            k = run.last + 1;
            continue;
        }
        let t = &toks[k];
        if let Some(c) = class_of(t.kind) {
            push_span(&mut out, src, &lines, (t.start, t.end), c, None);
        }
        k += 1;
    }
    out
}

/// Where the variants' names are written in an enum's declaration: a path at
/// the top level of the enum's braces that follows the opening brace, a
/// comma, or an attribute's closing bracket.
fn variant_name_offsets(src: &str, toks: &[Token]) -> BTreeSet<usize> {
    let mut out = BTreeSet::new();
    let mut depth = 0i32;
    let mut prev: Option<&str> = None;
    for t in toks {
        if matches!(t.kind, Tok::Space | Tok::Comment | Tok::Doc) {
            continue;
        }
        let s = tok_text(src, t);
        if t.kind == Tok::Ident && depth == 1 && matches!(prev, Some("{" | "," | "]")) {
            out.insert(t.start);
        }
        if t.kind == Tok::Punct {
            match s {
                "{" | "(" | "[" => depth += 1,
                "}" | ")" | "]" => depth -= 1,
                _ => {}
            }
        }
        prev = Some(s);
    }
    out
}

/// A region of a source file whose paths are resolved by one item's
/// signature.
struct Region {
    start: usize,
    end: usize,
    paths: Rc<PathMap>,
}

/// Builds the declarations and source spans of one crate.
struct CrateExtract<'a> {
    r: Resolver<'a>,
    texts: &'a BTreeMap<String, Text>,
}

impl CrateExtract<'_> {
    fn krate(&self) -> &Crate {
        &self.r.c.krate
    }

    fn item(&self, id: &Id) -> Option<&Item> {
        self.krate().index.get(id)
    }

    /// An item's text and byte range in its file.
    fn located(&self, item: &Item) -> Option<(&Text, usize, usize, String)> {
        let span = item.span.as_ref()?;
        let path = repo_path(self.r.prefix, &span.filename)?;
        let text = self.texts.get(&path)?;
        let (a, b) = text.range(span)?;
        Some((text, a, b, path))
    }

    /// An item's source with comments removed, dedented to its first line.
    fn item_text(&self, item: &Item) -> Option<String> {
        let (text, a, b, _) = self.located(item)?;
        let line_start = text.lines.start(text.lines.at(a).0)?;
        let indent = &text.src[line_start..a];
        let indent = if indent.trim().is_empty() {
            indent.len()
        } else {
            0
        };
        Some(strip_comments(&text.src[a..b], indent))
    }

    /// The type of a field, from its source: what follows the `:` of a named
    /// field, or a positional field without its visibility.
    fn field_type(&self, item: &Item) -> String {
        let Some(text) = self.item_text(item) else {
            return String::new();
        };
        let toks = rustlex::lex(&text);
        let colon = toks
            .iter()
            .find(|t| t.kind == Tok::Punct && tok_text(&text, t) == ":");
        let rest = match colon {
            Some(c) => &text[c.end..],
            None => {
                let mut rest = text.trim_start();
                if let Some(r) = rest.strip_prefix("pub") {
                    rest = r.trim_start();
                    if rest.starts_with('(') {
                        rest = rest.find(')').map_or(rest, |k| &rest[k + 1..]);
                    }
                }
                rest
            }
        };
        joined(rest.trim_end().trim_end_matches(','))
    }

    /// The members of a type: its fields, variants or trait items.
    fn members(&self, item: &Item) -> Vec<Member> {
        let member = |id: &Id, ty: String| -> Option<Member> {
            let it = self.item(id)?;
            Some(Member {
                name: it.name.clone().unwrap_or_default(),
                ty,
                mutable: false,
                doc: it.docs.clone(),
                doc_links: self.r.doc_links(it),
            })
        };
        let fields = |ids: &mut dyn Iterator<Item = &Id>| -> Vec<Member> {
            ids.filter_map(|id| member(id, self.item(id).map(|i| self.field_type(i))?))
                .collect()
        };
        match &item.inner {
            ItemEnum::Struct(s) => match &s.kind {
                StructKind::Plain { fields: f, .. } => fields(&mut f.iter()),
                StructKind::Tuple(f) => fields(&mut f.iter().flatten()),
                StructKind::Unit => Vec::new(),
            },
            ItemEnum::Union(u) => fields(&mut u.fields.iter()),
            ItemEnum::Enum(e) => e
                .variants
                .iter()
                .filter_map(|v| {
                    let it = self.item(v)?;
                    let name = it.name.clone().unwrap_or_default();
                    let text = joined(&self.item_text(it).unwrap_or_default());
                    let rest = text.strip_prefix(name.as_str()).unwrap_or("").trim();
                    let ty = if rest.starts_with('=') {
                        String::new()
                    } else {
                        rest.trim_end_matches(',').trim_end().to_string()
                    };
                    member(v, ty)
                })
                .collect(),
            ItemEnum::Trait(t) => t
                .items
                .iter()
                .filter_map(|i| {
                    let it = self.item(i)?;
                    let mut sig = elide_method_bodies(&format!(
                        "{{{}}}",
                        self.item_text(it).unwrap_or_default()
                    ));
                    sig = sig[1..sig.len() - 1].to_string();
                    let sig = joined(&sig);
                    let sig = sig.strip_suffix(" { … }").unwrap_or(&sig);
                    let sig = sig.strip_suffix(';').unwrap_or(sig);
                    member(i, sig.to_string())
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    /// The traits a type implements, other than auto traits, blanket impls
    /// and the compiler's marker for a derived `PartialEq`
    /// (`StructuralPartialEq`), which `#[derive(PartialEq)]` already says.
    fn impls(&self, item: &Item) -> Vec<Impl> {
        let ids: &[Id] = match &item.inner {
            ItemEnum::Struct(s) => &s.impls,
            ItemEnum::Enum(e) => &e.impls,
            ItemEnum::Union(u) => &u.impls,
            _ => &[],
        };
        let mut out = Vec::new();
        for id in ids {
            let Some(it) = self.item(id) else { continue };
            let ItemEnum::Impl(imp) = &it.inner else {
                continue;
            };
            let Some(tr) = &imp.trait_ else { continue };
            if imp.is_synthetic || imp.blanket_impl.is_some() {
                continue;
            }
            let derived = it.attrs.contains(&Attribute::AutomaticallyDerived);
            let last = tr.path.rsplit("::").next().unwrap_or(&tr.path).to_string();
            if last == "StructuralPartialEq" {
                continue;
            }
            let (trait_path, path, line) = if derived {
                (last, None, None)
            } else {
                let written = self.impl_trait_text(it).unwrap_or(last);
                match (self.located(it), &it.span) {
                    (Some((_, _, _, p)), Some(s)) => (written, Some(p), Some(s.begin.0 as u32)),
                    _ => (written, None, None),
                }
            };
            out.push(Impl {
                trait_path,
                target: self.r.target(&tr.id),
                derived,
                path,
                line,
            });
        }
        out
    }

    /// The trait an impl names, as written between `impl<...>` and `for`.
    fn impl_trait_text(&self, item: &Item) -> Option<String> {
        let (text, a, b, _) = self.located(item)?;
        let toks = text.toks_in(a, b);
        let src = &text.src;
        let body = first_at_depth0(src, toks, &[]).unwrap_or(b);
        let toks: Vec<&Token> = toks
            .iter()
            .filter(|t| t.start < body && !matches!(t.kind, Tok::Space | Tok::Comment | Tok::Doc))
            .collect();
        let mut k = toks.iter().position(|t| tok_text(src, t) == "impl")? + 1;
        // Skip the impl's own generic parameters.
        if toks.get(k).is_some_and(|t| tok_text(src, t) == "<") {
            let mut depth = 0;
            while k < toks.len() {
                match tok_text(src, toks[k]) {
                    "<" => depth += 1,
                    ">" if tok_text(src, toks[k - 1]) != "-" => depth -= 1,
                    _ => {}
                }
                k += 1;
                if depth == 0 {
                    break;
                }
            }
        }
        let start = toks.get(k)?.start;
        let mut depth = 0;
        for j in k..toks.len() {
            match tok_text(src, toks[j]) {
                "<" => depth += 1,
                ">" if tok_text(src, toks[j - 1]) != "-" => depth -= 1,
                "for" if depth == 0 && toks[j].kind == Tok::Keyword => {
                    return Some(joined(&src[start..toks[j].start]));
                }
                _ => {}
            }
        }
        None
    }

    /// A type's declaration.
    fn decl(&self, id: &Id) -> Result<TypeDecl, String> {
        let item = &self.krate().index[id];
        let name = item.name.clone().unwrap_or_default();
        let (module, _) = &self.r.c.type_at[id];
        let at = || format!("{module}::{name}");
        let span = item
            .span
            .as_ref()
            .ok_or_else(|| format!("{}: no span", at()))?;
        let mut text = self
            .item_text(item)
            .ok_or_else(|| format!("{}: its span does not fit its file", at()))?;
        let (kind, generics) = match &item.inner {
            ItemEnum::Struct(s) => (
                match s.kind {
                    StructKind::Tuple(_) => Kind::TupleStruct,
                    StructKind::Plain { .. } | StructKind::Unit => Kind::Struct,
                },
                &s.generics,
            ),
            ItemEnum::Union(u) => (Kind::Struct, &u.generics),
            ItemEnum::Enum(e) => (Kind::Enum, &e.generics),
            ItemEnum::Trait(t) => (Kind::Trait, &t.generics),
            ItemEnum::TypeAlias(t) => (Kind::Alias, &t.generics),
            _ => unreachable!("only type items are walked"),
        };
        if kind == Kind::Trait {
            text = elide_method_bodies(&text);
        }
        let params = generics
            .params
            .iter()
            .filter(|p| match &p.kind {
                GenericParamDefKind::Type { is_synthetic, .. } => !is_synthetic,
                _ => true,
            })
            .map(|p| p.name.clone())
            .collect();
        let paths = PathMap::of_item(self.krate(), item);
        let toks = rustlex::lex(&text);
        let own = self.r.target(id);
        let name_tok = toks
            .iter()
            .position(|t| t.kind == Tok::Ident && tok_text(&text, t) == name);
        let name_at = name_tok.map(|k| toks[k].start);
        let variant_names = if kind == Kind::Enum {
            variant_name_offsets(&text, &toks)
        } else {
            BTreeSet::new()
        };
        let decl_spans = spans_for(&text, &toks, &|at, written, binding| {
            if Some(at) == name_at && written == name {
                return own.clone();
            }
            if binding || variant_names.contains(&at) {
                return None;
            }
            self.r.target(&paths.get(written)?)
        });
        let mut refs: Vec<String> = Vec::new();
        for run in rustlex::paths(&text, &toks) {
            let at = toks[run.first].start;
            if is_binding(&text, &toks, &run) || variant_names.contains(&at) {
                continue;
            }
            let w = &text[at..toks[run.last].end];
            if let Some(pid) = paths.get(w) {
                if pid != *id && !refs.iter().any(|r| r == w) {
                    refs.push(w.to_string());
                }
            }
        }
        Ok(TypeDecl {
            name,
            line: span.begin.0 as u32,
            kind,
            params,
            recursive_group: None,
            decl: text,
            decl_spans,
            doc: item.docs.clone(),
            members: self.members(item),
            refs,
            doc_links: self.r.doc_links(item),
            impls: self.impls(item),
        })
    }

    /// The markup of one source file: every highlighted token, every type's
    /// name where it is declared, and every path in a signature that rustdoc
    /// resolved to a documented type.
    fn source(&self, path: &str) -> SourceFile {
        let text = &self.texts[path];
        let src = text.src.as_str();
        let mut regions: Vec<Region> = Vec::new();
        let mut bodies: Vec<(usize, usize)> = Vec::new();
        let mut names: HashMap<usize, Target> = HashMap::new();
        // Each `use` of a documented type: its range, the type's name, and
        // the type.
        let mut uses: Vec<(usize, usize, String, Target)> = Vec::new();
        // Where a variant's name is written.
        let mut variant_names: BTreeSet<usize> = BTreeSet::new();
        // The generic parameters of the impl or trait an item is in.
        let mut outer: HashMap<Id, &Generics> = HashMap::new();
        for item in self.krate().index.values() {
            match &item.inner {
                ItemEnum::Impl(i) => i.items.iter().for_each(|c| {
                    outer.insert(*c, &i.generics);
                }),
                ItemEnum::Trait(t) => t.items.iter().for_each(|c| {
                    outer.insert(*c, &t.generics);
                }),
                _ => {}
            }
        }
        for item in self.krate().index.values() {
            if item.crate_id != 0 {
                continue;
            }
            let Some((t, a, b, p)) = self.located(item) else {
                continue;
            };
            if p != path {
                continue;
            }
            let toks = t.toks_in(a, b);
            let region = |end: usize| {
                let mut paths = PathMap::of_item(self.krate(), item);
                if let Some(g) = outer.get(&item.id) {
                    paths.generic_names(g);
                }
                Region {
                    start: a,
                    end,
                    paths: Rc::new(paths),
                }
            };
            match &item.inner {
                ItemEnum::Struct(_)
                | ItemEnum::Enum(_)
                | ItemEnum::Union(_)
                | ItemEnum::Trait(_)
                | ItemEnum::TypeAlias(_) => {
                    regions.push(region(b));
                    if let (Some(target), Some(name)) = (self.r.target(&item.id), &item.name) {
                        if let Some(tok) = toks
                            .iter()
                            .find(|x| x.kind == Tok::Ident && tok_text(src, x) == name)
                        {
                            names.insert(tok.start, target);
                        }
                    }
                }
                ItemEnum::Function(_) => match first_at_depth0(src, toks, &[]) {
                    Some(open) => {
                        regions.push(region(open));
                        bodies.push((open, b));
                    }
                    None => regions.push(region(b)),
                },
                ItemEnum::Impl(imp) => {
                    if imp.is_synthetic
                        || imp.blanket_impl.is_some()
                        || item.attrs.contains(&Attribute::AutomaticallyDerived)
                    {
                        continue;
                    }
                    regions.push(region(first_at_depth0(src, toks, &[]).unwrap_or(b)));
                }
                ItemEnum::AssocType { .. } => regions.push(region(b)),
                ItemEnum::Constant { .. } | ItemEnum::AssocConst { .. } | ItemEnum::Static(_) => {
                    regions.push(region(first_at_depth0(src, toks, &["="]).unwrap_or(b)))
                }
                ItemEnum::Use(u) => {
                    let Some(target) = u.id.as_ref().and_then(|i| self.r.target(i)) else {
                        continue;
                    };
                    if u.is_glob {
                        continue;
                    }
                    let last = u.source.rsplit("::").next().unwrap_or(&u.source);
                    uses.push((a, b, last.to_string(), target));
                }
                ItemEnum::Variant(_) => {
                    if let Some(name) = &item.name {
                        if let Some(tok) = toks
                            .iter()
                            .find(|x| x.kind == Tok::Ident && tok_text(src, x) == name)
                        {
                            variant_names.insert(tok.start);
                        }
                    }
                }
                _ => {}
            }
        }
        let link = |at: usize, written: &str, binding: bool| -> Option<Target> {
            if let Some(t) = names.get(&at) {
                if !written.contains("::") {
                    return Some(t.clone());
                }
            }
            let last = written.rsplit("::").next().unwrap_or(written);
            if let Some((_, _, _, t)) = uses
                .iter()
                .find(|(a, b, name, _)| *a < at + written.len() && at < *b && name == last)
            {
                return Some(t.clone());
            }
            if binding || variant_names.contains(&at) {
                return None;
            }
            let r = regions
                .iter()
                .filter(|r| r.start <= at && at < r.end)
                .min_by_key(|r| r.end - r.start)?;
            let in_body = bodies
                .iter()
                .any(|&(s, e)| s <= at && at < e && r.start <= s && e <= r.end);
            if in_body {
                return None;
            }
            self.r.target(&r.paths.get(written)?)
        };
        SourceFile {
            path: path.to_string(),
            lines: src.lines().count(),
            bytes: src.len(),
            spans: spans_for(src, &text.toks, &link),
        }
    }
}

/// Extracts every crate of `specs`, in order, into one input. `prefix` is
/// the cargo workspace's directory relative to the repository root (`rust`),
/// and `repo_root` the repository root the source files are read from.
pub fn extract(specs: &[CrateSpec], prefix: &str, repo_root: &Path) -> Result<Input, String> {
    let mut loaded = Vec::new();
    for spec in specs {
        let krate = load(spec)?;
        loaded.push(walk(&spec.label, krate, prefix)?);
    }
    let mut world = World {
        types: BTreeSet::new(),
        modules: BTreeSet::new(),
        labels: HashMap::new(),
        files: BTreeSet::new(),
    };
    for c in &loaded {
        let root_name = c.krate.index[&c.krate.root]
            .name
            .clone()
            .unwrap_or_default();
        if world
            .labels
            .insert(root_name.clone(), c.label.clone())
            .is_some()
        {
            return Err(format!("crate {root_name} given twice"));
        }
        for m in &c.mods {
            world.modules.insert(m.name.clone());
            world.files.insert(m.file.clone());
        }
        for (module, name) in c.type_at.values() {
            world.types.insert((module.clone(), name.clone()));
        }
    }
    let mut texts = BTreeMap::new();
    for f in &world.files {
        let src = fs::read_to_string(repo_root.join(f)).map_err(|e| format!("{f}: {e}"))?;
        texts.insert(f.clone(), Text::new(src));
    }

    let mut modules = Vec::new();
    let mut sources = Vec::new();
    let mut seen_files = BTreeSet::new();
    for c in &loaded {
        let x = CrateExtract {
            r: Resolver {
                c,
                world: &world,
                prefix,
            },
            texts: &texts,
        };
        for m in &c.mods {
            let item = &c.krate.index[&m.id];
            let types = m
                .types
                .iter()
                .map(|t| x.decl(t))
                .collect::<Result<Vec<_>, _>>()?;
            modules.push(Module {
                name: m.name.clone(),
                path: m.file.clone(),
                doc: item.docs.clone(),
                doc_links: x.r.doc_links(item),
                types,
            });
            if seen_files.insert(m.file.clone()) {
                sources.push(x.source(&m.file));
            }
        }
    }
    Ok(Input {
        language: Language::Rust,
        modules,
        sources,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comments_are_stripped_and_lines_dedented() {
        let src = "struct A {\n        /// The x.\n        x: u32, // trailing\n        /* gone */\n        y: f64,\n    }";
        assert_eq!(
            strip_comments(src, 4),
            "struct A {\n    x: u32,\n    y: f64,\n}"
        );
    }

    #[test]
    fn provided_method_bodies_are_elided() {
        let src = "trait T {\n    fn a(&self) -> u32;\n    fn b(&self) -> u32 { if x { 1 } else { 2 } }\n}";
        assert_eq!(
            elide_method_bodies(src),
            "trait T {\n    fn a(&self) -> u32;\n    fn b(&self) -> u32 { … }\n}"
        );
    }

    #[test]
    fn a_foreign_format_version_is_refused() {
        let e = check_format_version(r#"{"format_version": 1, "root": 0}"#).unwrap_err();
        assert!(e.contains("format_version 1 is not 57"), "{e}");
        assert!(e.contains("rustdoc-types"), "{e}");
        assert!(check_format_version(r#"{"format_version": 57}"#).is_ok());
    }
}
