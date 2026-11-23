//! The site: which page each module is shown on, the anchor of each type, how
//! a type path written in one module is resolved to a declaration, and which
//! types refer to each type.

use std::collections::{BTreeMap, BTreeSet};

use crate::model::{Input, Language, Module, SourceFile, Target, Value};
use crate::overview::Overview;

/// The file name, without extension, of the index page. No module page may
/// take it.
pub const INDEX_SLUG: &str = "index";

/// Types every OCaml program can name without defining them. A reference to
/// one that no input module defines is not listed under "Refers to", because
/// it says nothing about how the documented modules fit together.
const OCAML_PREDEFINED: &[&str] = &[
    "int",
    "char",
    "string",
    "bytes",
    "float",
    "bool",
    "unit",
    "exn",
    "array",
    "list",
    "option",
    "result",
    "int32",
    "int64",
    "nativeint",
    "format",
    "format4",
    "format6",
    "lazy_t",
    "ref",
    "Seq.t",
    "Hashtbl.t",
];

/// Types and primitives every Rust program can name without defining them,
/// left out of "Refers to" for the reason given at [`OCAML_PREDEFINED`].
const RUST_PREDEFINED: &[&str] = &[
    "bool", "char", "str", "String", "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16",
    "i32", "i64", "i128", "isize", "f32", "f64", "Vec", "Option", "Result", "Box", "Rc", "Arc",
    "HashMap", "BTreeMap", "HashSet", "BTreeSet", "Self",
];

/// A type declaration, by its module's position in [`Site::modules`] and its
/// position in that module's `types`. Ordering follows the site: page order,
/// then source order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TypeId {
    pub module: usize,
    pub ty: usize,
}

/// The separator of a module path: `::` when the path has one (a Rust
/// path), `.` otherwise.
fn sep_of(path: &str) -> &'static str {
    if path.contains("::") {
        "::"
    } else {
        "."
    }
}

/// A module placed on a page.
#[derive(Debug)]
pub struct ModuleEntry<'a> {
    pub module: &'a Module,
    pub language: Language,
    /// The page's position in [`Site::pages`].
    pub page: usize,
    /// The module's path below its page's top-level module: empty for the
    /// top-level module itself, `Located` for `Ast.Located`.
    pub local: String,
    /// The anchor of each type, parallel to `module.types`, unique on the page.
    pub anchors: Vec<String>,
}

/// One generated module page: a top-level module and the modules nested in
/// it. For Rust, a page per source file: the crate's root file is listed by
/// the overview, and every other file is a page under it.
#[derive(Debug)]
pub struct Page {
    /// The top-level module's name: as the overview lists it, or for a Rust
    /// file that is not its crate's root, the file's module (`sim::pgas`).
    pub name: String,
    /// For a Rust file that is not its crate's root, the crate's page, which
    /// lists it; the index lists only pages without one.
    pub parent: Option<usize>,
    /// The file name without extension.
    pub slug: String,
    /// The overview group the page is listed under.
    pub group: usize,
    /// The sidebar position.
    pub order: i64,
    /// The modules shown, as positions in [`Site::modules`]: the top-level
    /// module first when the input has one, then nested modules in input order.
    pub modules: Vec<usize>,
}

/// Every module of every input, placed on pages and cross-referenced.
#[derive(Debug)]
pub struct Site<'a> {
    pub overview: &'a Overview,
    /// The sidebar position of the index page. The module pages follow it, at
    /// `order + 1`, `order + 2`, ... in overview order.
    pub order: i64,
    pub modules: Vec<ModuleEntry<'a>>,
    pub pages: Vec<Page>,
    /// The source files, one page each, in input order.
    pub sources: Vec<SourceEntry<'a>>,
    by_name: BTreeMap<&'a str, usize>,
    defining: BTreeMap<&'a str, Vec<usize>>,
    used_by: BTreeMap<TypeId, Vec<TypeId>>,
    implemented_by: BTreeMap<TypeId, Vec<TypeId>>,
    source_by_path: BTreeMap<&'a str, usize>,
}

/// The directory, below the output directory, the source pages are written
/// to. They are reached by links and are not in the sidebar.
pub const SOURCE_DIR: &str = "src";

/// A source file shown on a page of its own.
#[derive(Debug)]
pub struct SourceEntry<'a> {
    pub file: &'a SourceFile,
    /// The file name of its page, without extension: the file's name without
    /// its extension, unique among the sources.
    pub slug: String,
    /// The page of the modules the file defines.
    pub page: usize,
}

/// Where a link is written from. A Markdown link names the `.qmd` file,
/// which Quarto rewrites; a link in raw HTML, which Quarto does not rewrite,
/// names the rendered `.html` file.
#[derive(Debug, Clone, Copy)]
pub enum From<'s> {
    /// A module page, by its position in [`Site::pages`].
    Page(usize),
    /// The index page, in the module pages' directory.
    Index,
    /// A source page, one directory below the module pages.
    Source,
    /// A page outside the site's directory, such as an engineering page:
    /// the site's directory as a path relative to that page's, `../runtime`.
    Outside(&'s str),
}

/// What a full path written outside the code names: the destinations
/// [`Site::named`] gives the references of an engineering page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Named<'a> {
    /// A documented type.
    Type(TypeId),
    /// A field, constructor, variant or trait item of a documented type, by
    /// its position in the type's members. Members have no anchor of their
    /// own, so a link to one goes to its type.
    Member(TypeId, usize),
    /// A function, constant, static or method the pages do not document,
    /// shown at its line on its source file's page.
    Value(&'a Value),
    /// A module.
    Module(usize),
}

/// The top-level module a module path belongs to: its first segment.
fn top_of(name: &str) -> &str {
    name.split(sep_of(name)).next().unwrap_or(name)
}

/// The file name of a module page: the module name in lower case, with each
/// separator (`.` or `::`) as a hyphen.
pub fn slug_of(name: &str) -> String {
    name.to_lowercase().replace("::", "-").replace('.', "-")
}

/// An HTML id made from a path: characters other than ASCII letters, digits,
/// `_` and `-` become `-`.
fn sanitize_id(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect()
}

impl<'a> Site<'a> {
    /// Places every module of `inputs` on the page the overview gives it, the
    /// index page at sidebar position `order`.
    ///
    /// Fails, naming the modules, when an input module's top-level module is
    /// in no `{!modules: ...}` list of the overview, when a listed name has no
    /// input module, when a name occurs twice in the lists or in the inputs,
    /// or when a module page would overwrite the index page.
    pub fn build(
        overview: &'a Overview,
        order: i64,
        inputs: &'a [Input],
    ) -> Result<Site<'a>, String> {
        let all: Vec<(&'a Module, Language)> = inputs
            .iter()
            .flat_map(|i| i.modules.iter().map(move |m| (m, i.language)))
            .collect();

        let mut seen = BTreeSet::new();
        let dup_inputs: BTreeSet<&str> = all
            .iter()
            .filter(|(m, _)| !seen.insert(m.name.as_str()))
            .map(|(m, _)| m.name.as_str())
            .collect();
        if !dup_inputs.is_empty() {
            return Err(format!(
                "modules defined more than once in the input: {}",
                join(&dup_inputs)
            ));
        }

        let mut listed = BTreeSet::new();
        let dup_listed: BTreeSet<&str> = overview
            .module_names()
            .filter(|n| !listed.insert(*n))
            .collect();
        if !dup_listed.is_empty() {
            return Err(format!(
                "modules listed more than once in the overview: {}",
                join(&dup_listed)
            ));
        }
        let nested_listed: BTreeSet<&str> = listed
            .iter()
            .copied()
            .filter(|n| n.contains('.') || n.contains("::"))
            .collect();
        if !nested_listed.is_empty() {
            return Err(format!(
                "the overview lists nested modules, which are shown on their parent's page; \
                 list the parent instead: {}",
                join(&nested_listed)
            ));
        }

        let present: BTreeSet<&str> = all.iter().map(|(m, _)| top_of(&m.name)).collect();
        let unlisted: BTreeSet<&str> = all
            .iter()
            .map(|(m, _)| m.name.as_str())
            .filter(|n| !listed.contains(top_of(n)))
            .collect();
        let missing: BTreeSet<&str> = listed.difference(&present).copied().collect();
        if !unlisted.is_empty() || !missing.is_empty() {
            let mut msg = Vec::new();
            if !unlisted.is_empty() {
                msg.push(format!(
                    "input modules in no {{!modules: ...}} list of the overview: {}",
                    join(&unlisted)
                ));
            }
            if !missing.is_empty() {
                msg.push(format!(
                    "overview modules absent from the input: {}",
                    join(&missing)
                ));
            }
            return Err(msg.join("; "));
        }

        let mut modules = Vec::new();
        let mut pages = Vec::new();
        for (gi, group) in overview.groups.iter().enumerate() {
            for listed_name in &group.modules {
                let mut members: Vec<(&'a Module, Language)> = all
                    .iter()
                    .copied()
                    .filter(|(m, _)| top_of(&m.name) == listed_name)
                    .collect();
                // The top-level module first; nested ones keep input order.
                members.sort_by_key(|(m, _)| m.name != *listed_name);
                // The pages: one for an OCaml module, one per file for a Rust
                // crate, the crate root's file first.
                let mut files: Vec<(String, Vec<(&'a Module, Language)>)> = Vec::new();
                for (m, language) in members {
                    let key = match language {
                        Language::Ocaml => String::new(),
                        Language::Rust => m.path.clone(),
                    };
                    match files.iter_mut().find(|(k, _)| *k == key) {
                        Some((_, v)) => v.push((m, language)),
                        None => files.push((key, vec![(m, language)])),
                    }
                }
                let crate_page = pages.len();
                for (fi, (_, members)) in files.into_iter().enumerate() {
                    let name: String = if fi == 0 {
                        listed_name.clone()
                    } else {
                        // The file's module: the shortest path among its modules.
                        members
                            .iter()
                            .map(|(m, _)| m.name.as_str())
                            .min_by_key(|n| n.len())
                            .unwrap_or_default()
                            .to_string()
                    };
                    let name = &name;
                    let slug = slug_of(name);
                    if slug == INDEX_SLUG {
                        return Err(format!(
                            "module {name} would be written to {INDEX_SLUG}.qmd, the index page"
                        ));
                    }
                    let page = pages.len();
                    let mut used_anchors = BTreeSet::new();
                    let mut indices = Vec::new();
                    for (m, language) in members {
                        let sep = language.separator();
                        let local = m
                            .name
                            .strip_prefix(name.as_str())
                            .and_then(|r| r.strip_prefix(sep))
                            .unwrap_or("")
                            .to_string();
                        let anchors = m
                            .types
                            .iter()
                            .map(|t| {
                                let path = if local.is_empty() {
                                    t.name.clone()
                                } else {
                                    format!("{local}{sep}{}", t.name)
                                };
                                let base = format!("t-{}", sanitize_id(&path));
                                let mut anchor = base.clone();
                                let mut n = 2;
                                while !used_anchors.insert(anchor.clone()) {
                                    anchor = format!("{base}-{n}");
                                    n += 1;
                                }
                                anchor
                            })
                            .collect();
                        indices.push(modules.len());
                        modules.push(ModuleEntry {
                            module: m,
                            language,
                            page,
                            local,
                            anchors,
                        });
                    }
                    pages.push(Page {
                        name: name.clone(),
                        parent: (fi > 0).then_some(crate_page),
                        slug,
                        group: gi,
                        order: order + 1 + page as i64,
                        modules: indices,
                    });
                }
            }
        }

        let by_name = modules
            .iter()
            .enumerate()
            .map(|(i, e)| (e.module.name.as_str(), i))
            .collect();
        let mut defining: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
        for (i, e) in modules.iter().enumerate() {
            let names: BTreeSet<&str> = e.module.types.iter().map(|t| t.name.as_str()).collect();
            for n in names {
                defining.entry(n).or_default().push(i);
            }
        }
        let (sources, source_by_path) = place_sources(inputs, &modules, &pages)?;
        let mut site = Site {
            overview,
            order,
            modules,
            pages,
            sources,
            by_name,
            defining,
            used_by: BTreeMap::new(),
            implemented_by: BTreeMap::new(),
            source_by_path,
        };
        site.check_targets()?;
        let mut implemented_by: BTreeMap<TypeId, Vec<TypeId>> = BTreeMap::new();
        for (mi, e) in site.modules.iter().enumerate() {
            for (ti, t) in e.module.types.iter().enumerate() {
                for imp in &t.impls {
                    if let Some(tr) = imp.target.as_ref().and_then(|x| site.find_target(x)) {
                        let users = implemented_by.entry(tr).or_default();
                        let me = TypeId { module: mi, ty: ti };
                        if !users.contains(&me) {
                            users.push(me);
                        }
                    }
                }
            }
        }
        site.implemented_by = implemented_by;

        let mut used_by: BTreeMap<TypeId, Vec<TypeId>> = BTreeMap::new();
        for (mi, e) in site.modules.iter().enumerate() {
            for ti in 0..e.module.types.len() {
                let me = TypeId { module: mi, ty: ti };
                for target in site.refers_to(me) {
                    let users = used_by.entry(target).or_default();
                    if users.last() != Some(&me) {
                        users.push(me);
                    }
                }
            }
        }
        site.used_by = used_by;
        Ok(site)
    }

    /// The modules a name written in module `from` is looked up in, innermost
    /// first: `from` itself, then each enclosing module.
    fn scopes(&self, from: usize) -> Vec<&'a str> {
        let name = self.modules[from].module.name.as_str();
        let sep = self.modules[from].language.separator();
        let mut out = vec![name];
        let mut rest = name;
        while let Some(i) = rest.rfind(sep) {
            rest = &rest[..i];
            out.push(rest);
        }
        out
    }

    /// The last declaration of `name` in module `m` (a later OCaml declaration
    /// shadows an earlier one of the same name).
    fn find_in(&self, m: usize, name: &str) -> Option<TypeId> {
        self.modules[m]
            .module
            .types
            .iter()
            .rposition(|t| t.name == name)
            .map(|ty| TypeId { module: m, ty })
    }

    /// The module a module path written in module `from` names: looked up in
    /// each enclosing scope, innermost first, then with leading path segments
    /// dropped (a library's wrapper module, as in `Camdl.Ast`).
    pub fn resolve_module(&self, from: usize, path: &str) -> Option<usize> {
        let sep = self.modules[from].language.separator();
        for scope in self.scopes(from) {
            if let Some(&m) = self.by_name.get(format!("{scope}{sep}{path}").as_str()) {
                return Some(m);
            }
        }
        self.resolve_module_global(path)
    }

    /// The module a full module path names, outside any module: the path
    /// itself, then with leading segments dropped (a library's wrapper
    /// module, as in `Camdl.Ast`).
    fn resolve_module_global(&self, path: &str) -> Option<usize> {
        if let Some(&m) = self.by_name.get(path) {
            return Some(m);
        }
        let sep = sep_of(path);
        let segs: Vec<&str> = path.split(sep).collect();
        (1..segs.len()).find_map(|k| self.by_name.get(segs[k..].join(sep).as_str()).copied())
    }

    /// The declaration a type path written in module `from` names, if any
    /// input declares it.
    ///
    /// A qualified path `M.t` names `t` in the module [`Site::resolve_module`]
    /// finds for `M`. An unqualified `t` names the declaration in `from` or
    /// the nearest enclosing module, and otherwise the declaration in the one
    /// module of all the inputs that declares a type `t`; when several do, it
    /// is unresolved.
    pub fn resolve_type(&self, from: usize, path: &str) -> Option<TypeId> {
        let path = path.trim();
        match path.rsplit_once(self.modules[from].language.separator()) {
            Some((qual, name)) => self
                .resolve_module(from, qual)
                .and_then(|m| self.find_in(m, name)),
            None => {
                for scope in self.scopes(from) {
                    if let Some(found) =
                        self.by_name.get(scope).and_then(|&m| self.find_in(m, path))
                    {
                        return Some(found);
                    }
                }
                self.declared_once(path)
            }
        }
    }

    /// The declaration of `name` when exactly one module of the inputs
    /// declares a type of that name.
    fn declared_once(&self, name: &str) -> Option<TypeId> {
        match self.defining.get(name).map(Vec::as_slice) {
            Some([only]) => self.find_in(*only, name),
            _ => None,
        }
    }

    /// The link target, from the index page, of a reference written in the
    /// overview: a type path or a module path, written in full since the
    /// overview is in no module. `None` when it names nothing in the inputs.
    pub fn index_href(&self, path: &str) -> Option<String> {
        let path = path.trim();
        let ty = match path.rsplit_once(sep_of(path)) {
            Some((qual, name)) => self
                .resolve_module_global(qual)
                .and_then(|m| self.find_in(m, name)),
            None => self.declared_once(path),
        };
        match ty {
            Some(id) => Some(self.type_href(From::Index, id)),
            None => self
                .resolve_module_global(path)
                .map(|m| self.module_href(From::Index, m)),
        }
    }

    /// Whether a path that resolves to nothing names a type the language
    /// itself provides.
    pub fn is_predefined(&self, from: usize, path: &str) -> bool {
        let list = match self.modules[from].language {
            Language::Ocaml => OCAML_PREDEFINED,
            Language::Rust => RUST_PREDEFINED,
        };
        let path = path.strip_prefix("Stdlib.").unwrap_or(path);
        list.contains(&path)
    }

    /// The types that implement the trait `id`, in site order: those whose
    /// [`crate::model::TypeDecl::impls`] name it.
    pub fn implemented_by(&self, id: TypeId) -> &[TypeId] {
        self.implemented_by
            .get(&id)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// The module of the inputs named exactly `name`.
    pub fn module_named(&self, name: &str) -> Option<usize> {
        self.by_name.get(name).copied()
    }

    /// The source page of a file of the inputs, by its path.
    pub fn source_named(&self, path: &str) -> Option<&SourceEntry<'a>> {
        self.source_by_path.get(path).map(|&i| &self.sources[i])
    }

    /// The pages a crate's page lists: its crate's other files.
    pub fn children(&self, page: usize) -> impl Iterator<Item = usize> + '_ {
        (0..self.pages.len()).filter(move |&p| self.pages[p].parent == Some(page))
    }

    /// The types whose declarations refer to `id`, in site order.
    pub fn used_by(&self, id: TypeId) -> &[TypeId] {
        self.used_by.get(&id).map(Vec::as_slice).unwrap_or(&[])
    }

    /// The link to page `page`, at `anchor` when given, from `from`: the
    /// anchor alone on the same page, otherwise the page's file with the
    /// extension `ext`.
    fn page_href(&self, from: From, page: usize, anchor: Option<&str>, ext: &str) -> String {
        if let (From::Page(p), Some(a)) = (from, anchor) {
            if p == page {
                return format!("#{a}");
            }
        }
        let frag = anchor.map_or(String::new(), |a| format!("#{a}"));
        let file = format!("{}.{ext}{frag}", self.pages[page].slug);
        match from {
            From::Page(_) | From::Index => file,
            From::Source => format!("../{file}"),
            From::Outside(dir) => format!("{dir}/{file}"),
        }
    }

    /// The Markdown link target of a type, from `from`.
    pub fn type_href(&self, from: From, id: TypeId) -> String {
        let e = &self.modules[id.module];
        self.page_href(from, e.page, Some(&e.anchors[id.ty]), "qmd")
    }

    /// The anchor of a nested module's section on its page. The top-level
    /// module has none; its page is the target.
    pub fn module_anchor(&self, m: usize) -> Option<String> {
        let local = &self.modules[m].local;
        (!local.is_empty()).then(|| format!("m-{}", sanitize_id(local)))
    }

    /// The Markdown link target of a module, from `from`.
    pub fn module_href(&self, from: From, m: usize) -> String {
        let anchor = self.module_anchor(m);
        self.page_href(from, self.modules[m].page, anchor.as_deref(), "qmd")
    }

    /// The Markdown link target of a line of a source file of the inputs,
    /// from `from`. `None` when no input has the file.
    pub fn source_href(&self, from: From, path: &str, line: u32) -> Option<String> {
        let s = self.source_named(path)?;
        let file = format!("{SOURCE_DIR}/{}.qmd#L{line}", s.slug);
        Some(match from {
            From::Page(_) | From::Index => file,
            From::Source => format!("{}.qmd#L{line}", s.slug),
            From::Outside(dir) => format!("{dir}/{file}"),
        })
    }

    /// The declaration a span's target names: the declaration of that name
    /// in that module, the last when there are several, as for a qualified
    /// path. Exact: no scope search and no guessing.
    pub fn find_target(&self, t: &Target) -> Option<TypeId> {
        self.by_name
            .get(t.module.as_str())
            .and_then(|&m| self.find_in(m, &t.ty))
    }

    /// Fails, naming the first, when a span's target names no declaration
    /// of the inputs: the extractor emits targets only for the declarations
    /// it reports, so one that resolves to nothing is a fault, not a type
    /// to leave unlinked.
    fn check_targets(&self) -> Result<(), String> {
        let decls = self
            .modules
            .iter()
            .flat_map(|e| {
                e.module
                    .types
                    .iter()
                    .map(move |t| (e.module.path.as_str(), t))
            })
            .flat_map(|(path, t)| t.decl_spans.iter().map(move |s| (path, t.line, s)));
        let files = self.sources.iter().flat_map(|s| {
            s.file
                .spans
                .iter()
                .map(move |sp| (s.file.path.as_str(), 0, sp))
        });
        for e in &self.modules {
            let path = e.module.path.as_str();
            let mut links: Vec<(u32, &crate::model::DocLink)> =
                e.module.doc_links.iter().map(|l| (0, l)).collect();
            for t in &e.module.types {
                links.extend(t.doc_links.iter().map(|l| (t.line, l)));
                for m in &t.members {
                    links.extend(m.doc_links.iter().map(|l| (t.line, l)));
                }
                for imp in &t.impls {
                    if let Some(target) = &imp.target {
                        if self.find_target(target).is_none() {
                            return Err(format!(
                                "{path}, the declaration at line {}: an impl names trait {}.{}, \
                                 which no input declares",
                                t.line, target.module, target.ty
                            ));
                        }
                    }
                }
            }
            for (line, l) in links {
                let ok = match &l.to {
                    crate::model::LinkTo::Type { module, ty } => self
                        .find_target(&Target {
                            module: module.clone(),
                            ty: ty.clone(),
                        })
                        .is_some(),
                    crate::model::LinkTo::Module { module } => {
                        self.by_name.contains_key(module.as_str())
                    }
                    crate::model::LinkTo::Source { path, .. } => {
                        self.source_by_path.contains_key(path.as_str())
                    }
                };
                if !ok {
                    return Err(format!(
                        "{path}, line {line}: the doc link {:?} points to {:?}, which the inputs \
                         do not have",
                        l.text, l.to
                    ));
                }
            }
            for v in &e.module.values {
                if !self.source_by_path.contains_key(v.path.as_str()) {
                    return Err(format!(
                        "{path}: {}{}{} is written in {}, which is no source file of the inputs",
                        e.module.name,
                        e.language.separator(),
                        v.name,
                        v.path
                    ));
                }
            }
        }
        for (path, decl_line, s) in decls.chain(files) {
            if let Some(t) = &s.target {
                if self.find_target(t).is_none() {
                    let at = if decl_line == 0 {
                        format!("line {}", s.line)
                    } else {
                        format!("the declaration at line {decl_line}")
                    };
                    return Err(format!(
                        "{path}, {at}: a span links to {}.{}, which no input declares",
                        t.module, t.ty
                    ));
                }
            }
        }
        Ok(())
    }

    /// The declarations a type's declaration names, in the order it names
    /// them, without repeats and without the type itself: the targets of its
    /// [`crate::model::TypeDecl::decl_spans`], which the extractor resolved.
    pub fn refers_to(&self, id: TypeId) -> Vec<TypeId> {
        let t = &self.modules[id.module].module.types[id.ty];
        let mut out = Vec::new();
        for target in t.decl_spans.iter().filter_map(|s| s.target.as_ref()) {
            // Site::build checked that every target resolves.
            if let Some(found) = self.find_target(target) {
                if found != id && !out.contains(&found) {
                    out.push(found);
                }
            }
        }
        out
    }

    /// The source page of a module's file.
    pub fn source_of(&self, m: usize) -> &SourceEntry<'a> {
        // Site::build refuses a module whose file has no source entry.
        &self.sources[self.source_by_path[self.modules[m].module.path.as_str()]]
    }

    /// The link target, in rendered HTML, of a type's documentation.
    pub fn type_href_html(&self, from: From, id: TypeId) -> String {
        let e = &self.modules[id.module];
        self.page_href(from, e.page, Some(&e.anchors[id.ty]), "html")
    }

    /// A type's name as a reader of `from` should see it: qualified by its
    /// nested module on its own page, by its full module path elsewhere.
    pub fn type_display(&self, from: From, id: TypeId) -> String {
        let e = &self.modules[id.module];
        let name = &e.module.types[id.ty].name;
        let sep = e.language.separator();
        match from {
            From::Page(p) if p == e.page && e.local.is_empty() => name.clone(),
            From::Page(p) if p == e.page => format!("{}{sep}{name}", e.local),
            _ => format!("{}{sep}{name}", e.module.name),
        }
    }

    /// What a full path names, written outside any module as an engineering
    /// page's reference is: a type (`store::hash::Key`, `Ir.model`), a
    /// member of one (`store::hash::Key::to_hex`), a function, constant,
    /// static or method the extractor lists in [`Module::values`]
    /// (`ir::model_hash::model_ir_hash`, `store::hash::Key::digest_bytes`), or
    /// a module (`ir::model_hash`). Only modules in `language` are searched.
    ///
    /// Module paths are matched exactly, with neither the scope search nor
    /// the dropping of a wrapper module that [`Site::resolve_type`] does: a
    /// path written outside the code has no scope. When a path could name
    /// several things, the first of type, member, value and module is taken.
    pub fn named(&self, language: Language, path: &str) -> Option<Named<'a>> {
        let sep = language.separator();
        let module = |name: &str| {
            self.by_name
                .get(name)
                .copied()
                .filter(|&m| self.modules[m].language == language)
        };
        let qual_name = path.rsplit_once(sep);
        // `qual` split again, for a member or a method: its module and type.
        let qual_parts = qual_name.and_then(|(qual, _)| qual.rsplit_once(sep));
        if let Some((qual, name)) = qual_name {
            if let Some(id) = module(qual).and_then(|m| self.find_in(m, name)) {
                return Some(Named::Type(id));
            }
            if let Some((mq, ty)) = qual_parts {
                if let Some(id) = module(mq).and_then(|m| self.find_in(m, ty)) {
                    let t = &self.modules[id.module].module.types[id.ty];
                    if let Some(k) = t.members.iter().position(|mem| mem.name == name) {
                        return Some(Named::Member(id, k));
                    }
                }
            }
            let modules = &self.modules;
            let value_in = |m: &str, v: &str| -> Option<&'a Value> {
                let m = module(m)?;
                modules[m].module.values.iter().find(|x| x.name == v)
            };
            if let Some(v) = value_in(qual, name) {
                return Some(Named::Value(v));
            }
            if let Some((mq, ty)) = qual_parts {
                if let Some(v) = value_in(mq, &format!("{ty}{sep}{name}")) {
                    return Some(Named::Value(v));
                }
            }
        }
        module(path).map(Named::Module)
    }
}

/// Gives each source file its page. Fails when a module's file has no
/// source entry, when a file is listed twice or defines no module, or when
/// two files would share a page name.
fn place_sources<'a>(
    inputs: &'a [Input],
    modules: &[ModuleEntry<'a>],
    pages: &[Page],
) -> Result<(Vec<SourceEntry<'a>>, BTreeMap<&'a str, usize>), String> {
    let mut sources = Vec::new();
    let mut by_path = BTreeMap::new();
    let mut slugs: BTreeMap<String, &str> = BTreeMap::new();
    for file in inputs.iter().flat_map(|i| &i.sources) {
        let path = file.path.as_str();
        if by_path.contains_key(path) {
            return Err(format!("source file listed more than once: {path}"));
        }
        let Some(page) = modules
            .iter()
            .find(|e| e.module.path == path)
            .map(|e| e.page)
        else {
            return Err(format!(
                "source file defines no module of the input: {path}"
            ));
        };
        // An OCaml file is named after its module; Rust files repeat names
        // (`mod.rs`, `model.rs` in several crates), so each is named after
        // its page, which is one per file.
        let slug = match inputs_language(inputs, file) {
            Language::Ocaml => {
                let name = path.rsplit('/').next().unwrap_or(path);
                name.split('.').next().unwrap_or(name).to_string()
            }
            Language::Rust => modules
                .iter()
                .find(|e| e.module.path == path)
                .map(|e| pages[e.page].slug.clone())
                .unwrap_or_default(),
        };
        if let Some(other) = slugs.insert(slug.clone(), path) {
            return Err(format!(
                "source files {other} and {path} would both be shown on {SOURCE_DIR}/{slug}.qmd"
            ));
        }
        by_path.insert(path, sources.len());
        sources.push(SourceEntry { file, slug, page });
    }
    let missing: BTreeSet<&str> = modules
        .iter()
        .map(|e| e.module.path.as_str())
        .filter(|p| !by_path.contains_key(p))
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "module files with no source entry in the input: {}",
            join(&missing)
        ));
    }
    Ok((sources, by_path))
}

/// The language of the input a source file comes from.
fn inputs_language(inputs: &[Input], file: &SourceFile) -> Language {
    inputs
        .iter()
        .find(|i| i.sources.iter().any(|s| std::ptr::eq(s, file)))
        .map_or(Language::Ocaml, |i| i.language)
}

fn join(names: &BTreeSet<&str>) -> String {
    names.iter().copied().collect::<Vec<_>>().join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(json: &str) -> Input {
        serde_json::from_str(json).unwrap()
    }

    /// A type whose declaration mentions `refs` as written, and whose typed
    /// spans link to `targets`, each `Module.type`. Span positions are not
    /// checked by the site, only rendered.
    fn ty(name: &str, refs: &[&str], targets: &[&str]) -> String {
        let spans: Vec<String> = targets
            .iter()
            .map(|t| {
                let (m, n) = t.rsplit_once('.').unwrap();
                format!(
                    r#"{{"line":1,"col":0,"len":1,"class":"type","target":{{"module":"{m}","type":"{n}"}}}}"#
                )
            })
            .collect();
        format!(
            r#"{{"name":"{name}","line":1,"kind":"alias","params":[],"recursive_group":null,
               "decl":"type {name} = int","decl_spans":[{}],"doc":null,"members":[],"refs":{}}}"#,
            spans.join(","),
            serde_json::to_string(refs).unwrap()
        )
    }

    /// A module in the file named after its top-level module.
    fn module(name: &str, types: &[String]) -> String {
        format!(
            r#"{{"name":"{name}","path":"{}.ml","doc":null,"types":[{}]}}"#,
            top_of(name).to_lowercase(),
            types.join(",")
        )
    }

    /// An input of `modules`, with an empty source entry for each file.
    fn ocaml(modules: &[String], files: &[&str]) -> Input {
        let sources: Vec<String> = files
            .iter()
            .map(|f| format!(r#"{{"path":"{f}.ml","lines":1,"bytes":1,"spans":[]}}"#))
            .collect();
        input(&format!(
            r#"{{"language":"ocaml","modules":[{}],"sources":[{}]}}"#,
            modules.join(","),
            sources.join(",")
        ))
    }

    fn overview(groups: &[&[&str]]) -> Overview {
        let mut s = String::from("{0 T}\nSummary.\n");
        for (k, g) in groups.iter().enumerate() {
            s.push_str(&format!("{{1 G{k}}}\n{{!modules: {}}}\n", g.join(" ")));
        }
        Overview::parse(&s).unwrap()
    }

    fn sample() -> Vec<Input> {
        vec![ocaml(
            &[
                module(
                    "Ast",
                    &[
                        ty(
                            "expr",
                            &["expr", "Ir.expr", "Yojson.Safe.t", "int"],
                            &["Ast.expr", "Ir.expr"],
                        ),
                        ty("only_here", &[], &[]),
                    ],
                ),
                module(
                    "Ast.Loc",
                    &[
                        ty("t", &["expr"], &["Ast.expr"]),
                        ty("u", &["t"], &["Ast.Loc.t"]),
                    ],
                ),
                module("Ir", &[ty("expr", &[], &[]), ty("t", &[], &[])]),
                module(
                    "Util",
                    &[ty(
                        "w",
                        &["only_here", "Loc.t", "Camdl.Ir.t"],
                        &["Ast.only_here", "Ast.Loc.t", "Ir.t"],
                    )],
                ),
            ],
            &["ast", "ir", "util"],
        )]
    }

    #[test]
    fn resolves_qualified_unqualified_nested_and_wrapped_paths() {
        let inputs = sample();
        let m = overview(&[&["Ast", "Ir"], &["Util"]]);
        let site = Site::build(&m, 10, &inputs).unwrap();
        let ast = 0;
        let loc = 1;
        let ir = 2;
        let util = 3;
        let id = |module, ty| Some(TypeId { module, ty });
        // Unqualified: the same module first.
        assert_eq!(site.resolve_type(ast, "expr"), id(ast, 0));
        // Unqualified in a nested module: the enclosing module.
        assert_eq!(site.resolve_type(loc, "expr"), id(ast, 0));
        assert_eq!(site.resolve_type(loc, "t"), id(loc, 0));
        // Qualified.
        assert_eq!(site.resolve_type(ast, "Ir.expr"), id(ir, 0));
        // Unqualified, defined in exactly one other module.
        assert_eq!(site.resolve_type(util, "only_here"), id(ast, 1));
        // Ambiguous across modules: `t` is in Ast.Loc and Ir.
        assert_eq!(site.resolve_type(util, "t"), None);
        // A wrapper prefix is dropped.
        assert_eq!(site.resolve_type(util, "Camdl.Ir.t"), id(ir, 1));
        // A nested module named relative to its parent, from inside the parent.
        assert_eq!(site.resolve_type(ast, "Loc.t"), id(loc, 0));
        // Outside inputs stay unresolved.
        assert_eq!(site.resolve_type(ast, "Yojson.Safe.t"), None);
        assert!(site.is_predefined(ast, "int"));
        assert!(!site.is_predefined(ast, "Yojson.Safe.t"));
    }

    #[test]
    fn reverse_index_excludes_self_references() {
        let inputs = sample();
        let m = overview(&[&["Ast", "Ir"], &["Util"]]);
        let site = Site::build(&m, 10, &inputs).unwrap();
        let ast_expr = TypeId { module: 0, ty: 0 };
        assert_eq!(
            site.used_by(ast_expr),
            &[TypeId { module: 1, ty: 0 }],
            "Ast.expr refers to itself, which must not count as a use"
        );
        assert_eq!(
            site.used_by(TypeId { module: 2, ty: 0 }),
            &[ast_expr],
            "Ir.expr is used by Ast.expr"
        );
    }

    #[test]
    fn index_hrefs_resolve_full_paths() {
        let inputs = sample();
        let m = overview(&[&["Ast", "Ir"], &["Util"]]);
        let site = Site::build(&m, 10, &inputs).unwrap();
        let href = |p: &str| site.index_href(p);
        assert_eq!(href("Ir.expr").as_deref(), Some("ir.qmd#t-expr"));
        assert_eq!(href("Ast.Loc.t").as_deref(), Some("ast.qmd#t-Loc-t"));
        assert_eq!(href("only_here").as_deref(), Some("ast.qmd#t-only_here"));
        assert_eq!(href("Ir").as_deref(), Some("ir.qmd"));
        assert_eq!(href("Ast.Loc").as_deref(), Some("ast.qmd#m-Loc"));
        assert_eq!(href("Camdl.Ir.t").as_deref(), Some("ir.qmd#t-t"));
        // Ambiguous, and absent.
        assert_eq!(href("t"), None);
        assert_eq!(href("Gone.t"), None);
    }

    #[test]
    fn anchors_and_hrefs() {
        let inputs = sample();
        let m = overview(&[&["Ast", "Ir"], &["Util"]]);
        let site = Site::build(&m, 10, &inputs).unwrap();
        assert_eq!(site.modules[1].anchors, vec!["t-Loc-t", "t-Loc-u"]);
        assert_eq!(
            site.type_href(From::Page(0), TypeId { module: 1, ty: 0 }),
            "#t-Loc-t"
        );
        assert_eq!(
            site.type_href(From::Page(2), TypeId { module: 1, ty: 0 }),
            "ast.qmd#t-Loc-t"
        );
        assert_eq!(
            site.type_display(From::Page(2), TypeId { module: 1, ty: 0 }),
            "Ast.Loc.t"
        );
        assert_eq!(
            site.type_display(From::Page(0), TypeId { module: 1, ty: 0 }),
            "Loc.t"
        );
        assert_eq!(site.module_href(From::Page(1), 1), "ast.qmd#m-Loc");
        assert_eq!(site.pages[2].order, 13);
    }

    #[test]
    fn unlisted_and_missing_modules_fail_naming_them() {
        let inputs = sample();
        let m = overview(&[&["Ast", "Gone"], &["Util"]]);
        let err = Site::build(&m, 10, &inputs).unwrap_err();
        assert!(
            err.contains("input modules in no {!modules: ...} list of the overview: Ir"),
            "{err}"
        );
        assert!(
            err.contains("overview modules absent from the input: Gone"),
            "{err}"
        );
    }

    #[test]
    fn an_unlisted_module_alone_fails() {
        let inputs = sample();
        let m = overview(&[&["Ast", "Util"]]);
        let err = Site::build(&m, 10, &inputs).unwrap_err();
        assert_eq!(
            err,
            "input modules in no {!modules: ...} list of the overview: Ir"
        );
    }

    #[test]
    fn nested_modules_are_covered_by_their_parent() {
        let inputs = sample();
        let m = overview(&[&["Ast", "Ir", "Util"]]);
        assert!(Site::build(&m, 10, &inputs).is_ok());
        let m = overview(&[&["Ir", "Util", "Ast.Loc"]]);
        let err = Site::build(&m, 10, &inputs).unwrap_err();
        assert!(err.contains("nested modules"), "{err}");
    }

    #[test]
    fn duplicate_names_fail() {
        let inputs = sample();
        let m = overview(&[&["Ast", "Ir"], &["Util", "Ir"]]);
        let err = Site::build(&m, 10, &inputs).unwrap_err();
        assert!(err.contains("more than once in the overview: Ir"), "{err}");
        let twice = vec![sample().remove(0), sample().remove(0)];
        let m = overview(&[&["Ast", "Ir", "Util"]]);
        let err = Site::build(&m, 10, &twice).unwrap_err();
        assert!(err.contains("more than once in the input"), "{err}");
    }

    #[test]
    fn targets_resolve_exactly_by_module_and_name() {
        // Mutation check: resolving a target by its type name alone (as
        // resolve_type does for an unqualified path) makes `expr` ambiguous
        // between Ast and Ir and fails the first two assertions.
        let inputs = sample();
        let m = overview(&[&["Ast", "Ir"], &["Util"]]);
        let site = Site::build(&m, 10, &inputs).unwrap();
        let target = |m: &str, t: &str| Target {
            module: m.to_string(),
            ty: t.to_string(),
        };
        let id = |module, ty| Some(TypeId { module, ty });
        assert_eq!(site.find_target(&target("Ast", "expr")), id(0, 0));
        assert_eq!(site.find_target(&target("Ir", "expr")), id(2, 0));
        assert_eq!(site.find_target(&target("Ast.Loc", "t")), id(1, 0));
        assert_eq!(site.find_target(&target("Ast", "t")), None);
        assert_eq!(site.find_target(&target("Loc", "t")), None);
        assert_eq!(
            site.type_href_html(From::Source, TypeId { module: 2, ty: 0 }),
            "../ir.html#t-expr"
        );
        assert_eq!(
            site.type_href_html(From::Page(0), TypeId { module: 2, ty: 0 }),
            "ir.html#t-expr"
        );
        assert_eq!(
            site.type_href_html(From::Page(0), TypeId { module: 1, ty: 0 }),
            "#t-Loc-t"
        );
        assert_eq!(site.source_of(1).slug, "ast");
    }

    #[test]
    fn a_target_naming_no_declaration_fails() {
        let inputs = vec![ocaml(
            &[module("Ast", &[ty("e", &[], &["Ast.gone"])])],
            &["ast"],
        )];
        let m = overview(&[&["Ast"]]);
        let err = Site::build(&m, 10, &inputs).unwrap_err();
        assert_eq!(
            err,
            "ast.ml, the declaration at line 1: a span links to Ast.gone, which no input declares"
        );
    }

    #[test]
    fn every_module_file_needs_a_source_entry() {
        let inputs = vec![ocaml(&[module("Ast", &[]), module("Ir", &[])], &["ast"])];
        let m = overview(&[&["Ast", "Ir"]]);
        let err = Site::build(&m, 10, &inputs).unwrap_err();
        assert_eq!(err, "module files with no source entry in the input: ir.ml");
    }

    #[test]
    fn a_module_named_index_is_refused() {
        let inputs = vec![ocaml(&[module("Index", &[])], &["index"])];
        let m = overview(&[&["Index"]]);
        let err = Site::build(&m, 10, &inputs).unwrap_err();
        assert!(err.contains("index page"), "{err}");
    }
}
