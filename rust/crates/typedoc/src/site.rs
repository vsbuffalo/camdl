//! The site: which page each module is shown on, the anchor of each type, how
//! a type path written in one module is resolved to a declaration, and which
//! types refer to each type.

use std::collections::{BTreeMap, BTreeSet};

use crate::manifest::Manifest;
use crate::model::{Input, Language, Module};

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

/// One generated module page: a top-level module and the modules nested in it.
#[derive(Debug)]
pub struct Page {
    /// The top-level module's name, as the manifest lists it.
    pub name: String,
    /// The file name without extension.
    pub slug: String,
    /// The manifest group the page is listed under.
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
    pub manifest: &'a Manifest,
    pub modules: Vec<ModuleEntry<'a>>,
    pub pages: Vec<Page>,
    by_name: BTreeMap<&'a str, usize>,
    defining: BTreeMap<&'a str, Vec<usize>>,
    used_by: BTreeMap<TypeId, Vec<TypeId>>,
}

/// The top-level module a module path belongs to: the part before the first dot.
fn top_of(name: &str) -> &str {
    name.split('.').next().unwrap_or(name)
}

/// The file name of a module page: the module name in lower case, with dots
/// as hyphens.
pub fn slug_of(name: &str) -> String {
    name.to_lowercase().replace('.', "-")
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
    /// Places every module of `inputs` on the page the manifest gives it.
    ///
    /// Fails, naming the modules, when an input module's top-level module is
    /// not in the manifest, when a manifest module has no input module, when
    /// a name occurs twice in the manifest or in the inputs, or when a module
    /// page would overwrite the index page.
    pub fn build(manifest: &'a Manifest, inputs: &'a [Input]) -> Result<Site<'a>, String> {
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
        let dup_listed: BTreeSet<&str> = manifest
            .module_names()
            .filter(|n| !listed.insert(*n))
            .collect();
        if !dup_listed.is_empty() {
            return Err(format!(
                "modules listed more than once in the manifest: {}",
                join(&dup_listed)
            ));
        }
        let nested_listed: BTreeSet<&str> =
            listed.iter().copied().filter(|n| n.contains('.')).collect();
        if !nested_listed.is_empty() {
            return Err(format!(
                "the manifest lists nested modules, which are shown on their parent's page; \
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
                    "input modules not in the manifest: {}",
                    join(&unlisted)
                ));
            }
            if !missing.is_empty() {
                msg.push(format!(
                    "manifest modules absent from the input: {}",
                    join(&missing)
                ));
            }
            return Err(msg.join("; "));
        }

        let mut modules = Vec::new();
        let mut pages = Vec::new();
        for (gi, group) in manifest.groups.iter().enumerate() {
            for name in &group.modules {
                let slug = slug_of(name);
                if slug == INDEX_SLUG {
                    return Err(format!(
                        "module {name} would be written to {INDEX_SLUG}.qmd, the index page"
                    ));
                }
                let page = pages.len();
                let mut members: Vec<(&'a Module, Language)> = all
                    .iter()
                    .copied()
                    .filter(|(m, _)| top_of(&m.name) == name)
                    .collect();
                // The top-level module first; nested ones keep input order.
                members.sort_by_key(|(m, _)| m.name != *name);
                let mut used_anchors = BTreeSet::new();
                let mut indices = Vec::new();
                for (m, language) in members {
                    let local = m
                        .name
                        .strip_prefix(name.as_str())
                        .and_then(|r| r.strip_prefix('.'))
                        .unwrap_or("")
                        .to_string();
                    let anchors = m
                        .types
                        .iter()
                        .map(|t| {
                            let path = if local.is_empty() {
                                t.name.clone()
                            } else {
                                format!("{local}.{}", t.name)
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
                    slug,
                    group: gi,
                    order: manifest.order + 1 + page as i64,
                    modules: indices,
                });
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
        let mut site = Site {
            manifest,
            modules,
            pages,
            by_name,
            defining,
            used_by: BTreeMap::new(),
        };

        let mut used_by: BTreeMap<TypeId, Vec<TypeId>> = BTreeMap::new();
        for (mi, e) in site.modules.iter().enumerate() {
            for (ti, t) in e.module.types.iter().enumerate() {
                let me = TypeId { module: mi, ty: ti };
                for r in &t.refs {
                    if let Some(target) = site.resolve_type(mi, r) {
                        if target != me {
                            let users = used_by.entry(target).or_default();
                            if users.last() != Some(&me) {
                                users.push(me);
                            }
                        }
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
        let mut out = vec![name];
        let mut rest = name;
        while let Some(i) = rest.rfind('.') {
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
        for scope in self.scopes(from) {
            if let Some(&m) = self.by_name.get(format!("{scope}.{path}").as_str()) {
                return Some(m);
            }
        }
        if let Some(&m) = self.by_name.get(path) {
            return Some(m);
        }
        let segs: Vec<&str> = path.split('.').collect();
        (1..segs.len()).find_map(|k| self.by_name.get(segs[k..].join(".").as_str()).copied())
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
        match path.rsplit_once('.') {
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
                match self.defining.get(path).map(Vec::as_slice) {
                    Some([only]) => self.find_in(*only, path),
                    _ => None,
                }
            }
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

    /// The types whose declarations refer to `id`, in site order.
    pub fn used_by(&self, id: TypeId) -> &[TypeId] {
        self.used_by.get(&id).map(Vec::as_slice).unwrap_or(&[])
    }

    /// The link target of a type from a page: an anchor on the same page, or
    /// the other page's file and anchor.
    pub fn type_href(&self, from_page: usize, id: TypeId) -> String {
        let e = &self.modules[id.module];
        let anchor = &e.anchors[id.ty];
        if e.page == from_page {
            format!("#{anchor}")
        } else {
            format!("{}.qmd#{anchor}", self.pages[e.page].slug)
        }
    }

    /// The anchor of a nested module's section on its page. The top-level
    /// module has none; its page is the target.
    pub fn module_anchor(&self, m: usize) -> Option<String> {
        let local = &self.modules[m].local;
        (!local.is_empty()).then(|| format!("m-{}", sanitize_id(local)))
    }

    /// The link target of a module from a page.
    pub fn module_href(&self, from_page: usize, m: usize) -> String {
        let e = &self.modules[m];
        let file = format!("{}.qmd", self.pages[e.page].slug);
        match (self.module_anchor(m), e.page == from_page) {
            (Some(a), true) => format!("#{a}"),
            (Some(a), false) => format!("{file}#{a}"),
            (None, _) => file,
        }
    }

    /// A type's name as a reader of `from_page` should see it: qualified by
    /// its nested module on its own page, by its full module path elsewhere.
    pub fn type_display(&self, from_page: usize, id: TypeId) -> String {
        let e = &self.modules[id.module];
        let name = &e.module.types[id.ty].name;
        if e.page != from_page {
            format!("{}.{name}", e.module.name)
        } else if e.local.is_empty() {
            name.clone()
        } else {
            format!("{}.{name}", e.local)
        }
    }
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

    fn ty(name: &str, refs: &[&str]) -> String {
        format!(
            r#"{{"name":"{name}","line":1,"kind":"alias","params":[],"recursive_group":null,
               "decl":"type {name} = int","doc":null,"members":[],"refs":{}}}"#,
            serde_json::to_string(refs).unwrap()
        )
    }

    fn module(name: &str, types: &[String]) -> String {
        format!(
            r#"{{"name":"{name}","path":"x.ml","doc":null,"types":[{}]}}"#,
            types.join(",")
        )
    }

    fn manifest(groups: &[&[&str]]) -> Manifest {
        let mut s = String::from("title='T'\nsummary='S'\nrepo_blob='https://h/'\norder=10\n");
        for g in groups {
            s.push_str(&format!(
                "[[group]]\ntitle='G'\nmodules={}\n",
                serde_json::to_string(g).unwrap()
            ));
        }
        Manifest::from_toml(&s).unwrap()
    }

    fn sample() -> Vec<Input> {
        vec![input(&format!(
            r#"{{"language":"ocaml","modules":[{},{},{},{}]}}"#,
            module(
                "Ast",
                &[
                    ty("expr", &["expr", "Ir.expr", "Yojson.Safe.t", "int"]),
                    ty("only_here", &[])
                ]
            ),
            module("Ast.Loc", &[ty("t", &["expr"]), ty("u", &["t"])]),
            module("Ir", &[ty("expr", &[]), ty("t", &[])]),
            module("Util", &[ty("w", &["only_here", "Loc.t", "Camdl.Ir.t"])]),
        ))]
    }

    #[test]
    fn resolves_qualified_unqualified_nested_and_wrapped_paths() {
        let inputs = sample();
        let m = manifest(&[&["Ast", "Ir"], &["Util"]]);
        let site = Site::build(&m, &inputs).unwrap();
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
        let m = manifest(&[&["Ast", "Ir"], &["Util"]]);
        let site = Site::build(&m, &inputs).unwrap();
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
    fn anchors_and_hrefs() {
        let inputs = sample();
        let m = manifest(&[&["Ast", "Ir"], &["Util"]]);
        let site = Site::build(&m, &inputs).unwrap();
        assert_eq!(site.modules[1].anchors, vec!["t-Loc-t", "t-Loc-u"]);
        assert_eq!(site.type_href(0, TypeId { module: 1, ty: 0 }), "#t-Loc-t");
        assert_eq!(
            site.type_href(2, TypeId { module: 1, ty: 0 }),
            "ast.qmd#t-Loc-t"
        );
        assert_eq!(
            site.type_display(2, TypeId { module: 1, ty: 0 }),
            "Ast.Loc.t"
        );
        assert_eq!(site.type_display(0, TypeId { module: 1, ty: 0 }), "Loc.t");
        assert_eq!(site.module_href(1, 1), "ast.qmd#m-Loc");
        assert_eq!(site.pages[2].order, 13);
    }

    #[test]
    fn unlisted_and_missing_modules_fail_naming_them() {
        let inputs = sample();
        let m = manifest(&[&["Ast", "Gone"], &["Util"]]);
        let err = Site::build(&m, &inputs).unwrap_err();
        assert!(
            err.contains("input modules not in the manifest: Ir"),
            "{err}"
        );
        assert!(
            err.contains("manifest modules absent from the input: Gone"),
            "{err}"
        );
    }

    #[test]
    fn nested_modules_are_covered_by_their_parent() {
        let inputs = sample();
        let m = manifest(&[&["Ast", "Ir", "Util"]]);
        assert!(Site::build(&m, &inputs).is_ok());
        let m = manifest(&[&["Ir", "Util", "Ast.Loc"]]);
        let err = Site::build(&m, &inputs).unwrap_err();
        assert!(err.contains("nested modules"), "{err}");
    }

    #[test]
    fn duplicate_names_fail() {
        let inputs = sample();
        let m = manifest(&[&["Ast", "Ir"], &["Util", "Ir"]]);
        let err = Site::build(&m, &inputs).unwrap_err();
        assert!(err.contains("more than once in the manifest: Ir"), "{err}");
        let twice = vec![sample().remove(0), sample().remove(0)];
        let m = manifest(&[&["Ast", "Ir", "Util"]]);
        let err = Site::build(&m, &twice).unwrap_err();
        assert!(err.contains("more than once in the input"), "{err}");
    }

    #[test]
    fn a_module_named_index_is_refused() {
        let inputs = vec![input(&format!(
            r#"{{"language":"ocaml","modules":[{}]}}"#,
            module("Index", &[])
        ))];
        let m = manifest(&[&["Index"]]);
        let err = Site::build(&m, &inputs).unwrap_err();
        assert!(err.contains("index page"), "{err}");
    }
}
