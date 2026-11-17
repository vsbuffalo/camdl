//! The JSON a language extractor writes: the modules of one language's source
//! tree and the type declarations in each.
//!
//! One extractor runs per language and writes one [`Input`]. The renderer
//! reads any number of them, so the compiler's OCaml types and the runtime's
//! Rust types can refer to each other on the generated pages. Every field is
//! required except the doc comments, which are `null` when the source has
//! none, and the fields only the Rust extractor writes ([`TypeDecl::impls`]
//! and the `doc_links` of modules, types and members), which are empty when
//! absent. An unknown field is refused, so a change to what an extractor
//! writes fails the render instead of being ignored.
//!
//! The renderer reads this JSON and the Rust extractor
//! ([`crate::rustdoc`]) writes it from these same types, so the two cannot
//! disagree about its shape.

use serde::{Deserialize, Serialize};

/// Everything one extractor found in one language's source tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    /// The language the modules are written in. It decides how doc comments
    /// are read and how declarations are highlighted.
    pub language: Language,
    /// The modules, in the order the extractor visited them.
    pub modules: Vec<Module>,
    /// One entry per source file the modules come from, carrying the markup
    /// of its text. Every [`Module::path`] must have one.
    pub sources: Vec<SourceFile>,
}

/// The markup of one source file, which the renderer shows whole on a page
/// of its own. The text is not in the JSON: the renderer reads it from
/// [`SourceFile::path`] and refuses it when its size differs from what the
/// extractor read.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceFile {
    /// The file, relative to the repository root, as [`Module::path`] names it.
    pub path: String,
    /// The number of lines: the count of newlines, plus one when the text
    /// does not end with a newline.
    pub lines: usize,
    /// The file's size in bytes.
    pub bytes: usize,
    /// The parts of the text that get markup, in text order; text between
    /// them is shown plain. Lines count from 1, as in the file.
    pub spans: Vec<Span>,
}

/// A run of text on one line that gets markup: a class, and a link when it
/// names a documented type. Spans never overlap and never cross a line end;
/// one that would is split into one span per line.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Span {
    /// The line, counting from 1.
    pub line: usize,
    /// The start, in bytes from the start of the line, counting from 0.
    pub col: usize,
    /// The length in bytes.
    pub len: usize,
    /// What the text is.
    pub class: SpanClass,
    /// The documented type the text names. Absent for a span that does not
    /// link. The extractor resolves it from the language's own analysis, so
    /// it is exact: a name resolved by guessing never gets one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<Target>,
}

/// What a span of source text is, for highlighting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpanClass {
    /// A keyword of the language: `type`, `let`, `match`.
    Keyword,
    /// An implementation comment.
    Comment,
    /// A doc comment.
    Doc,
    /// A string literal.
    String,
    /// A character literal.
    Char,
    /// A number literal.
    Number,
    /// A type the language itself provides: `int`, `list`, `option`.
    BuiltinType,
    /// The name of a type at a use or at its definition. It links when it
    /// carries a [`Span::target`].
    Type,
}

/// A documented type, named by its module's path ([`Module::name`]) and its
/// name ([`TypeDecl::name`]). The extractor emits it only for the
/// declaration that module binds the name to last, so the pair is unique.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Target {
    pub module: String,
    #[serde(rename = "type")]
    pub ty: String,
}

/// A source language the extractors support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    /// OCaml. Doc comments are odoc markup, written `(** ... *)`.
    Ocaml,
    /// Rust. Doc comments are Markdown, written `///` and `//!`.
    Rust,
}

impl Language {
    /// The info string of a fenced code block highlighted as this language.
    pub fn fence(self) -> &'static str {
        match self {
            Language::Ocaml => "ocaml",
            Language::Rust => "rust",
        }
    }

    /// What joins the segments of a module path: `Ast.Located` in OCaml,
    /// `sim::inference` in Rust.
    pub fn separator(self) -> &'static str {
        match self {
            Language::Ocaml => ".",
            Language::Rust => "::",
        }
    }
}

/// One module: a source file, or a module nested inside one.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Module {
    /// The module's path as the language names it, its segments joined by
    /// [`Language::separator`]. In OCaml a file's module is its capitalised
    /// basename (`Ast`), and a module nested in it is written with a dot
    /// (`Ast.Located`); the first segment names the page the module is shown
    /// on. In Rust the first segment is the crate (`sim::inference::pgas`),
    /// and each source file's modules get a page of their own.
    pub name: String,
    /// The source file, relative to the repository root.
    pub path: String,
    /// The module's doc comment, as raw markup of [`Input::language`].
    pub doc: Option<String>,
    /// What the links in [`Module::doc`] point to, as the language resolved
    /// them. Rust only.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub doc_links: Vec<DocLink>,
    /// The type declarations, in source order.
    pub types: Vec<TypeDecl>,
    /// The functions, constants and statics the module declares, and the
    /// methods and associated constants of its types' inherent impls, which
    /// the pages do not document but an engineering page may link to. Rust
    /// only.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub values: Vec<Value>,
}

/// A function, constant, static or inherent method, by where it is written.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Value {
    /// Its name in the module: `compute` for a function, `Seed::new` for a
    /// method of the module's type `Seed`, joined by
    /// [`Language::separator`]. Unique within the module: a name two items
    /// share (methods of two impls of one generic type) is left out rather
    /// than resolved to either.
    pub name: String,
    /// The file it is written in, relative to the repository root: a source
    /// file of the inputs, though not always the module's own (an impl may
    /// be written in another file of the crate).
    pub path: String,
    /// The line it starts on, counting from 1.
    pub line: u32,
}

/// One type declaration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypeDecl {
    /// The type's name, without parameters or module path.
    pub name: String,
    /// The line of the source file the declaration starts on, counting from 1.
    pub line: u32,
    /// What form of declaration it is.
    pub kind: Kind,
    /// The type parameters as written, such as `'a` or `T`.
    pub params: Vec<String>,
    /// Declarations that share a number are defined together and may refer
    /// to one another (in OCaml, joined by `and`). The number identifies the
    /// group within the module only. `null` for a declaration defined alone.
    pub recursive_group: Option<u32>,
    /// The declaration's source text with every comment removed.
    pub decl: String,
    /// The markup of [`TypeDecl::decl`], as in [`SourceFile::spans`] but
    /// placed in the declaration's own text: line 1 is its first line, and
    /// columns count after the indentation was removed.
    pub decl_spans: Vec<Span>,
    /// The type's doc comment, as raw markup of [`Input::language`].
    pub doc: Option<String>,
    /// The constructors of a variant or enum, or the fields of a record or
    /// struct, in source order. Empty for the other kinds.
    pub members: Vec<Member>,
    /// The type paths the declaration mentions, as written in the source:
    /// `expr`, `Ir.expr`. A path may name the declaration itself.
    pub refs: Vec<String>,
    /// What the links in [`TypeDecl::doc`] point to. Rust only.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub doc_links: Vec<DocLink>,
    /// The traits the type implements, in the order the compiler lists the
    /// impls. Rust only: OCaml has no counterpart.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub impls: Vec<Impl>,
}

/// A trait a Rust type implements. Only impls written for the type itself
/// are listed: auto traits (`Send`, `Sync`) the compiler infers and blanket
/// impls (`impl<T: Display> ToString for T`) are left out, because they say
/// nothing about the type that its declaration does not.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Impl {
    /// The trait as the impl names it, with its arguments: `Clone`,
    /// `From<ir::Model>`, `std::fmt::Display`. For a derived impl, the name
    /// in the `#[derive]` list.
    #[serde(rename = "trait")]
    pub trait_path: String,
    /// The trait's declaration, when an input declares it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<Target>,
    /// Whether the impl comes from `#[derive(...)]` rather than being
    /// written out.
    pub derived: bool,
    /// The file the impl is written in, relative to the repository root.
    /// `null` for a derived impl.
    pub path: Option<String>,
    /// The line the impl starts on, counting from 1. `null` for a derived
    /// impl.
    pub line: Option<u32>,
}

/// A link in a doc comment, as the language's own tooling resolved it, so
/// the renderer does not have to guess what a name means in the comment's
/// scope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DocLink {
    /// The link's destination as written, without backticks: `Model`,
    /// `crate::expr::Expr`, `Self::new`.
    pub text: String,
    /// What it points to.
    pub to: LinkTo,
}

/// The destination of a [`DocLink`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum LinkTo {
    /// A documented type, or a variant or field of one, which is shown with
    /// the type.
    Type {
        module: String,
        #[serde(rename = "type")]
        ty: String,
    },
    /// A module of the inputs.
    Module { module: String },
    /// Something the pages do not document, such as a function, method or
    /// constant, shown at its line on its source file's page. `path` is
    /// relative to the repository root and must be a source file of the
    /// inputs.
    Source { path: String, line: u32 },
}

/// The form of a type declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// An OCaml variant type: a choice between constructors.
    Variant,
    /// An OCaml record type: named fields.
    Record,
    /// A name for another type expression.
    Alias,
    /// A type whose definition is not shown, such as an OCaml `type t`.
    Abstract,
    /// An OCaml extensible variant, `type t = ..`, to which other modules
    /// add constructors.
    Extensible,
    /// A Rust struct with named fields.
    Struct,
    /// A Rust struct with positional fields.
    TupleStruct,
    /// A Rust enum: a choice between variants.
    Enum,
    /// A Rust trait. Its members are its associated types, constants and
    /// methods, each with its signature as the member's type.
    Trait,
}

impl Kind {
    /// What this kind calls its members, in the plural: constructors for a
    /// choice, fields for a product, or members otherwise.
    pub fn member_noun(self) -> &'static str {
        match self {
            Kind::Variant | Kind::Extensible => "constructors",
            Kind::Enum => "variants",
            Kind::Record | Kind::Struct | Kind::TupleStruct => "fields",
            Kind::Alias | Kind::Abstract | Kind::Trait => "members",
        }
    }

    /// Whether the members are named fields of one value, as opposed to
    /// alternatives.
    pub fn has_fields(self) -> bool {
        matches!(self, Kind::Record | Kind::Struct | Kind::TupleStruct)
    }
}

/// A constructor of a variant or enum, a field of a record or struct, or an
/// item of a Rust trait.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Member {
    /// The constructor's, field's or trait item's name.
    pub name: String,
    /// For a field, its type. For a constructor, the type of its argument,
    /// or the empty string for a constructor that takes none; a Rust enum
    /// variant's is written with its delimiters, `(f64, u32)` or
    /// `{ at: f64 }`. For a trait item, its signature, such as
    /// `fn rate(&self, t: f64) -> f64`.
    #[serde(rename = "type")]
    pub ty: String,
    /// Whether the field can be assigned after the value is built (an OCaml
    /// `mutable` field). Always `false` for a constructor and in Rust, where
    /// mutability belongs to the binding, not the field.
    pub mutable: bool,
    /// The member's doc comment, as raw markup of [`Input::language`].
    pub doc: Option<String>,
    /// What the links in [`Member::doc`] point to. Rust only.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub doc_links: Vec<DocLink>,
}
