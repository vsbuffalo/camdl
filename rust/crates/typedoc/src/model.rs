//! The JSON a language extractor writes: the modules of one language's source
//! tree and the type declarations in each.
//!
//! One extractor runs per language and writes one [`Input`]. The renderer
//! reads any number of them, so the compiler's OCaml types and the runtime's
//! Rust types can refer to each other on the generated pages. Every field is
//! required except the doc comments, which are `null` when the source has
//! none, and an unknown field is refused, so a change to what an extractor
//! writes fails the render instead of being ignored.

use serde::Deserialize;

/// Everything one extractor found in one language's source tree.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    /// The language the modules are written in. It decides how doc comments
    /// are read and how declarations are highlighted.
    pub language: Language,
    /// The modules, in the order the extractor visited them.
    pub modules: Vec<Module>,
}

/// A source language the extractors support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
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
}

/// One module: a source file, or a module nested inside one.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Module {
    /// The module's path as the language names it. In OCaml a file's module
    /// is its capitalised basename (`Ast`), and a module nested in it is
    /// written with a dot (`Ast.Located`). The part before the first dot
    /// names the page the module is shown on.
    pub name: String,
    /// The source file, relative to the repository root.
    pub path: String,
    /// The module's doc comment, as raw markup of [`Input::language`].
    pub doc: Option<String>,
    /// The type declarations, in source order.
    pub types: Vec<TypeDecl>,
}

/// One type declaration.
#[derive(Debug, Clone, Deserialize)]
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
    /// The type's doc comment, as raw markup of [`Input::language`].
    pub doc: Option<String>,
    /// The constructors of a variant or enum, or the fields of a record or
    /// struct, in source order. Empty for the other kinds.
    pub members: Vec<Member>,
    /// The type paths the declaration mentions, as written in the source:
    /// `expr`, `Ir.expr`. A path may name the declaration itself.
    pub refs: Vec<String>,
}

/// The form of a type declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
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
    /// A Rust trait.
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

/// A constructor of a variant or enum, or a field of a record or struct.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Member {
    /// The constructor's or field's name.
    pub name: String,
    /// For a field, its type. For a constructor, the type of its argument,
    /// or the empty string for a constructor that takes none.
    #[serde(rename = "type")]
    pub ty: String,
    /// Whether the field can be assigned after the value is built. Always
    /// `false` for a constructor.
    pub mutable: bool,
    /// The member's doc comment, as raw markup of [`Input::language`].
    pub doc: Option<String>,
}
