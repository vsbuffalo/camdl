//! `#[derive(Inputs)]` — the class of every command-line input, declared on
//! the clap field itself (`cli::input_class`).
//!
//! Each field of a clap `Args` struct carries one class:
//!
//! ```text
//! #[input(keyed(<level>))]   #[input(execution)]   #[input(provenance)]
//! #[input(presentation)]     #[input(child)]       #[input(expands)]
//! #[input(removed)]
//! ```
//!
//! `<level>` is a `Level` variant in lower case (`model`, `config`, `fit`,
//! `box`, ...). A field that clap flattens (`#[command(flatten)]`) carries no
//! class: its type must derive `Inputs` too, and its fields come in with the
//! classes that type declares, unless the flatten site overrides some:
//!
//! ```text
//! #[command(flatten)]
//! #[input(override(allow_forcing_extrapolation = keyed(fit)))]
//! pub forcing_extrapolation: ForcingExtrapolationArg,
//! ```
//!
//! A struct that is a command's argument list adds the levels its leaf has and
//! the global options it receives, with their overrides:
//!
//! ```text
//! #[input(command(levels(fit, method, seed),
//!                 global(crate::GlobalArgs, override(no_licm = keyed(fit)))))]
//! ```
//!
//! Generated:
//!
//! - `impl crate::input_class::InputFields for T`, whose `FIELDS` is every
//!   argument of `T`, its own and its flattened groups', by clap argument id
//!   (the field name), with its class;
//! - for a command, inherent consts `T::INPUTS` (`FIELDS` and the global
//!   options) and `T::LEVELS`, and a constant check, evaluated in every build,
//!   that no argument id occurs twice and that every `keyed` class names a
//!   level in `LEVELS`.
//!
//! What fails to compile:
//!
//! - a field with no class, or a flattened field with one (a macro error on the
//!   field);
//! - a flattened group, or global options, whose type does not derive `Inputs`
//!   (`T: InputFields` is not satisfied);
//! - an override naming a field the group does not have (a failed constant
//!   assertion at the override);
//! - a `keyed` class naming no `Level` variant, or one the command does not
//!   declare in `levels(..)`.
//!
//! The generated code names `crate::input_class`, so the derive is used only
//! in the `cli` crate, where that module lives.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote, quote_spanned};
use syn::ext::IdentExt;
use syn::parse::ParseStream;
use syn::spanned::Spanned;
use syn::{parenthesized, parse_macro_input, Attribute, Data, DeriveInput, Fields, Ident, Token, Type};

#[proc_macro_derive(Inputs, attributes(input))]
pub fn derive_inputs(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match expand(&input) {
        Ok(ts) => ts.into(),
        Err(e) => {
            // The error, and an empty inventory, so that a struct flattening
            // this one reports this error alone, not also an unimplemented trait.
            let err = e.to_compile_error();
            let ident = &input.ident;
            let stub = input.generics.params.is_empty().then(|| {
                quote! {
                    impl crate::input_class::InputFields for #ident {
                        const FIELDS: crate::input_class::Inventory = &[];
                    }
                }
            });
            quote!(#err #stub).into()
        }
    }
}

const CLASSES: &str = "keyed(<level>), execution, provenance, presentation, child, expands or removed";

/// A class as written in an attribute.
enum Class {
    Keyed(Ident),
    Plain(Ident),
}

impl Class {
    fn tokens(&self) -> TokenStream2 {
        match self {
            Class::Keyed(level) => {
                let v = Ident::new(&capitalize(&level.unraw().to_string()), level.span());
                quote_spanned!(level.span()=> crate::input_class::InputClass::Keyed(crate::input_class::Level::#v))
            }
            Class::Plain(name) => {
                let v = Ident::new(&capitalize(&name.to_string()), name.span());
                quote_spanned!(name.span()=> crate::input_class::InputClass::#v)
            }
        }
    }
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().chain(c).collect(),
        None => String::new(),
    }
}

fn parse_class(input: ParseStream) -> syn::Result<Class> {
    let name = input.call(Ident::parse_any)?;
    match name.to_string().as_str() {
        "keyed" => {
            let content;
            parenthesized!(content in input);
            let level = content.call(Ident::parse_any)?;
            if !content.is_empty() {
                return Err(content.error("keyed takes one level, e.g. keyed(config)"));
            }
            Ok(Class::Keyed(level))
        }
        "execution" | "provenance" | "presentation" | "child" | "expands" | "removed" => Ok(Class::Plain(name)),
        _ => Err(syn::Error::new(name.span(), format!("expected an input class: {CLASSES}"))),
    }
}

/// `override(name = class, ...)`, after the `override` keyword.
fn parse_overrides(input: ParseStream) -> syn::Result<Vec<(Ident, Class)>> {
    let content;
    parenthesized!(content in input);
    let mut out: Vec<(Ident, Class)> = Vec::new();
    while !content.is_empty() {
        let name = content.call(Ident::parse_any)?;
        content.parse::<Token![=]>()?;
        let class = parse_class(&content)?;
        if out.iter().any(|(n, _)| n == &name) {
            return Err(syn::Error::new(name.span(), format!("`{name}` is overridden twice")));
        }
        out.push((name, class));
        if content.is_empty() {
            break;
        }
        content.parse::<Token![,]>()?;
    }
    Ok(out)
}

enum FieldAttr {
    Class(Class),
    Override(Vec<(Ident, Class)>),
}

fn parse_field_attr(attr: &Attribute) -> syn::Result<FieldAttr> {
    attr.parse_args_with(|input: ParseStream| {
        let fork = input.fork();
        let head = fork.call(Ident::parse_any)?;
        let out = if head == "override" {
            input.call(Ident::parse_any)?;
            FieldAttr::Override(parse_overrides(input)?)
        } else {
            FieldAttr::Class(parse_class(input)?)
        };
        if !input.is_empty() {
            return Err(input.error("one class (or one override list) per #[input(..)]"));
        }
        Ok(out)
    })
}

/// `#[input(command(levels(..), global(Type, override(..))))]`.
struct Command {
    levels: Vec<Ident>,
    global: Type,
    global_overrides: Vec<(Ident, Class)>,
}

fn parse_command(attr: &Attribute) -> syn::Result<Command> {
    attr.parse_args_with(|input: ParseStream| {
        let kw = input.call(Ident::parse_any)?;
        if kw != "command" {
            return Err(syn::Error::new(kw.span(), "on a struct, #[input(..)] takes command(levels(..), global(..))"));
        }
        let content;
        parenthesized!(content in input);
        let (mut levels, mut global, mut global_overrides) = (None, None, Vec::new());
        while !content.is_empty() {
            let item = content.call(Ident::parse_any)?;
            match item.to_string().as_str() {
                "levels" => {
                    let inner;
                    parenthesized!(inner in content);
                    let mut ls = Vec::new();
                    while !inner.is_empty() {
                        ls.push(inner.call(Ident::parse_any)?);
                        if inner.is_empty() {
                            break;
                        }
                        inner.parse::<Token![,]>()?;
                    }
                    levels = Some(ls);
                }
                "global" => {
                    let inner;
                    parenthesized!(inner in content);
                    global = Some(inner.parse::<Type>()?);
                    if !inner.is_empty() {
                        inner.parse::<Token![,]>()?;
                        let kw = inner.call(Ident::parse_any)?;
                        if kw != "override" {
                            return Err(syn::Error::new(kw.span(), "expected override(..)"));
                        }
                        global_overrides = parse_overrides(&inner)?;
                    }
                }
                _ => return Err(syn::Error::new(item.span(), "expected levels(..) or global(..)")),
            }
            if content.is_empty() {
                break;
            }
            content.parse::<Token![,]>()?;
        }
        let levels = levels.ok_or_else(|| syn::Error::new(kw.span(), "a command declares its levels(..)"))?;
        let global = global.ok_or_else(|| {
            syn::Error::new(kw.span(), "a command declares the global options it receives: global(crate::GlobalArgs)")
        })?;
        Ok(Command { levels, global, global_overrides })
    })
}

/// Whether clap flattens this field: `#[command(flatten)]` or `#[clap(flatten)]`.
fn is_flatten(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|a| {
        (a.path().is_ident("command") || a.path().is_ident("clap"))
            && a.meta.require_list().is_ok_and(|l| {
                l.tokens.clone().into_iter().any(|t| matches!(t, proc_macro2::TokenTree::Ident(i) if i == "flatten"))
            })
    })
}

/// One part of the inventory, in declaration order.
enum Part {
    /// A run of the struct's own fields.
    Own(Vec<(String, TokenStream2)>),
    /// A flattened group, with the overrides at its flatten site.
    Group(Box<Type>, Vec<(Ident, Class)>),
}

fn inventory_entries(entries: &[(Ident, Class)]) -> TokenStream2 {
    let items = entries.iter().map(|(n, c)| {
        let n = n.unraw().to_string();
        let c = c.tokens();
        quote!((#n, #c))
    });
    quote!(&[#(#items),*])
}

/// The tokens of a group's inventory with the overrides applied, as a constant
/// named `name`, and the assertion that each override names a field of the
/// group.
fn group_part(name: &Ident, ty: &Type, overrides: &[(Ident, Class)]) -> (TokenStream2, TokenStream2, TokenStream2) {
    let fields = quote_spanned!(ty.span()=> <#ty as crate::input_class::InputFields>::FIELDS);
    if overrides.is_empty() {
        return (quote!(), fields, quote!());
    }
    let ov = inventory_entries(overrides);
    let def = quote! {
        const #name: [(&str, crate::input_class::InputClass); #fields.len()] =
            crate::input_class::derive_support::overridden::<{ #fields.len() }>(#fields, #ov);
    };
    let checks = overrides.iter().map(|(n, _)| {
        let s = n.unraw().to_string();
        let msg = format!("#[input(override)] names `{s}`, which is not an input of `{}`", quote!(#ty));
        quote_spanned!(n.span()=> assert!(crate::input_class::derive_support::has_field(#fields, #s), #msg);)
    });
    (def, quote!(&#name), quote!(#(#checks)*))
}

fn expand(input: &DeriveInput) -> syn::Result<TokenStream2> {
    let ident = &input.ident;
    if !input.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(&input.generics, "Inputs cannot be derived for a generic type"));
    }
    let fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(f) => &f.named,
            _ => return Err(syn::Error::new_spanned(ident, "Inputs is derived for a struct with named fields")),
        },
        _ => return Err(syn::Error::new_spanned(ident, "Inputs is derived for a struct with named fields")),
    };

    let mut errors: Option<syn::Error> = None;
    let mut push_err = |e: syn::Error| match &mut errors {
        Some(all) => all.combine(e),
        None => errors = Some(e),
    };

    let mut parts: Vec<Part> = Vec::new();
    for field in fields {
        let name = field.ident.as_ref().expect("named field");
        let attrs: Vec<&Attribute> = field.attrs.iter().filter(|a| a.path().is_ident("input")).collect();
        let mut parsed = Vec::new();
        for a in &attrs {
            match parse_field_attr(a) {
                Ok(p) => parsed.push(p),
                Err(e) => push_err(e),
            }
        }
        if attrs.len() > 1 {
            push_err(syn::Error::new_spanned(attrs[1], "one #[input(..)] per field"));
            continue;
        }
        if is_flatten(&field.attrs) {
            match parsed.pop() {
                None => parts.push(Part::Group(Box::new(field.ty.clone()), Vec::new())),
                Some(FieldAttr::Override(ov)) => parts.push(Part::Group(Box::new(field.ty.clone()), ov)),
                Some(FieldAttr::Class(_)) => push_err(syn::Error::new_spanned(
                    attrs[0],
                    format!(
                        "`{name}` is flattened: its fields take their classes from `{}`'s own \
                         #[input(..)], changed here with #[input(override(field = class))]",
                        quote!(#field.ty)
                    ),
                )),
            }
            continue;
        }
        match parsed.pop() {
            Some(FieldAttr::Class(c)) => {
                let entry = (name.unraw().to_string(), c.tokens());
                match parts.last_mut() {
                    Some(Part::Own(v)) => v.push(entry),
                    _ => parts.push(Part::Own(vec![entry])),
                }
            }
            Some(FieldAttr::Override(_)) => push_err(syn::Error::new_spanned(
                attrs[0],
                format!("`{name}` is not flattened, so it has no fields to override: give it a class, one of {CLASSES}"),
            )),
            None if attrs.is_empty() => push_err(syn::Error::new_spanned(
                name,
                format!(
                    "`{ident}::{name}` has no input class. Every argument says what it may do to a \
                     stored result: add #[input(..)] with one of {CLASSES} (see cli::input_class)"
                ),
            )),
            None => {}
        }
    }

    let command = {
        let attrs: Vec<&Attribute> = input.attrs.iter().filter(|a| a.path().is_ident("input")).collect();
        if attrs.len() > 1 {
            push_err(syn::Error::new_spanned(attrs[1], "one #[input(command(..))] per struct"));
        }
        match attrs.first().map(|a| parse_command(a)) {
            Some(Ok(c)) => Some(c),
            Some(Err(e)) => {
                push_err(e);
                None
            }
            None => None,
        }
    };

    if let Some(e) = errors {
        return Err(e);
    }

    let mut defs = Vec::new();
    let mut refs = Vec::new();
    let mut checks = Vec::new();
    for (i, part) in parts.iter().enumerate() {
        match part {
            Part::Own(entries) => {
                let items = entries.iter().map(|(n, c)| quote!((#n, #c)));
                refs.push(quote!(&[#(#items),*]));
            }
            Part::Group(ty, ov) => {
                let (d, r, c) = group_part(&format_ident!("__P{i}"), ty, ov);
                defs.push(d);
                refs.push(r);
                checks.push(c);
            }
        }
    }

    let fields_impl = quote! {
        impl crate::input_class::InputFields for #ident {
            const FIELDS: crate::input_class::Inventory = {
                #(#defs)*
                const __PARTS: &[crate::input_class::Inventory] = &[#(#refs),*];
                const __N: usize = crate::input_class::derive_support::total_len(__PARTS);
                const __ALL: [(&str, crate::input_class::InputClass); __N] =
                    crate::input_class::derive_support::concat::<__N>(__PARTS);
                &__ALL
            };
        }
    };

    let command_impl = match &command {
        None => quote!(),
        Some(Command { levels, global, global_overrides }) => {
            let (gdef, gref, gcheck) = group_part(&format_ident!("__GLOBAL"), global, global_overrides);
            checks.push(gcheck);
            let levels_tokens = levels.iter().map(|l| {
                let v = Ident::new(&capitalize(&l.unraw().to_string()), l.span());
                quote_spanned!(l.span()=> crate::input_class::Level::#v)
            });
            let name = ident.to_string();
            quote! {
                impl #ident {
                    /// Every input of this command, by clap argument id, with its
                    /// class: the struct's own fields, its flattened groups' and
                    /// the global options (`#[derive(Inputs)]`).
                    #[allow(dead_code)]
                    pub const INPUTS: crate::input_class::Inventory = {
                        #gdef
                        const __PARTS: &[crate::input_class::Inventory] =
                            &[<#ident as crate::input_class::InputFields>::FIELDS, #gref];
                        const __N: usize = crate::input_class::derive_support::total_len(__PARTS);
                        const __ALL: [(&str, crate::input_class::InputClass); __N] =
                            crate::input_class::derive_support::concat::<__N>(__PARTS);
                        &__ALL
                    };
                    /// The levels of this command's leaf: a `keyed` input names one.
                    #[allow(dead_code)]
                    pub const LEVELS: &'static [crate::input_class::Level] = &[#(#levels_tokens),*];
                }
                const _: () = crate::input_class::derive_support::check_command(#name, #ident::INPUTS, #ident::LEVELS);
            }
        }
    };

    Ok(quote! {
        #fields_impl
        #command_impl
        const _: () = { #(#checks)* };
    })
}
