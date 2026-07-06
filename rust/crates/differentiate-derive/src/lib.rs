//! `#[derive(Differentiate)]` — generates `Differentiable::diffables` over all
//! fields, **include-by-default** (the same technique as `runid-derive`'s
//! `#[derive(RunInput)]`).
//!
//! The generated `diffables` returns `(field_name, &Diffable)` for every field
//! in declaration order, so adding a differentiable argument to a likelihood
//! automatically enters every consumer that iterates `diffables()` — coverage is
//! a property of the type, not a hand-written pass (proposal
//! `2026-07-06-seal-differentiation-coverage-3b.md` §4.2).
//!
//! Attributes:
//! - `#[differentiate(skip)]` on a field — exclude it (a θ-independent field such
//!   as a Binomial/BetaBinomial `n`, which carries no gradient).
//!
//! A field that is **not** skipped and whose type is not [`crate::Diffable`] is a
//! **compile error** (the emitted `&self.f` cannot coerce to a `&Diffable` slot).
//! This is deliberate: a new argument accidentally typed `Expr` rather than
//! `Diffable` is rejected loudly, never silently dropped — the silent-miss this
//! derive exists to prevent. For that reason `Expr` must **not** implement
//! `Differentiable`.
//!
//! Scope: this derive is used only within the `ir` crate (the likelihood structs
//! and the `Likelihood` enum live there), so the generated impls reference the
//! trait and type via `crate::` paths, resolved at the `ir` derive site.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{parse_macro_input, Data, DataEnum, DataStruct, DeriveInput, Fields};

#[proc_macro_derive(Differentiate, attributes(differentiate))]
pub fn derive_differentiate(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let ident = &input.ident;

    let body = match &input.data {
        Data::Struct(s) => match struct_body(s) {
            Ok(b) => b,
            Err(e) => return e.to_compile_error().into(),
        },
        Data::Enum(e) => match enum_body(ident, e) {
            Ok(b) => b,
            Err(e) => return e.to_compile_error().into(),
        },
        Data::Union(_) => {
            return syn::Error::new_spanned(ident, "Differentiate cannot be derived for unions")
                .to_compile_error()
                .into();
        }
    };

    quote! {
        impl crate::Differentiable for #ident {
            fn diffables(&self) -> ::std::vec::Vec<(&'static str, &crate::Diffable)> {
                #body
            }
        }
    }
    .into()
}

/// A struct folds every non-`skip` named field into a `(name, &self.field)` pair.
fn struct_body(s: &DataStruct) -> syn::Result<TokenStream2> {
    let named = match &s.fields {
        Fields::Named(named) => named,
        _ => {
            return Err(syn::Error::new_spanned(
                &s.fields,
                "Differentiate requires a struct with named fields",
            ))
        }
    };
    let pushes = named
        .named
        .iter()
        .filter(|f| !is_skip(&f.attrs))
        .map(|f| {
            let name = f.ident.as_ref().unwrap();
            quote! { __v.push((::core::stringify!(#name), &self.#name)); }
        });
    Ok(quote! {
        let mut __v = ::std::vec::Vec::new();
        #(#pushes)*
        __v
    })
}

/// An enum delegates to the single-field tuple payload of the active variant
/// (`Likelihood::NegBinomial(l) => l.diffables()`). A new variant without an arm
/// is a compile error (exhaustive `match`), so the enum stays sealed too.
fn enum_body(ident: &syn::Ident, data: &DataEnum) -> syn::Result<TokenStream2> {
    let arms = data
        .variants
        .iter()
        .map(|v| {
            let vname = &v.ident;
            match &v.fields {
                Fields::Unnamed(u) if u.unnamed.len() == 1 => Ok(quote! {
                    #ident::#vname(__inner) => crate::Differentiable::diffables(__inner),
                }),
                _ => Err(syn::Error::new_spanned(
                    v,
                    "Differentiate on an enum requires every variant to be a \
                     single-field tuple wrapping a Differentiable struct",
                )),
            }
        })
        .collect::<syn::Result<Vec<_>>>()?;
    Ok(quote! {
        match self {
            #(#arms)*
        }
    })
}

/// `true` if the field carries `#[differentiate(skip)]`.
fn is_skip(attrs: &[syn::Attribute]) -> bool {
    let mut found = false;
    for attr in attrs {
        if !attr.path().is_ident("differentiate") {
            continue;
        }
        let _ = attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("skip") {
                found = true;
            }
            Ok(())
        });
    }
    found
}
