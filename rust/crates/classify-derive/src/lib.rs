//! `#[derive(Classify)]` — every input of a command surface carries one class.
//!
//! Derived on a clap `Args` struct (or any struct of inputs, such as the
//! environment module's), it requires each field to carry exactly one
//! `#[class(...)]` declaration and emits the struct's inventory of
//! `(input, class)` pairs. A field without one does not compile. It generates no
//! configuration types and carries no test parameters
//! (`docs/dev/proposals/2026-11-17-honest-input-addressed-store.md`, phase 3
//! "Classification").
//!
//! The classes:
//!
//! - `keyed(model|config|params|scenario|seed|obs)` — can change the result; it
//!   is part of the resolved configuration at that identity level.
//! - `execution` — changes how the result is computed, not what (`--parallel`).
//! - `provenance` — how the run was asked for (`--label`).
//! - `presentation` — how output is written outside the store (`-o`).
//! - `expands` — defines the set of runs (`--seeds`, `--sweep`).
//! - `removed` — refuses an old spelling with a migration message.
//!
//! A field marked `#[command(flatten)]` takes no class of its own: its inventory
//! is the flattened struct's, which must derive `Classify` too, so a shared
//! group is classified once, where it is declared. `command` is registered as
//! a helper attribute of this derive as well as clap's, so the struct of a
//! test that does not derive clap's `Args` may carry it too.
//!
//! The generated code names `crate::input_class::{Classify, Input, InputClass,
//! Level}`: the derive is used inside the `cli` crate, which defines them. A
//! proc-macro crate cannot export those types itself.

use proc_macro::TokenStream;
use proc_macro2::{TokenStream as TokenStream2, TokenTree};
use quote::quote;
use syn::{parse_macro_input, Attribute, Data, DeriveInput, Fields, Meta};

#[proc_macro_derive(Classify, attributes(class, command))]
pub fn derive_classify(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match expand(&input) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn expand(input: &DeriveInput) -> syn::Result<TokenStream2> {
    let ident = &input.ident;
    let (impl_g, ty_g, where_g) = input.generics.split_for_impl();
    let fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(n) => &n.named,
            _ => {
                return Err(syn::Error::new_spanned(
                    ident,
                    "Classify needs a struct with named fields: an input is named by its field",
                ))
            }
        },
        _ => return Err(syn::Error::new_spanned(ident, "Classify can only be derived for a struct")),
    };

    let mut pushes = Vec::new();
    let mut errors: Option<syn::Error> = None;
    let mut add_error = |e: syn::Error| match errors.as_mut() {
        Some(acc) => acc.combine(e),
        None => errors = Some(e),
    };
    for f in fields {
        let name = f.ident.as_ref().expect("named field");
        let name_str = name.to_string();
        let classes: Vec<&Attribute> = f.attrs.iter().filter(|a| a.path().is_ident("class")).collect();
        let flattened = f.attrs.iter().any(is_flatten);
        if flattened {
            if let Some(a) = classes.first() {
                add_error(syn::Error::new_spanned(
                    a,
                    format!(
                        "`{name_str}` is flattened: its inputs are classified where its struct is \
                         declared, so it takes no #[class(...)] here"
                    ),
                ));
                continue;
            }
            let ty = &f.ty;
            pushes.push(quote! {
                __v.extend(<#ty as crate::input_class::Classify>::inventory());
            });
            continue;
        }
        match classes.as_slice() {
            [] => add_error(syn::Error::new_spanned(
                name,
                format!(
                    "input `{name_str}` has no class: mark it #[class(keyed(<level>))], \
                     #[class(execution)], #[class(provenance)], #[class(presentation)], \
                     #[class(expands)] or #[class(removed)]"
                ),
            )),
            [one] => match parse_class(one) {
                Ok(class) => pushes.push(quote! {
                    __v.push(crate::input_class::Input { field: #name_str, class: #class });
                }),
                Err(e) => add_error(e),
            },
            [_, second, ..] => add_error(syn::Error::new_spanned(
                second,
                format!("input `{name_str}` has more than one class; it must have exactly one"),
            )),
        }
    }
    if let Some(e) = errors {
        return Err(e);
    }

    Ok(quote! {
        impl #impl_g crate::input_class::Classify for #ident #ty_g #where_g {
            fn inventory() -> ::std::vec::Vec<crate::input_class::Input> {
                let mut __v = ::std::vec::Vec::new();
                #(#pushes)*
                __v
            }
        }
    })
}

/// Whether `attr` is clap's `#[command(flatten)]` (or the older `#[clap(flatten)]`).
fn is_flatten(attr: &Attribute) -> bool {
    if !(attr.path().is_ident("command") || attr.path().is_ident("clap")) {
        return false;
    }
    let Meta::List(list) = &attr.meta else { return false };
    list.tokens.clone().into_iter().any(|t| matches!(t, TokenTree::Ident(i) if i == "flatten"))
}

/// The `InputClass` expression for one `#[class(...)]` attribute.
fn parse_class(attr: &Attribute) -> syn::Result<TokenStream2> {
    let mut out: Option<TokenStream2> = None;
    attr.parse_nested_meta(|meta| {
        if out.is_some() {
            return Err(meta.error("one class per input"));
        }
        let path = &meta.path;
        let class = if path.is_ident("keyed") {
            let mut level: Option<TokenStream2> = None;
            meta.parse_nested_meta(|lv| {
                let variant = match lv.path.get_ident().map(|i| i.to_string()).as_deref() {
                    Some("model") => quote!(Model),
                    Some("config") => quote!(Config),
                    Some("params") => quote!(Params),
                    Some("scenario") => quote!(Scenario),
                    Some("seed") => quote!(Seed),
                    Some("obs") => quote!(Obs),
                    _ => {
                        return Err(lv.error(
                            "unknown identity level: expected model, config, params, scenario, \
                             seed or obs",
                        ))
                    }
                };
                if level.replace(variant).is_some() {
                    return Err(lv.error("keyed(...) names one level"));
                }
                Ok(())
            })?;
            let Some(level) = level else {
                return Err(meta.error("keyed(...) must name its level, e.g. keyed(config)"));
            };
            quote!(crate::input_class::InputClass::Keyed(crate::input_class::Level::#level))
        } else if path.is_ident("execution") {
            quote!(crate::input_class::InputClass::Execution)
        } else if path.is_ident("provenance") {
            quote!(crate::input_class::InputClass::Provenance)
        } else if path.is_ident("presentation") {
            quote!(crate::input_class::InputClass::Presentation)
        } else if path.is_ident("expands") {
            quote!(crate::input_class::InputClass::Expands)
        } else if path.is_ident("removed") {
            quote!(crate::input_class::InputClass::Removed)
        } else {
            return Err(meta.error(
                "unknown class: expected keyed(<level>), execution, provenance, presentation, \
                 expands or removed",
            ));
        };
        out = Some(class);
        Ok(())
    })?;
    out.ok_or_else(|| syn::Error::new_spanned(attr, "#[class(...)] is empty; name one class"))
}
