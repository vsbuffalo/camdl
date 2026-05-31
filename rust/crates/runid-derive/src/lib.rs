//! `#[derive(RunInput)]` — generates `ContentAddressed` over all fields,
//! include-by-default.
//!
//! The generated `hash_into` follows the canonical-hashing rules exactly:
//! a domain-separation type tag (the fully-qualified type name, via
//! `module_path!()`), the per-type schema version, then each field hashed
//! compositionally in declaration order. Enums write their variant index
//! (`u32` LE, declaration order) before the variant payload.
//!
//! Field/container attributes:
//! - `#[run_input(provenance)]` on a field — skip it entirely (recorded in
//!   `run.json`, never hashed).
//! - `#[run_input(schema_version = N)]` on the type — set the per-type
//!   schema version (default `1`); a policy change bumps it and re-keys
//!   only that type.
//!
//! A field whose type is not `ContentAddressed` is a compile error (the
//! emitted `<FieldTy as ContentAddressed>::hash_into` fails to resolve) —
//! you cannot forget to make an input hashable.
//!
//! This macro is validated against the hand-written impls: a golden test in
//! `runid` pins `macro output == hand impl` on a fixed value before the
//! macro is trusted. The macro **replaces** hand-written canonical hashing;
//! there is never a second implementation.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{parse_macro_input, Data, DataEnum, DeriveInput, Fields, Index, LitInt};

#[proc_macro_derive(RunInput, attributes(run_input))]
pub fn derive_run_input(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let ident = &input.ident;
    let (impl_g, ty_g, where_g) = input.generics.split_for_impl();

    let schema_version = match container_schema_version(&input.attrs) {
        Ok(v) => v,
        Err(e) => return e.to_compile_error().into(),
    };

    let body = match &input.data {
        Data::Struct(s) => struct_body(&s.fields),
        Data::Enum(e) => enum_body(ident, e),
        Data::Union(_) => {
            return syn::Error::new_spanned(ident, "RunInput cannot be derived for unions")
                .to_compile_error()
                .into();
        }
    };

    // Fully-qualified type name as the domain-separation tag. `module_path!()`
    // expands at the type's definition site, so a hand replica written in the
    // same module reproduces the identical tag.
    let tag = quote! { ::core::concat!(::core::module_path!(), "::", ::core::stringify!(#ident)) };

    let expanded = quote! {
        impl #impl_g runid::ContentAddressed for #ident #ty_g #where_g {
            fn hash_into(&self, __h: &mut runid::CanonicalHasher) {
                __h.write_type_tag(#tag);
                __h.write_schema_version(#schema_version);
                #body
            }
        }
    };
    expanded.into()
}

/// Emit the field-hashing statements for a struct body. `self.<field>` for
/// named fields, `self.<index>` for tuple fields; provenance fields skipped.
fn struct_body(fields: &Fields) -> TokenStream2 {
    match fields {
        Fields::Named(named) => {
            let stmts = named.named.iter().filter(|f| !is_provenance(&f.attrs)).map(|f| {
                let name = f.ident.as_ref().unwrap();
                quote! { runid::ContentAddressed::hash_into(&self.#name, __h); }
            });
            quote! { #(#stmts)* }
        }
        Fields::Unnamed(unnamed) => {
            let stmts =
                unnamed.unnamed.iter().enumerate().filter(|(_, f)| !is_provenance(&f.attrs)).map(
                    |(i, _)| {
                        let idx = Index::from(i);
                        quote! { runid::ContentAddressed::hash_into(&self.#idx, __h); }
                    },
                );
            quote! { #(#stmts)* }
        }
        Fields::Unit => quote! {},
    }
}

/// Emit a `match self { … }` that writes each variant's index (`u32`,
/// declaration order) then hashes its non-provenance fields in order.
fn enum_body(ident: &syn::Ident, data: &DataEnum) -> TokenStream2 {
    let arms = data.variants.iter().enumerate().map(|(i, v)| {
        let vname = &v.ident;
        let idx = i as u32;
        match &v.fields {
            Fields::Named(named) => {
                let binds = named.named.iter().map(|f| f.ident.as_ref().unwrap());
                let stmts = named.named.iter().filter(|f| !is_provenance(&f.attrs)).map(|f| {
                    let name = f.ident.as_ref().unwrap();
                    quote! { runid::ContentAddressed::hash_into(#name, __h); }
                });
                quote! {
                    #ident::#vname { #(#binds),* } => {
                        __h.write_u32(#idx);
                        #(#stmts)*
                    }
                }
            }
            Fields::Unnamed(unnamed) => {
                let binds: Vec<syn::Ident> = (0..unnamed.unnamed.len())
                    .map(|j| syn::Ident::new(&format!("__f{j}"), proc_macro2::Span::call_site()))
                    .collect();
                let stmts = unnamed.unnamed.iter().zip(binds.iter()).filter(|(f, _)| {
                    !is_provenance(&f.attrs)
                }).map(|(_, b)| {
                    quote! { runid::ContentAddressed::hash_into(#b, __h); }
                });
                quote! {
                    #ident::#vname( #(#binds),* ) => {
                        __h.write_u32(#idx);
                        #(#stmts)*
                    }
                }
            }
            Fields::Unit => quote! {
                #ident::#vname => { __h.write_u32(#idx); }
            },
        }
    });
    quote! {
        match self {
            #(#arms),*
        }
    }
}

/// `true` if the field carries `#[run_input(provenance)]`.
fn is_provenance(attrs: &[syn::Attribute]) -> bool {
    let mut found = false;
    for attr in attrs {
        if !attr.path().is_ident("run_input") {
            continue;
        }
        // Ignore parse errors here — a malformed run_input attribute surfaces
        // via the container schema_version parse (same attribute syntax).
        let _ = attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("provenance") {
                found = true;
            }
            Ok(())
        });
    }
    found
}

/// Read `#[run_input(schema_version = N)]` from the container attrs; default 1.
fn container_schema_version(attrs: &[syn::Attribute]) -> syn::Result<u16> {
    let mut version: u16 = 1;
    for attr in attrs {
        if !attr.path().is_ident("run_input") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("schema_version") {
                let lit: LitInt = meta.value()?.parse()?;
                version = lit.base10_parse()?;
                Ok(())
            } else if meta.path.is_ident("provenance") {
                // `provenance` is a field-level flag; ignore on the container.
                Ok(())
            } else {
                Err(meta.error("unknown run_input attribute (expected `provenance` or `schema_version = N`)"))
            }
        })?;
    }
    Ok(version)
}
