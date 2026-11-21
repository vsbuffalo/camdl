//! Derives for the `store` crate: `Identity`, `Inputs`, `FileInputs` and
//! `EnvInputs`.
//!
//! `#[derive(Identity)]` hashes a struct or enum through `store`'s one
//! encoder: the tag the type states and its version, then each field's name
//! and bytes in name order (an enum writes its variant's name first). It
//! generates the record form, `<Name>Record`, with each field's type replaced
//! by that type's record, which hashes to the same bytes.
//!
//! The input derives give every field a class, refuse a field without one,
//! and split the struct by class into generated structs, one per destination.
//! See the `store` crate's `inputs` module for what each class means.

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{format_ident, quote};
use syn::parse::ParseStream;
use syn::{parse_macro_input, Data, DeriveInput, Fields, Ident, LitInt, LitStr, Path, Type};

// ── Identity ─────────────────────────────────────────────────────────────────

#[proc_macro_derive(Identity, attributes(identity))]
pub fn derive_identity(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    identity_impl(&input).unwrap_or_else(|e| e.to_compile_error()).into()
}

fn identity_attr(input: &DeriveInput) -> syn::Result<(LitStr, LitInt)> {
    let mut tag = None;
    let mut version = None;
    for attr in input.attrs.iter().filter(|a| a.path().is_ident("identity")) {
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("tag") {
                tag = Some(meta.value()?.parse::<LitStr>()?);
                Ok(())
            } else if meta.path.is_ident("version") {
                version = Some(meta.value()?.parse::<LitInt>()?);
                Ok(())
            } else {
                Err(meta.error("expected `tag = \"...\"` or `version = N`"))
            }
        })?;
    }
    match (tag, version) {
        (Some(t), Some(v)) => Ok((t, v)),
        _ => Err(syn::Error::new_spanned(
            &input.ident,
            "#[derive(Identity)] needs #[identity(tag = \"...\", version = N)]",
        )),
    }
}

fn identity_impl(input: &DeriveInput) -> syn::Result<TokenStream2> {
    if !input.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(&input.generics, "#[derive(Identity)] does not support generic types"));
    }
    let (tag, version) = identity_attr(input)?;
    let name = &input.ident;
    let record = format_ident!("{}Record", name);
    let vis = &input.vis;
    let serde_crate = quote!(#[serde(crate = "::store::__private::serde")]);

    match &input.data {
        Data::Struct(s) => {
            let fields = named_or_unit(&s.fields, name)?;
            let mut sorted: Vec<&syn::Field> = fields.to_vec();
            sorted.sort_by_key(|f| f.ident.as_ref().unwrap().to_string());
            let hash_fields = sorted.iter().map(|f| {
                let id = f.ident.as_ref().unwrap();
                let s = id.to_string();
                quote! { h.write_str(#s); ::store::ContentAddressed::hash_into(&self.#id, h); }
            });
            let hash_fields2 = hash_fields.clone();
            let rec_fields = fields.iter().map(|f| {
                let (id, ty, fvis) = (&f.ident, &f.ty, &f.vis);
                quote! { #fvis #id: <#ty as ::store::Identity>::Record }
            });
            let to_rec = fields.iter().map(|f| {
                let id = &f.ident;
                quote! { #id: ::store::Identity::record(&self.#id) }
            });
            let held = fields.iter().map(|f| {
                let id = &f.ident;
                quote! { v.extend(::store::Identity::held_content(&self.#id)); }
            });
            let deps = fields.iter().map(|f| {
                let id = &f.ident;
                quote! { v.extend(::store::Identity::deps(&self.#id)); }
            });
            Ok(quote! {
                impl ::store::ContentAddressed for #name {
                    fn hash_into(&self, h: &mut ::store::CanonicalHasher) {
                        h.write_tag(#tag, #version);
                        #(#hash_fields)*
                    }
                }

                /// The record form of the identity of the same name: each
                /// field's type replaced by its record. Hashes as the identity does.
                #[derive(::core::clone::Clone, ::core::fmt::Debug, ::core::cmp::PartialEq,
                         ::store::__private::serde::Serialize, ::store::__private::serde::Deserialize)]
                #serde_crate
                #vis struct #record { #(#rec_fields,)* }

                impl ::store::ContentAddressed for #record {
                    fn hash_into(&self, h: &mut ::store::CanonicalHasher) {
                        h.write_tag(#tag, #version);
                        #(#hash_fields2)*
                    }
                }

                impl ::store::Identity for #record {
                    type Record = #record;
                    fn record(&self) -> #record { ::core::clone::Clone::clone(self) }
                    fn held_content(&self) -> ::std::vec::Vec<::store::Key> { ::std::vec::Vec::new() }
                    fn deps(&self) -> ::std::vec::Vec<::store::DepInfo> { ::std::vec::Vec::new() }
                }

                impl ::store::Identity for #name {
                    type Record = #record;
                    fn record(&self) -> #record { #record { #(#to_rec,)* } }
                    fn held_content(&self) -> ::std::vec::Vec<::store::Key> {
                        let mut v = ::std::vec::Vec::new();
                        #(#held)*
                        v
                    }
                    fn deps(&self) -> ::std::vec::Vec<::store::DepInfo> {
                        let mut v = ::std::vec::Vec::new();
                        #(#deps)*
                        v
                    }
                }
            })
        }
        Data::Enum(e) => {
            let mut hash_arms = Vec::new();
            let mut rec_variants = Vec::new();
            let mut to_rec_arms = Vec::new();
            let mut held_arms = Vec::new();
            let mut deps_arms = Vec::new();
            for v in &e.variants {
                let vname = &v.ident;
                let vstr = vname.to_string();
                match &v.fields {
                    Fields::Unit => {
                        hash_arms.push(quote! { Self::#vname => { h.write_tag(#tag, #version); h.write_str(#vstr); } });
                        rec_variants.push(quote! { #vname });
                        to_rec_arms.push(quote! { #name::#vname => #record::#vname });
                        held_arms.push(quote! { #name::#vname => {} });
                        deps_arms.push(quote! { #name::#vname => {} });
                    }
                    Fields::Named(n) => {
                        let ids: Vec<&Ident> = n.named.iter().map(|f| f.ident.as_ref().unwrap()).collect();
                        let mut sorted = ids.clone();
                        sorted.sort_by_key(|i| i.to_string());
                        let hs = sorted.iter().map(|i| {
                            let s = i.to_string();
                            quote! { h.write_str(#s); ::store::ContentAddressed::hash_into(#i, h); }
                        });
                        hash_arms.push(quote! { Self::#vname { #(#ids,)* } => { h.write_tag(#tag, #version); h.write_str(#vstr); #(#hs)* } });
                        let rf = n.named.iter().map(|f| {
                            let (id, ty) = (&f.ident, &f.ty);
                            quote! { #id: <#ty as ::store::Identity>::Record }
                        });
                        rec_variants.push(quote! { #vname { #(#rf,)* } });
                        to_rec_arms.push(quote! { #name::#vname { #(#ids,)* } => #record::#vname { #(#ids: ::store::Identity::record(#ids),)* } });
                        held_arms.push(quote! { #name::#vname { #(#ids,)* } => { #(v.extend(::store::Identity::held_content(#ids));)* } });
                        deps_arms.push(quote! { #name::#vname { #(#ids,)* } => { #(v.extend(::store::Identity::deps(#ids));)* } });
                    }
                    Fields::Unnamed(u) => {
                        let ids: Vec<Ident> = (0..u.unnamed.len()).map(|i| format_ident!("__f{}", i)).collect();
                        hash_arms.push(quote! { Self::#vname ( #(#ids,)* ) => { h.write_tag(#tag, #version); h.write_str(#vstr); #(::store::ContentAddressed::hash_into(#ids, h);)* } });
                        let rf = u.unnamed.iter().map(|f| {
                            let ty = &f.ty;
                            quote! { <#ty as ::store::Identity>::Record }
                        });
                        rec_variants.push(quote! { #vname ( #(#rf,)* ) });
                        to_rec_arms.push(quote! { #name::#vname ( #(#ids,)* ) => #record::#vname ( #(::store::Identity::record(#ids),)* ) });
                        held_arms.push(quote! { #name::#vname ( #(#ids,)* ) => { #(v.extend(::store::Identity::held_content(#ids));)* } });
                        deps_arms.push(quote! { #name::#vname ( #(#ids,)* ) => { #(v.extend(::store::Identity::deps(#ids));)* } });
                    }
                }
            }
            let rec_hash_arms: Vec<TokenStream2> = hash_arms.clone();
            Ok(quote! {
                impl ::store::ContentAddressed for #name {
                    fn hash_into(&self, h: &mut ::store::CanonicalHasher) {
                        match self { #(#hash_arms)* }
                    }
                }

                /// The record form of the identity of the same name: each
                /// field's type replaced by its record. Hashes as the identity does.
                #[derive(::core::clone::Clone, ::core::fmt::Debug, ::core::cmp::PartialEq,
                         ::store::__private::serde::Serialize, ::store::__private::serde::Deserialize)]
                #serde_crate
                #vis enum #record { #(#rec_variants,)* }

                impl ::store::ContentAddressed for #record {
                    fn hash_into(&self, h: &mut ::store::CanonicalHasher) {
                        match self { #(#rec_hash_arms)* }
                    }
                }

                impl ::store::Identity for #record {
                    type Record = #record;
                    fn record(&self) -> #record { ::core::clone::Clone::clone(self) }
                    fn held_content(&self) -> ::std::vec::Vec<::store::Key> { ::std::vec::Vec::new() }
                    fn deps(&self) -> ::std::vec::Vec<::store::DepInfo> { ::std::vec::Vec::new() }
                }

                impl ::store::Identity for #name {
                    type Record = #record;
                    fn record(&self) -> #record { match self { #(#to_rec_arms,)* } }
                    #[allow(unused_variables)]
                    fn held_content(&self) -> ::std::vec::Vec<::store::Key> {
                        let mut v = ::std::vec::Vec::new();
                        match self { #(#held_arms)* }
                        v
                    }
                    #[allow(unused_variables)]
                    fn deps(&self) -> ::std::vec::Vec<::store::DepInfo> {
                        let mut v = ::std::vec::Vec::new();
                        match self { #(#deps_arms)* }
                        v
                    }
                }
            })
        }
        Data::Union(_) => Err(syn::Error::new_spanned(name, "#[derive(Identity)] does not support unions")),
    }
}

fn named_or_unit<'a>(fields: &'a Fields, name: &Ident) -> syn::Result<Vec<&'a syn::Field>> {
    match fields {
        Fields::Named(n) => Ok(n.named.iter().collect()),
        Fields::Unit => Ok(Vec::new()),
        Fields::Unnamed(_) => Err(syn::Error::new_spanned(name, "this derive needs named fields")),
    }
}

// ── Input classes ────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Class {
    Identity,
    Expands,
    Adds,
    Execution,
    Export,
    Render,
    Store,
    Recorded,
    Group,
    Skip,
}

impl Class {
    fn parse(id: &Ident) -> Option<Class> {
        Some(match id.to_string().as_str() {
            "identity" => Class::Identity,
            "expands" => Class::Expands,
            "adds" => Class::Adds,
            "execution" => Class::Execution,
            "export" => Class::Export,
            "render" => Class::Render,
            "store" => Class::Store,
            "recorded" => Class::Recorded,
            "group" => Class::Group,
            "skip" => Class::Skip,
            _ => return None,
        })
    }

    fn tokens(self) -> TokenStream2 {
        match self {
            Class::Identity => quote!(::store::Class::Identity),
            Class::Expands => quote!(::store::Class::Expands),
            Class::Adds => quote!(::store::Class::Adds),
            Class::Execution => quote!(::store::Class::Execution),
            Class::Export => quote!(::store::Class::Export),
            Class::Render => quote!(::store::Class::Render),
            Class::Store => quote!(::store::Class::Store),
            Class::Recorded => quote!(::store::Class::Recorded),
            Class::Skip => quote!(::store::Class::Skip),
            Class::Group => unreachable!("a group's fields are listed, not the group"),
        }
    }

    /// The generated part a field of this class goes into.
    fn part(self) -> Option<Part> {
        match self {
            Class::Identity | Class::Expands | Class::Adds => Some(Part::Declared),
            Class::Execution => Some(Part::Execution),
            Class::Export | Class::Render => Some(Part::Exports),
            Class::Store => Some(Part::Request),
            Class::Recorded | Class::Skip | Class::Group => None,
        }
    }

    fn recorded(self) -> bool {
        !matches!(self, Class::Store | Class::Skip | Class::Group)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Part {
    Declared,
    Execution,
    Exports,
    Request,
}

struct InputField {
    ident: Ident,
    ty: Type,
    class: Class,
    vary: Option<Path>,
    exercised_by: Option<Path>,
    /// The name the field is listed under: the clap id, or the settings key.
    key: String,
    /// `Some(inner)` when the field is an `Option<inner>` group.
    optional_group: Option<Type>,
}

fn last_segment(ty: &Type) -> Option<&syn::PathSegment> {
    match ty {
        Type::Path(p) => p.path.segments.last(),
        _ => None,
    }
}

fn option_inner(ty: &Type) -> Option<Type> {
    let seg = last_segment(ty)?;
    if seg.ident != "Option" {
        return None;
    }
    match &seg.arguments {
        syn::PathArguments::AngleBracketed(a) => match a.args.first()? {
            syn::GenericArgument::Type(t) => Some(t.clone()),
            _ => None,
        },
        _ => None,
    }
}

fn serde_rename(field: &syn::Field) -> syn::Result<Option<String>> {
    let mut out = None;
    for attr in field.attrs.iter().filter(|a| a.path().is_ident("serde")) {
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("rename") {
                out = Some(meta.value()?.parse::<LitStr>()?.value());
            } else if meta.input.peek(syn::Token![=]) {
                let _: syn::Expr = meta.value()?.parse()?;
            } else if meta.input.peek(syn::token::Paren) {
                let content;
                syn::parenthesized!(content in meta.input);
                let _: TokenStream2 = content.parse()?;
            }
            Ok(())
        })?;
    }
    Ok(out)
}

fn input_fields(input: &DeriveInput, file: bool, env: bool) -> syn::Result<Vec<InputField>> {
    let Data::Struct(s) = &input.data else {
        return Err(syn::Error::new_spanned(&input.ident, "an input derive needs a struct"));
    };
    let mut out = Vec::new();
    for f in named_or_unit(&s.fields, &input.ident)? {
        let ident = f.ident.clone().unwrap();
        let mut class = None;
        let mut vary = None;
        let mut exercised_by = None;
        for attr in f.attrs.iter().filter(|a| a.path().is_ident("input")) {
            attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("vary") {
                    vary = Some(meta.value()?.parse::<Path>()?);
                    return Ok(());
                }
                if meta.path.is_ident("exercised_by") {
                    exercised_by = Some(meta.value()?.parse::<Path>()?);
                    return Ok(());
                }
                let id = meta.path.get_ident().ok_or_else(|| meta.error("expected a class"))?;
                match Class::parse(id) {
                    Some(c) if class.is_none() => {
                        class = Some(c);
                        Ok(())
                    }
                    Some(_) => Err(meta.error("a field has one class")),
                    None => Err(meta.error(
                        "unknown class; expected identity, expands, adds, execution, export, render, store, recorded, group or skip",
                    )),
                }
            })?;
        }
        let Some(class) = class else {
            return Err(syn::Error::new_spanned(
                &ident,
                format!("the field `{ident}` has no #[input(..)] class: every input decides whether it can change a result"),
            ));
        };
        if class == Class::Execution && exercised_by.is_none() && !env {
            return Err(syn::Error::new_spanned(
                &ident,
                "an execution field needs `exercised_by = <fixture>`: a fixture on which it is used, \
                 so the check that it changes no result is not vacuous",
            ));
        }
        if class == Class::Execution && last_segment(&f.ty).is_some_and(|s| s.ident == "InputFile") {
            return Err(syn::Error::new_spanned(&f.ty, "an execution field may not be an InputFile: a file that is read is an input"));
        }
        if file && matches!(class, Class::Export | Class::Render) {
            return Err(syn::Error::new_spanned(&ident, "a settings key cannot be an export or render input"));
        }
        let key = if file { serde_rename(f)?.unwrap_or_else(|| ident.to_string()) } else { ident.to_string() };
        let optional_group = if class == Class::Group && file { option_inner(&f.ty) } else { None };
        out.push(InputField { ident, ty: f.ty.clone(), class, vary, exercised_by, key, optional_group });
    }
    Ok(out)
}

fn base_name(name: &Ident) -> String {
    let s = name.to_string();
    s.strip_suffix("Args").filter(|b| !b.is_empty()).map(str::to_string).unwrap_or(s)
}

fn path_text(p: &Path) -> String {
    quote!(#p).to_string().replace(' ', "")
}

/// One field of a generated part: its name, its type, and the expression
/// that builds it in the split.
type PartField = (Ident, TokenStream2, TokenStream2);

/// One generated part: its type, its struct definition, and how the split
/// builds it from the destructured fields.
struct PartGen {
    ty: TokenStream2,
    def: TokenStream2,
    build: TokenStream2,
}

fn gen_part(
    vis: &syn::Visibility,
    name: Ident,
    fields: &[PartField],
    public: bool,
    serialize: bool,
) -> PartGen {
    if fields.is_empty() {
        return PartGen { ty: quote!(()), def: quote!(), build: quote!(()) };
    }
    let ids: Vec<&Ident> = fields.iter().map(|(i, _, _)| i).collect();
    let tys: Vec<&TokenStream2> = fields.iter().map(|(_, t, _)| t).collect();
    let vals: Vec<&TokenStream2> = fields.iter().map(|(_, _, v)| v).collect();
    let derive = if serialize {
        quote! {
            #[derive(::store::__private::serde::Serialize)]
            #[serde(crate = "::store::__private::serde")]
        }
    } else {
        quote!()
    };
    let field_defs = if public {
        quote! { #(pub #ids: #tys,)* }
    } else {
        quote! { #(#ids: #tys,)* }
    };
    let into_parts = if public {
        quote!()
    } else {
        quote! {
            impl #name {
                /// Every field, in declaration order. The only way to reach them.
                pub fn into_parts(self) -> ( #(#tys,)* ) { ( #(self.#ids,)* ) }
            }
        }
    };
    PartGen {
        ty: quote!(#name),
        def: quote! {
            #derive
            #vis struct #name { #field_defs }
            #into_parts
        },
        build: quote! { #name { #(#ids: #vals,)* } },
    }
}

// ── Inputs ───────────────────────────────────────────────────────────────────

#[proc_macro_derive(Inputs, attributes(input))]
pub fn derive_inputs(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    inputs_impl(&input, false).unwrap_or_else(|e| e.to_compile_error()).into()
}

#[proc_macro_derive(FileInputs, attributes(input))]
pub fn derive_file_inputs(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    inputs_impl(&input, true).unwrap_or_else(|e| e.to_compile_error()).into()
}

fn inputs_impl(input: &DeriveInput, file: bool) -> syn::Result<TokenStream2> {
    if !input.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(&input.generics, "an input derive does not support generic types"));
    }
    let fields = input_fields(input, file, false)?;
    let name = &input.ident;
    let vis = &input.vis;
    let base = base_name(name);
    let trait_path = if file { quote!(::store::FileInputs) } else { quote!(::store::Inputs) };

    let mut parts: Vec<(Part, Vec<PartField>)> =
        vec![(Part::Declared, vec![]), (Part::Execution, vec![]), (Part::Exports, vec![]), (Part::Request, vec![])];
    let mut group_splits = Vec::new();
    let mut given_inserts = Vec::new();
    let mut field_infos = Vec::new();
    let mut varied_arms = Vec::new();
    let mut exercised_arms = Vec::new();

    for f in &fields {
        let id = &f.ident;
        let ty = &f.ty;
        let key = &f.key;
        if f.class == Class::Group {
            let gty = f.optional_group.as_ref().unwrap_or(ty);
            let gd = format_ident!("__{}_declared", id);
            let ge = format_ident!("__{}_execution", id);
            let gx = format_ident!("__{}_exports", id);
            let gr = format_ident!("__{}_request", id);
            let gg = format_ident!("__{}_given", id);
            let wrap = |t: TokenStream2| if f.optional_group.is_some() { quote!(::core::option::Option<#t>) } else { t };
            parts[0].1.push((id.clone(), wrap(quote!(<#gty as #trait_path>::Declared)), quote!(#gd)));
            parts[1].1.push((id.clone(), wrap(quote!(<#gty as #trait_path>::Execution)), quote!(#ge)));
            if !file {
                parts[2].1.push((id.clone(), quote!(<#gty as ::store::Inputs>::Exports), quote!(#gx)));
            }
            parts[3].1.push((id.clone(), wrap(quote!(<#gty as #trait_path>::Request)), quote!(#gr)));
            let split_one = if file {
                quote! { <#gty as ::store::FileInputs>::__split_table(__g) }
            } else {
                quote! {{ let (a, b, c, d, e) = <#gty as ::store::Inputs>::split(__g).into_parts(); (a, b, c, d, e) }}
            };
            group_splits.push(if file {
                if f.optional_group.is_some() {
                    quote! {
                        let (#gd, #ge, #gr, #gg) = match #id {
                            ::core::option::Option::Some(__g) => {
                                let (a, b, d, e) = #split_one;
                                (::core::option::Option::Some(a), ::core::option::Option::Some(b), ::core::option::Option::Some(d), ::core::option::Option::Some(e))
                            }
                            ::core::option::Option::None => (None, None, None, None),
                        };
                        if let ::core::option::Option::Some(g) = #gg { __given.__nest(#key, g); }
                    }
                } else {
                    quote! {
                        let __g = #id;
                        let (#gd, #ge, #gr, #gg) = #split_one;
                        __given.__nest(#key, #gg);
                    }
                }
            } else {
                quote! {
                    let __g = #id;
                    let (#gd, #ge, #gx, #gr, #gg) = #split_one;
                    __given.__nest(#key, #gg);
                }
            });
            field_infos.push(quote! { v.extend(::store::__private::__prefixed(#key, <#gty as #trait_path>::fields())); });
            let prefix = format!("{key}.");
            varied_arms.push(if f.optional_group.is_some() {
                quote! {
                    if let ::core::option::Option::Some(rest) = field.strip_prefix(#prefix) {
                        let mut c = ::core::clone::Clone::clone(self);
                        c.#id = self.#id.as_ref().map(|g| <#gty as #trait_path>::varied(g, rest));
                        return c;
                    }
                }
            } else {
                quote! {
                    if let ::core::option::Option::Some(rest) = field.strip_prefix(#prefix) {
                        let mut c = ::core::clone::Clone::clone(self);
                        c.#id = <#gty as #trait_path>::varied(&self.#id, rest);
                        return c;
                    }
                }
            });
            continue;
        }
        if let Some(part) = f.class.part() {
            let slot = parts.iter_mut().find(|(p, _)| *p == part).unwrap();
            slot.1.push((id.clone(), quote!(#ty), quote!(#id)));
        }
        if f.class.recorded() {
            given_inserts.push(quote! { __fields.insert(::std::string::String::from(#key), ::store::Given::__value(&#id)); });
        }
        let class = f.class.tokens();
        let vary_text = f.vary.as_ref().map(path_text);
        let ex_text = f.exercised_by.as_ref().map(path_text);
        let vary_tok = match &vary_text {
            Some(t) => quote!(::core::option::Option::Some(#t)),
            None => quote!(::core::option::Option::None),
        };
        let ex_tok = match &ex_text {
            Some(t) => quote!(::core::option::Option::Some(#t)),
            None => quote!(::core::option::Option::None),
        };
        field_infos.push(quote! { v.push(::store::FieldInfo { name: #key, class: #class, vary: #vary_tok, exercised_by: #ex_tok }); });
        if matches!(f.class, Class::Identity | Class::Render) {
            let second = match &f.vary {
                Some(p) => quote!(#p()),
                None => quote!(::store::Vary::vary(&self.#id)),
            };
            varied_arms.push(quote! {
                if field == #key {
                    let mut c = ::core::clone::Clone::clone(self);
                    c.#id = #second;
                    return c;
                }
            });
        }
        if f.class == Class::Execution {
            let p = f.exercised_by.as_ref().unwrap();
            exercised_arms.push(quote! { if field == #key { return #p(); } });
        }
    }

    let part_names = ["Declared", "Execution", "Exports", "Request"];
    let mut gens = Vec::new();
    for (i, (part, pf)) in parts.iter().enumerate() {
        if file && *part == Part::Exports {
            gens.push(None);
            continue;
        }
        let pname = Ident::new(&format!("{base}{}", part_names[i]), Span::call_site());
        gens.push(Some(gen_part(vis, pname, pf, *part == Part::Execution, *part == Part::Execution)));
    }
    let defs = gens.iter().flatten().map(|g| &g.def);
    let all_ids: Vec<&Ident> = fields.iter().map(|f| &f.ident).collect();
    let d = gens[0].as_ref().unwrap();
    let e = gens[1].as_ref().unwrap();
    let r = gens[3].as_ref().unwrap();
    let (dty, ety, rty) = (&d.ty, &e.ty, &r.ty);
    let (dbuild, ebuild, rbuild) = (&d.build, &e.build, &r.build);

    let fields_fn = quote! {
        fn fields() -> &'static [::store::FieldInfo] {
            static FIELDS: ::std::sync::OnceLock<::std::vec::Vec<::store::FieldInfo>> = ::std::sync::OnceLock::new();
            FIELDS.get_or_init(|| {
                let mut v = ::std::vec::Vec::new();
                #(#field_infos)*
                v
            })
        }
    };
    let varied_fn = quote! {
        fn varied(&self, field: &str) -> Self {
            #(#varied_arms)*
            panic!("`{}` is not an identity or render field of {}, so it has no second value", field, ::core::stringify!(#name));
        }
    };
    let split_body = quote! {
        let #name { #(#all_ids,)* } = self;
        let mut __fields = ::store::__private::serde_json::Map::new();
        #(#given_inserts)*
        let mut __given = ::store::Given::__from_fields(__fields);
        #(#group_splits)*
    };

    if file {
        Ok(quote! {
            #(#defs)*

            #[allow(unused_variables, unused_mut, clippy::unused_unit)]
            impl ::store::FileInputs for #name {
                type Declared = #dty;
                type Execution = #ety;
                type Request = #rty;
                #fields_fn
                fn __split_table(self) -> (#dty, #ety, #rty, ::store::Given) {
                    #split_body
                    (#dbuild, #ebuild, #rbuild, __given)
                }
                #varied_fn
            }
        })
    } else {
        let x = gens[2].as_ref().unwrap();
        let (xty, xbuild) = (&x.ty, &x.build);
        Ok(quote! {
            #(#defs)*

            #[allow(unused_variables, unused_mut, clippy::unused_unit)]
            impl ::store::Inputs for #name {
                type Declared = #dty;
                type Execution = #ety;
                type Exports = #xty;
                type Request = #rty;
                #fields_fn
                fn split(self) -> ::store::Split<Self> {
                    #split_body
                    ::store::Split::__new(#dbuild, #ebuild, #xbuild, #rbuild, __given)
                }
                #varied_fn
                fn exercised(field: &str) -> (Self, Self) {
                    #(#exercised_arms)*
                    panic!("`{}` is not an execution field of {}, so it has no fixture", field, ::core::stringify!(#name));
                }
            }
        })
    }
}

// ── EnvInputs ────────────────────────────────────────────────────────────────

enum EnvSource {
    Var { name: LitStr, parse: Option<Path> },
    Prefix(LitStr),
}

fn env_source(field: &syn::Field) -> syn::Result<EnvSource> {
    let attr = field
        .attrs
        .iter()
        .find(|a| a.path().is_ident("env"))
        .ok_or_else(|| syn::Error::new_spanned(field, "an environment field needs #[env(\"NAME\")] or #[env(prefix = \"P_\")]"))?;
    attr.parse_args_with(|input: ParseStream| {
        if input.peek(LitStr) {
            let name: LitStr = input.parse()?;
            let mut parse = None;
            if input.peek(syn::Token![,]) {
                let _: syn::Token![,] = input.parse()?;
                let k: Ident = input.parse()?;
                if k != "parse" {
                    return Err(syn::Error::new(k.span(), "expected `parse = <function>`"));
                }
                let _: syn::Token![=] = input.parse()?;
                parse = Some(input.parse::<Path>()?);
            }
            Ok(EnvSource::Var { name, parse })
        } else {
            let k: Ident = input.parse()?;
            if k != "prefix" {
                return Err(syn::Error::new(k.span(), "expected \"NAME\" or `prefix = \"P_\"`"));
            }
            let _: syn::Token![=] = input.parse()?;
            Ok(EnvSource::Prefix(input.parse()?))
        }
    })
}

fn snake(s: &str) -> String {
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if c.is_ascii_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

#[proc_macro_derive(EnvInputs, attributes(input, env))]
pub fn derive_env_inputs(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    env_impl(&input).unwrap_or_else(|e| e.to_compile_error()).into()
}

fn env_impl(input: &DeriveInput) -> syn::Result<TokenStream2> {
    let Data::Struct(s) = &input.data else {
        return Err(syn::Error::new_spanned(&input.ident, "#[derive(EnvInputs)] needs a struct"));
    };
    let name = &input.ident;
    let vis = &input.vis;
    let tag = snake(&name.to_string());
    let fields = input_fields(input, false, true)?;
    let raw: Vec<&syn::Field> = named_or_unit(&s.fields, name)?;
    let mut identity = Vec::new();
    let mut execution = Vec::new();
    let mut request = Vec::new();
    let mut reads = Vec::new();
    let mut given = Vec::new();
    let mut infos = Vec::new();
    for (f, rf) in fields.iter().zip(raw) {
        let (id, ty) = (&f.ident, &f.ty);
        let src = env_source(rf)?;
        let (var_text, read) = match &src {
            EnvSource::Prefix(p) => (p.value(), quote!(::store::__private::env::prefixed(#p)?)),
            EnvSource::Var { name: v, parse: Some(p) } => {
                (v.value(), quote!(::store::__private::env::parsed(#v, #p(::store::__private::env::raw(#v)?))?))
            }
            EnvSource::Var { name: v, parse: None } => {
                let seg = last_segment(ty).map(|s| s.ident.to_string()).unwrap_or_default();
                let r = match seg.as_str() {
                    "bool" => quote!(::store::__private::env::flag(#v)?),
                    "Option" => quote!(::store::__private::env::optional(#v)?),
                    _ => quote!(::store::__private::env::required(#v)?),
                };
                (v.value(), r)
            }
        };
        reads.push(quote! { let #id: #ty = #read; });
        let class = match f.class {
            Class::Identity | Class::Execution | Class::Store | Class::Recorded | Class::Skip => f.class,
            _ => return Err(syn::Error::new_spanned(id, "an environment variable is identity, execution, store, recorded or skip")),
        };
        if class.recorded() {
            given.push(quote! { __fields.insert(::std::string::String::from(#var_text), ::store::Given::__value(&#id)); });
        }
        let ct = class.tokens();
        infos.push(quote! { ::store::FieldInfo { name: #var_text, class: #ct, vary: None, exercised_by: None } });
        match class {
            Class::Identity => identity.push((id.clone(), ty.clone())),
            Class::Execution => execution.push((id.clone(), ty.clone())),
            Class::Store => request.push((id.clone(), quote!(#ty), quote!(#id))),
            _ => {}
        }
    }
    let all_env_ids: Vec<&Ident> = fields.iter().map(|f| &f.ident).collect();
    let ident_name = format_ident!("{}Identity", name);
    let (iids, itys): (Vec<_>, Vec<_>) = identity.iter().cloned().unzip();
    let exec = if execution.is_empty() {
        (quote!(()), quote!(), quote!(()))
    } else {
        let en = format_ident!("{}Execution", name);
        let (eids, etys): (Vec<_>, Vec<_>) = execution.iter().cloned().unzip();
        (quote!(#en), quote! { #vis struct #en { #(pub #eids: #etys,)* } }, quote!(#en { #(#eids,)* }))
    };
    let (ety, edef, ebuild) = exec;
    let req = gen_part(vis, format_ident!("{}Request", name), &request, false, false);
    let (rty, rdef, rbuild) = (req.ty, req.def, req.build);
    Ok(quote! {
        /// The identity-class environment variables, hashed as read.
        #[derive(::store::Identity, ::core::clone::Clone, ::core::fmt::Debug)]
        #[identity(tag = #tag, version = 1)]
        #vis struct #ident_name { #(pub #iids: #itys,)* }

        #edef
        #rdef

        impl ::store::EnvInputs for #name {
            type Identity = #ident_name;
            type Execution = #ety;
            type Request = #rty;
            fn fields() -> &'static [::store::FieldInfo] {
                static FIELDS: ::std::sync::OnceLock<::std::vec::Vec<::store::FieldInfo>> = ::std::sync::OnceLock::new();
                FIELDS.get_or_init(|| ::std::vec![#(#infos),*])
            }
            #[allow(unused_mut, clippy::unused_unit)]
            fn read() -> ::store::Result<(#ident_name, #ety, #rty, ::store::Given)> {
                #(#reads)*
                // The struct is the declaration; the values pass through it once.
                let #name { #(#all_env_ids,)* } = #name { #(#all_env_ids,)* };
                let mut __fields = ::store::__private::serde_json::Map::new();
                #(#given)*
                let __given = ::store::Given::__from_fields(__fields).__without_argv();
                ::core::result::Result::Ok((#ident_name { #(#iids,)* }, #ebuild, #rbuild, __given))
            }
        }
    })
}
