//! The manifest: the title of a set of internals pages, and the groups the
//! modules are listed under on its index page.

use serde::Deserialize;

/// A set of internals pages, as the maintainer describes it in TOML.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// The index page's title.
    pub title: String,
    /// One or two sentences under the index page's title, saying what the
    /// pages describe.
    pub summary: String,
    /// The URL prefix that a repository-relative path is appended to, to make
    /// a link to the source file on the code host. Ends in a slash.
    pub repo_blob: String,
    /// The sidebar position of the index page. The module pages follow it, at
    /// `order + 1`, `order + 2`, ... in manifest order.
    pub order: i64,
    /// The groups, in the order the index lists them.
    #[serde(rename = "group")]
    pub groups: Vec<Group>,
}

/// A titled list of modules on the index page.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Group {
    /// The heading the group is listed under.
    pub title: String,
    /// The top-level modules in the group, by name. A module nested in one of
    /// them (`Ast.Located` in `Ast`) is shown on its parent's page and is not
    /// listed here.
    pub modules: Vec<String>,
}

impl Manifest {
    /// Reads a manifest from TOML text.
    pub fn from_toml(text: &str) -> Result<Manifest, String> {
        toml::from_str(text).map_err(|e| format!("manifest: {e}"))
    }

    /// The listed module names in manifest order: groups in order, and
    /// modules in order within each group.
    pub fn module_names(&self) -> impl Iterator<Item = &str> {
        self.groups
            .iter()
            .flat_map(|g| g.modules.iter().map(String::as_str))
    }
}
