-- Links from a published page to a docs/ markdown file that is not on the
-- website (a docs/dev/ proposal, an unlisted guide) are rewritten to the file
-- on GitHub. Without this Quarto copies the raw .md into _site/ and links to it,
-- which serves unrendered, unreviewed files as if they were site pages.
--
-- The published set is `camdl-published` in _quarto.yml, the same YAML anchor
-- that drives `project.render`, so the two cannot disagree. Quarto rewrites
-- path-like metadata relative to the page, so on guide/first-fit.qmd the
-- entry "concepts.md" arrives as "../concepts.md"; each entry is resolved
-- against the page's directory before use.

local published = {}
local blob_base = "https://github.com/vsbuffalo/camdl/blob/main/docs/"

-- Resolve "a/../b/./c" to "b/c".
-- Returns nil for a path that climbs above docs/.
local function normalize(path)
  local parts = {}
  for seg in path:gmatch("[^/]+") do
    if seg == ".." then
      if #parts == 0 then
        return nil
      end
      table.remove(parts)
    elseif seg ~= "." then
      table.insert(parts, seg)
    end
  end
  return table.concat(parts, "/")
end

-- Directory of the page being rendered, relative to docs/.
local function page_dir()
  local input = quarto.doc.input_file
  local root = quarto.project.directory
  if input:sub(1, #root + 1) ~= root .. "/" then
    error("unpublished-links.lua: page " .. input .. " is not under the project " .. root)
  end
  local rel = input:sub(#root + 2)
  return rel:match("^(.*)/[^/]+$") or ""
end

return {
  {
    Meta = function(meta)
      local dir = page_dir()
      for _, item in ipairs(meta["camdl-published"] or {}) do
        local entry = pandoc.utils.stringify(item)
        local resolved = normalize(dir == "" and entry or (dir .. "/" .. entry))
        if resolved == nil then
          error("unpublished-links.lua: published entry " .. entry .. " resolves above docs/")
        end
        published[resolved] = true
      end
      -- Without the list every docs link would be sent to GitHub.
      if next(published) == nil then
        error("unpublished-links.lua: camdl-published is missing from _quarto.yml")
      end
    end,
  },
  {
    Link = function(link)
      local target = link.target
      if target:match("^%a[%w+.-]*:") or target:match("^#") then
        return nil -- external URL or same-page anchor
      end
      local path, frag = target:match("^([^#?]+%.md)([#?]?.*)$")
      if not path then
        return nil
      end
      local dir = page_dir()
      local joined = dir == "" and path or (dir .. "/" .. path)
      local resolved = normalize(joined)
      if resolved == nil then
        return nil -- above docs/ (e.g. ../CHANGELOG.md): not a docs page
      end
      if published[resolved] then
        return nil
      end
      link.target = blob_base .. resolved .. frag
      return link
    end,
  },
}
