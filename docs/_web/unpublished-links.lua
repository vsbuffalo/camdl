-- Links from a published page to a docs/ markdown file that is not on the
-- website (a docs/dev/ proposal, an unlisted guide) are rewritten to the file
-- on GitHub. Without this Quarto copies the raw .md into _site/ and links to it,
-- which serves unrendered, unreviewed files as if they were site pages.
--
-- The published set is `camdl-published` in _quarto.yml, the same YAML anchor
-- that drives `project.render`, so the two cannot disagree.

local published = {}
local blob_base = "https://github.com/vsbuffalo/camdl/blob/main/docs/"

-- Resolve "a/../b/./c" to "b/c".
local function normalize(path)
  local parts = {}
  for seg in path:gmatch("[^/]+") do
    if seg == ".." then
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
  local rel = input:sub(#root + 2)
  return rel:match("^(.*)/[^/]+$") or ""
end

return {
  {
    Meta = function(meta)
      for _, item in ipairs(meta["camdl-published"] or {}) do
        published[pandoc.utils.stringify(item)] = true
      end
    end,
  },
  {
    Link = function(link)
      local target = link.target
      if target:match("^%a[%w+.-]*:") or target:match("^#") then
        return nil -- external URL or same-page anchor
      end
      local path, frag = target:match("^([^#]+%.md)(#?.*)$")
      if not path then
        return nil
      end
      local dir = page_dir()
      local resolved = normalize(dir == "" and path or (dir .. "/" .. path))
      if published[resolved] then
        return nil
      end
      link.target = blob_base .. resolved .. frag
      return link
    end,
  },
}
