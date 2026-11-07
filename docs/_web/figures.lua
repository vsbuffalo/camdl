-- Web-only figures, placed by an HTML comment on its own line:
--
--   <!-- figure: architecture -->
--
-- GitHub hides the comment, and `camdl docs` prints it as a link to the
-- figure on the website (rust/crates/cli/src/docs.rs, `terminal_text`), so
-- the markdown reads the same without the drawing. Here the comment becomes
-- docs/figures/<name>.svg, inlined (not <img>) so the drawing takes the page's
-- colour variables and follows light and dark mode, inside
-- <figure id="fig-<name>">, the anchor the terminal link points at.
--
-- A text diagram the drawing should replace goes between the marker and a
-- closing one:
--
--   <!-- figure: date-axis -->
--   ```
--   Jan ── Feb ── Mar
--   ```
--   <!-- /figure -->
--
-- The website shows the SVG in place of everything between them; the terminal
-- and GitHub keep the text diagram. Without a closing marker the figure is
-- added and nothing is removed.
--
--   <!-- figure: architecture, caption: How a model becomes output. -->
--
-- adds a caption under the drawing. The caption is markdown (inline code,
-- emphasis, links) and must come last, so its text cannot be read as an option.
--
-- A marker naming a missing SVG, and a closing marker with no figure open,
-- fail the render. Conventions: docs/dev/documentation.md.

local function read_svg(name)
  local path = quarto.project.directory .. "/figures/" .. name .. ".svg"
  local f = io.open(path, "r")
  if not f then
    error("figures.lua: <!-- figure: " .. name .. " --> but " .. path .. " does not exist")
  end
  local svg = f:read("*a")
  f:close()
  return svg
end

local function is_close(block)
  return block.t == "RawBlock" and block.format == "html"
    and block.text:match("^%s*<!%-%-%s*/figure%s*%-%->%s*$") ~= nil
end

-- Returns the figure name and caption (or nil) of an opening marker.
local function parse_marker(block)
  if block.t ~= "RawBlock" or block.format ~= "html" then
    return nil
  end
  local body = block.text:match("^%s*<!%-%-%s*figure:%s*(.-)%s*%-%->%s*$")
  if not body then
    return nil
  end
  local name, rest = body:match("^([%w%-_]+)%s*,?%s*(.*)$")
  if not name then
    error("figures.lua: cannot read figure marker: " .. block.text)
  end
  local caption = rest:match("caption:%s*(.-)%s*$")
  return name, caption
end

local function caption_html(md)
  local doc = pandoc.read(md, "commonmark_x")
  local inlines = pandoc.utils.blocks_to_inlines(doc.blocks)
  return "<figcaption>" .. pandoc.write(pandoc.Pandoc({ pandoc.Plain(inlines) }), "html") .. "</figcaption>"
end

-- The index of the closing marker that pairs with the opening one at i, or
-- nil when the figure is open-ended (another figure, or the end of the list,
-- comes first).
local function find_close(blocks, i)
  for j = i + 1, #blocks do
    if is_close(blocks[j]) then
      return j
    end
    if parse_marker(blocks[j]) then
      return nil
    end
  end
  return nil
end

function Blocks(blocks)
  local out = pandoc.Blocks({})
  local i = 1
  while i <= #blocks do
    local block = blocks[i]
    local name, caption = parse_marker(block)
    if name then
      local cap = caption and caption_html(caption) or ""
      -- Each SVG titles itself with ids "t" and "d"; prefix them so two
      -- figures on one page keep distinct accessible names.
      local svg = read_svg(name)
        :gsub(' id="t"', ' id="fig-' .. name .. '-t"', 1)
        :gsub(' id="d"', ' id="fig-' .. name .. '-d"', 1)
        :gsub('aria%-labelledby="t d"', 'aria-labelledby="fig-' .. name .. '-t fig-' .. name .. '-d"', 1)
      out:insert(pandoc.RawBlock("html",
        '<figure class="camdl-fig" id="fig-' .. name .. '">' .. svg .. cap .. "</figure>"))
      i = (find_close(blocks, i) or i) + 1
    elseif is_close(block) then
      error("figures.lua: <!-- /figure --> with no figure open before it")
    else
      out:insert(block)
      i = i + 1
    end
  end
  return out
end
