-- The five semantic boxes of docs/documentation-style.md, section 5.
--
--   ::: {.practice title="Checking a fit without reading its full log"}
--   ...
--   :::
--
-- Each authoring class becomes a Quarto callout, so the disclosure button,
-- its keyboard handling and its aria attributes are Quarto's own:
--
--   class        colour  label             callout type  default
--   .practice    green   In practice:      tip           collapsed
--   .modeling    yellow  Modeling choice:  warning       collapsed
--   .internals   orange  Under the hood:   caution       collapsed
--   .warning     red     Warning:          important     always open
--   .agents      slate   For AI agents:    note          collapsed
--
-- `collapse="false"` shows a practice, modeling, internals or agents box open (it can
-- still be closed). A `.warning` box is never collapsible: any `collapse` on
-- it is ignored, because a warning that changes whether to trust a result
-- belongs in the reading path.
--
-- The label is written into the visible title, so the kind of box never rests
-- on colour alone; a title that already starts with it is left as it is. A box
-- needs a title, and a box inside another box fails the render, as does a
-- paragraph starting with `:::` (a fence reflowed into its text). The colours
-- are in _web/theme.scss, keyed on the `camdl-box-<kind>` class added here.
--
-- `camdl docs` prints the same boxes open, as their titled label
-- (rust/crates/cli/src/docs.rs, BOX_LABELS, which a test holds to KINDS).

local KINDS = {
  practice = { type = "tip", label = "In practice:", collapsible = true },
  modeling = { type = "warning", label = "Modeling choice:", collapsible = true },
  internals = { type = "caution", label = "Under the hood:", collapsible = true },
  warning = { type = "important", label = "Warning:", collapsible = false },
  agents = { type = "note", label = "For AI agents:", collapsible = true },
}

local function kind_of(div)
  for _, class in ipairs(div.classes) do
    if KINDS[class] then
      return class
    end
  end
  return nil
end

local function title_inlines(text)
  local doc = pandoc.read(text, "commonmark_x")
  if #doc.blocks ~= 1 or not doc.blocks[1].content then
    error("boxes: a box title must be one line of inline text, got: " .. text)
  end
  return doc.blocks[1].content
end

local function box(div)
  local kind = kind_of(div)
  if not kind then
    return nil
  end
  local spec = KINDS[kind]
  local title = div.attributes["title"]
  if not title or title:match("^%s*$") then
    error("boxes: a ." .. kind .. " box needs a title that says what it answers")
  end
  title = title:gsub("^%s+", "")
  if title:sub(1, #spec.label):lower() ~= spec.label:lower() then
    title = spec.label .. " " .. title
  end

  pandoc.walk_block(pandoc.Div(div.content), {
    Div = function(inner)
      if kind_of(inner) or inner.classes:find_if(function(c) return c:match("^callout") end) then
        error("boxes: a box inside the ." .. kind .. " box \"" .. title
          .. "\"; boxes do not nest")
      end
    end,
  })

  local collapse = nil
  if spec.collapsible then
    local requested = div.attributes["collapse"] or "true"
    if requested ~= "true" and requested ~= "false" then
      error("boxes: collapse must be \"true\" or \"false\", got " .. requested)
    end
    collapse = requested
  end

  local classes = pandoc.List({ "camdl-box", "camdl-box-" .. kind })
  for _, class in ipairs(div.classes) do
    if class ~= kind then
      classes:insert(class)
    end
  end
  local attributes = {}
  for k, v in pairs(div.attributes) do
    if k ~= "title" and k ~= "collapse" then
      attributes[k] = v
    end
  end

  return quarto.Callout({
    type = spec.type,
    title = title_inlines(title),
    collapse = collapse,
    icon = false,
    content = div.content,
    attr = pandoc.Attr(div.identifier, classes, attributes),
  })
end

-- A paragraph that starts with `:::` is a fence pandoc did not read as one.
-- The usual cause is dprint: without a blank line between a fence and the
-- box's text, it reflows them into one paragraph. Left alone it would print
-- as text.
local function stray_fence(para)
  local first = para.content[1]
  if first and first.t == "Str" and first.text:match("^:::") then
    error("boxes: a paragraph starts with `:::`, so a fenced div was not read as one "
      .. "(is a blank line missing after the fence?): " .. pandoc.utils.stringify(para):sub(1, 80))
  end
end

-- Top-down, so the outer box is seen first and can refuse a box inside it.
return {
  { traverse = "topdown",
    Div = function(div)
      local out = box(div)
      if out then
        return out, false
      end
    end,
    Para = stray_fence,
    Plain = stray_fence },
}
