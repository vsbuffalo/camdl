-- Website-only orientation note at the top of the language specification.
-- The file itself is untouched: agents reading it, and `camdl docs language`,
-- still see the design principles first. Only the rendered page gains a
-- pointer for a newcomer who arrived from search.

local note = [[
This is the normative reference for the camdl language: each section states
exactly what the compiler accepts. To learn the language, start with
[A tour of the language](intro.html) and
[Dimensions and stratification](guide/dimensions.html); come back here to
look things up.
]]

function Pandoc(doc)
  if not quarto.doc.input_file:match("camdl%-language%-spec%.md$") then
    return nil
  end
  local callout = quarto.Callout({
    type = "tip",
    title = "Reading this page",
    content = pandoc.read(note, "markdown").blocks,
  })
  for i, block in ipairs(doc.blocks) do
    if block.t == "Header" and block.level == 1 then
      table.insert(doc.blocks, i + 1, callout)
      return doc
    end
  end
  return nil
end
