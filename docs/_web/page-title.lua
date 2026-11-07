-- The docs/*.md pages carry no YAML front matter (they are also printed raw by
-- `camdl docs`), so their title is the first level-1 heading. Use it as the
-- browser-tab title; without this the tab shows the file name ("intro"), which
-- Quarto pre-fills into `pagetitle` before user filters run.
-- Only `pagetitle` is set — setting `title` would render the heading twice.

function Pandoc(doc)
  if doc.meta.title then
    return nil
  end
  for _, block in ipairs(doc.blocks) do
    if block.t == "Header" and block.level == 1 then
      doc.meta.pagetitle = pandoc.utils.stringify(block.content)
      return doc
    end
  end
  return nil
end
