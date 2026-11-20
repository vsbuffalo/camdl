-- Drops the doc-test preamble fences from the page.
--
--   ```camdl-doctest-preamble sir
--   compartments { S, I, R }
--   ...
--   ```
--
-- A preamble holds the declarations a fragment on the page borrows so that
-- `camdlc doctest` (`make test-docs`) can compile the fragment alone; it is
-- not meant to be read. It is a fence rather than an HTML comment because its
-- model text has arrows (`S --> I`), and the first `-->` would end a comment
-- and show the rest of the model on the page (gh#1324).

return {
  {
    CodeBlock = function(block)
      if block.classes:includes("camdl-doctest-preamble") then
        return {}
      end
    end,
  },
}
