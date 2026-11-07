-- Executed examples for the tutorial pages (docs/guide/*.qmd).
--
-- Two code-block forms, both resolved relative to the page's own directory:
--
--   ```{.camdl include="dimensions/seir_erlang.camdl" lines="5-20"}
--   ```
--     Shows the file's contents (optionally a line range) instead of a pasted
--     copy, so the model on the page is the model that the commands below run.
--
--   ```{.camdl-run}
--   camdl inspect dimensions/seir_erlang.camdl --transitions
--   ```
--     Runs the command at render time and shows it with its real output. A
--     non-zero exit fails the render; mark a deliberate error with
--     `expect="fail"` (and then a zero exit fails the render instead).
--
-- The binaries come from PATH. `make docs-site` puts this checkout's freshly
-- built camdl first, so the output always matches the code being documented.

local function page_dir()
  local input = quarto.doc.input_file
  return input:match("^(.*)/[^/]+$") or "."
end

local function read_file(path)
  local f = io.open(path, "r")
  if not f then
    error("camdl-run: cannot read " .. path)
  end
  local text = f:read("*a")
  f:close()
  return text
end

local function line_range(text, spec)
  local first, last = spec:match("^(%d+)%-(%d+)$")
  if not first then
    error("camdl-run: lines must look like 5-20, got " .. spec)
  end
  first, last = tonumber(first), tonumber(last)
  local out, n = {}, 0
  for line in (text .. "\n"):gmatch("(.-)\n") do
    n = n + 1
    if n >= first and n <= last then
      table.insert(out, line)
    end
  end
  -- A range the file no longer reaches means the excerpt drifted from it.
  if first < 1 or last < first or last > n then
    error("camdl-run: lines=" .. spec .. " but the file has " .. n .. " lines")
  end
  return table.concat(out, "\n")
end

local function strip_ansi(s)
  return (s:gsub("\27%[[%d;]*[A-Za-z]", ""))
end

local function shell_quote(s)
  return "'" .. s:gsub("'", "'\\''") .. "'"
end

-- Runs the block under bash with -e and pipefail, so a failure anywhere in a
-- multi-line block or a pipeline is its exit status, and with stderr folded
-- into the shown output for every command, not only the last.
local function run(cmd)
  local script = "set -eo pipefail\nexport NO_COLOR=1 CLICOLOR=0 TERM=dumb\n"
    .. "cd " .. shell_quote(page_dir()) .. " || exit 125\n" .. cmd
  local full = "bash -c " .. shell_quote(script)
    .. " 2>&1; printf '\\n__camdl_run_exit=%s' $?"
  local handle = io.popen(full)
  local out = handle:read("*a")
  handle:close()
  local body, code = out:match("^(.-)\n__camdl_run_exit=(%d+)$")
  return strip_ansi(body or out):gsub("%s+$", ""), tonumber(code or "1")
end

function CodeBlock(block)
  local include = block.attributes["include"]
  if include then
    local text = read_file(page_dir() .. "/" .. include)
    local lines = block.attributes["lines"]
    if lines then
      text = line_range(text, lines)
    end
    block.text = text:gsub("%s+$", "")
    block.attributes["include"] = nil
    block.attributes["lines"] = nil
    -- Quarto attaches `filename` headers before user filters run, so build
    -- the same markup it would: a file-name strip above the code block.
    local label = block.attributes["filename"]
      or (include .. (lines and " (excerpt)" or ""))
    block.attributes["filename"] = nil
    local header = pandoc.RawBlock("html",
      '<div class="code-with-filename-file"><pre><strong>'
        .. label:gsub("&", "&amp;"):gsub("<", "&lt;")
        .. "</strong></pre></div>")
    return pandoc.Div({ header, block }, pandoc.Attr("", { "code-with-filename" }))
  end

  if block.classes:includes("camdl-run") then
    local cmd = block.text:gsub("%s+$", "")
    local output, code = run(cmd)
    local expect_fail = block.attributes["expect"] == "fail"
    -- 125: cd failed; 126/127: not executable / not found. These are a broken
    -- build environment, never the failure a block means to show.
    if code == 125 or code == 126 or code == 127 then
      error("camdl-run: `" .. cmd .. "` could not run (exit " .. code .. "):\n" .. output)
    elseif expect_fail and code == 0 then
      error("camdl-run: expected a failure but `" .. cmd .. "` succeeded")
    elseif not expect_fail and code ~= 0 then
      error("camdl-run: `" .. cmd .. "` exited " .. code .. ":\n" .. output)
    end
    return pandoc.Div({
      pandoc.CodeBlock(cmd, pandoc.Attr("", { "bash", "camdl-run-cmd" })),
      pandoc.CodeBlock(output, pandoc.Attr("", { "camdl-run-out" })),
    }, pandoc.Attr("", { "camdl-run" }))
  end
end
