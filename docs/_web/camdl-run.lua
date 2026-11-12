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
--     `figure="prior.svg"` inlines an SVG the command wrote below its output,
--     with an optional `caption="..."`; a missing file fails the render.
--
--   ```{.camdl-run show-lines="1-6,40-52"}
--   ```
--     Shows only those lines of the output (a single line is `7`; ranges
--     ascending, not overlapping), with a `...` line wherever lines were cut:
--     between two ranges that do not touch, and before the first or after
--     the last range when the output continues there. The command still runs
--     in full and its exit status is checked as above. A range past the end
--     of the output fails the render, as `lines=` does on an include, so a
--     change in the output cannot silently shift which lines the page shows.
--
--   ```{.camdl-run show-match="NOT converged|acceptance" show-context="1"}
--   ```
--     Shows the lines containing any of the `|`-separated texts, each with
--     `show-context` lines (default 0) either side, cut with `...` the same
--     way. The texts are plain text, not Lua patterns or regular
--     expressions, matched against the output with its colours removed; a
--     literal `|` cannot be matched. Each text must match at least one line,
--     or the render fails. `show-lines` and `show-match` do not combine.
--
-- A page that writes results declares a scratch directory in its front matter:
--
--   camdl-run-scratch: first-fit
--
-- Every block then runs in a fresh copy of that subdirectory, made once per
-- render and deleted after it, so the commands read as a user types them in
-- their project (`camdl fit run fit.toml`), the checkout never collects
-- results/, and a run label (which can never be reused) is new each render.
-- `include` paths stay relative to the page.
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

-- The colours camdl paints with (rust/crates/cli/src/out/paint.rs, `Style`,
-- and camdlc's ocaml/lib/compiler/term_style.ml, which adds blue for
-- transitions and magenta for compartments), each mapped to a class the site
-- theme colours for light and dark. A code not
-- listed here fails the render: an unmapped code would otherwise vanish or
-- show as raw escape bytes, and the page would stop matching the terminal.
local SGR_CLASS = {
  ["1"] = "ansi-bold",
  ["2"] = "ansi-dim",
  ["31"] = "ansi-err",
  ["32"] = "ansi-ok",
  ["33"] = "ansi-warn",
  ["34"] = "ansi-transition",
  ["35"] = "ansi-compartment",
  ["36"] = "ansi-info",
}

local function html_escape(s)
  return (s:gsub("&", "&amp;"):gsub("<", "&lt;"):gsub(">", "&gt;"))
end

-- A run's output as HTML, one entry per line: text escaped, each SGR colour a
-- span, `ESC[0m` (or `ESC[m`) closing every open span. Any other escape
-- sequence fails. A colour left open at a line break is closed at the end of
-- that line and reopened at the start of the next, so every line is balanced
-- HTML on its own and a selection of lines (`show-lines`) cannot leave a span
-- open or close one it never opened.
local function ansi_to_html_lines(s, cmd)
  local function refuse(what)
    error("camdl-run: `" .. cmd .. "` printed " .. what
      .. "; map it in SGR_CLASS (docs/_web/camdl-run.lua) and the theme, or stop printing it")
  end
  local lines, open = {}, {}
  local function span(class)
    return '<span class="' .. class .. '">'
  end
  for line in (s .. "\n"):gmatch("(.-)\n") do
    local out, pos = {}, 1
    for _, class in ipairs(open) do
      table.insert(out, span(class))
    end
    local function text(t)
      if t:find("\27", 1, true) then
        refuse("an escape sequence that is not a colour")
      end
      table.insert(out, html_escape(t))
    end
    while true do
      local a, b, params, final = line:find("\27%[([%d;]*)(%a)", pos)
      if not a then
        text(line:sub(pos))
        break
      end
      text(line:sub(pos, a - 1))
      if final ~= "m" then
        refuse("the escape sequence ESC[" .. params .. final .. ", which is not a colour")
      end
      for code in ((params == "") and "0" or params):gmatch("[^;]+") do
        if code == "0" then
          table.insert(out, string.rep("</span>", #open))
          open = {}
        elseif SGR_CLASS[code] then
          table.insert(out, span(SGR_CLASS[code]))
          table.insert(open, SGR_CLASS[code])
        else
          refuse("colour code " .. code .. ", which the docs do not map")
        end
      end
      pos = b + 1
    end
    table.insert(out, string.rep("</span>", #open))
    table.insert(lines, table.concat(out))
  end
  return lines
end

local function strip_ansi(s)
  return (s:gsub("\27%[[%d;]*m", ""))
end

local function split_lines(s)
  local lines = {}
  for line in (s .. "\n"):gmatch("(.-)\n") do
    table.insert(lines, line)
  end
  return lines
end

-- `show-lines="1-6,40-52"`: the set of output lines to show. Ranges are
-- ascending and do not overlap; a single line is `7`. A range past the end
-- of the output fails, as `lines=` does on an include, because the excerpt
-- no longer shows what it was written against.
local function lines_by_number(spec, n, cmd)
  local keep, last_seen = {}, 0
  for part in (spec .. ","):gmatch("(.-),") do
    part = part:gsub("^%s+", ""):gsub("%s+$", "")
    local first, last = part:match("^(%d+)%-(%d+)$")
    if not first then
      first = part:match("^(%d+)$")
      last = first
    end
    if not first then
      error("camdl-run: show-lines must look like 1-6,40-52, got " .. spec)
    end
    first, last = tonumber(first), tonumber(last)
    if first < 1 or last < first or first <= last_seen then
      error("camdl-run: show-lines=" .. spec .. ": ranges must be ascending and must not overlap")
    end
    if last > n then
      error("camdl-run: show-lines=" .. spec .. " but `" .. cmd .. "` printed " .. n .. " lines")
    end
    for i = first, last do
      keep[i] = true
    end
    last_seen = last
  end
  return keep
end

-- `show-match="NOT converged|acceptance"`: the lines containing any of the
-- `|`-separated texts (plain text, not patterns, matched against the output
-- without its colours), each with `show-context` lines either side. Every
-- alternative must match at least one line, so a line that stops being
-- printed fails the render instead of quietly dropping out of the excerpt.
local function lines_by_match(spec, context, plain, cmd)
  local keep = {}
  for needle in (spec .. "|"):gmatch("(.-)|") do
    if needle == "" then
      error("camdl-run: show-match has an empty alternative: " .. spec)
    end
    local found = false
    for i, line in ipairs(plain) do
      if line:find(needle, 1, true) then
        found = true
        for j = math.max(1, i - context), math.min(#plain, i + context) do
          keep[j] = true
        end
      end
    end
    if not found then
      error("camdl-run: show-match: no line of `" .. cmd .. "` contains \"" .. needle .. "\"")
    end
  end
  return keep
end

-- The shown lines, with `...` wherever lines were cut: between two kept runs
-- and before the first or after the last when the output continues there.
local function excerpt(lines, keep, cut)
  local out, prev = {}, 0
  for i = 1, #lines do
    if keep[i] then
      if i > prev + 1 then
        table.insert(out, cut)
      end
      table.insert(out, lines[i])
      prev = i
    end
  end
  if prev < #lines then
    table.insert(out, cut)
  end
  return out
end

local function shell_quote(s)
  return "'" .. s:gsub("'", "'\\''") .. "'"
end

-- Runs the block under bash with -e and pipefail, so a failure anywhere in a
-- multi-line block or a pipeline is its exit status, and with stderr folded
-- into the shown output for every command, not only the last. camdl paints
-- its output as it would in a terminal (CLICOLOR_FORCE, NO_COLOR unset);
-- CLICOLOR stays unset so `ls` and friends do not colour theirs. The raw
-- output, escapes included, is returned; the caller renders or strips them.
local workdir = nil -- set per page by setup(); nil means the page directory

local function run(cmd)
  local script = "set -eo pipefail\nunset NO_COLOR CLICOLOR\nexport CLICOLOR_FORCE=1 TERM=dumb\n"
    .. "cd " .. shell_quote(workdir or page_dir()) .. " || exit 125\n" .. cmd
  local full = "bash -c " .. shell_quote(script)
    .. " 2>&1; printf '\\n__camdl_run_exit=%s' $?"
  local handle = io.popen(full)
  local out = handle:read("*a")
  handle:close()
  local body, code = out:match("^(.-)\n__camdl_run_exit=(%d+)$")
  return (body or out):gsub("%s+$", ""), tonumber(code or "1")
end

local function capture(cmd)
  local handle = io.popen(cmd)
  local out = handle:read("*a")
  local ok = handle:close()
  return out:gsub("%s+$", ""), ok
end

local function setup(meta)
  local sub = meta["camdl-run-scratch"]
  if not sub then
    return
  end
  sub = pandoc.utils.stringify(sub)
  if sub == "" or sub:find("%.%.") or sub:sub(1, 1) == "/" then
    error("camdl-run-scratch must name a subdirectory of the page, got " .. sub)
  end
  local src = page_dir() .. "/" .. sub
  local tmp, ok = capture("mktemp -d")
  if not ok or tmp == "" then
    error("camdl-run: mktemp -d failed")
  end
  local _, copied = capture("cp -R " .. shell_quote(src) .. "/. " .. shell_quote(tmp) .. "/")
  if not copied then
    error("camdl-run: cannot copy " .. src .. " to " .. tmp)
  end
  workdir = tmp
end

local function teardown()
  if workdir then
    capture("rm -rf " .. shell_quote(workdir))
    workdir = nil
  end
end

-- An SVG the block wrote, inlined so it takes the page's colours (a plot
-- drawn in `currentColor` follows the light and dark themes).
local function figure_block(path, caption)
  if path:find("%.%.") or path:sub(1, 1) == "/" or not path:match("%.svg$") then
    error("camdl-run: figure must be a relative .svg path, got " .. path)
  end
  local svg = read_file((workdir or page_dir()) .. "/" .. path)
  svg = svg:gsub("^%s*<%?xml.-%?>%s*", ""):gsub("^%s*<!DOCTYPE.->%s*", "")
  local html = '<figure class="camdl-fig camdl-plot">' .. svg
  if caption then
    html = html .. "<figcaption>" .. caption:gsub("&", "&amp;"):gsub("<", "&lt;") .. "</figcaption>"
  end
  return pandoc.RawBlock("html", html .. "</figure>")
end

local function code_block(block)
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
      error("camdl-run: `" .. cmd .. "` could not run (exit " .. code .. "):\n" .. strip_ansi(output))
    elseif expect_fail and code == 0 then
      error("camdl-run: expected a failure but `" .. cmd .. "` succeeded")
    elseif not expect_fail and code ~= 0 then
      error("camdl-run: `" .. cmd .. "` exited " .. code .. ":\n" .. strip_ansi(output))
    end
    local blocks = { pandoc.CodeBlock(cmd, pandoc.Attr("", { "bash", "camdl-run-cmd" })) }
    if output ~= "" then
      -- Converting every line first refuses an unmapped colour anywhere in
      -- the output, shown or not.
      local html = ansi_to_html_lines(output, cmd)
      local plain = split_lines(strip_ansi(output))
      local by_lines, by_match = block.attributes["show-lines"], block.attributes["show-match"]
      local context = block.attributes["show-context"]
      if by_lines and by_match then
        error("camdl-run: use show-lines or show-match, not both")
      elseif context and not by_match then
        error("camdl-run: show-context only applies with show-match")
      end
      local keep = nil
      if by_lines then
        keep = lines_by_number(by_lines, #plain, cmd)
      elseif by_match then
        local n = tonumber(context or "0")
        if not n or n < 0 or n ~= math.floor(n) then
          error("camdl-run: show-context must be a whole number, got " .. context)
        end
        keep = lines_by_match(by_match, n, plain, cmd)
      end
      if keep then
        html = excerpt(html, keep, '<span class="camdl-run-cut">...</span>')
        plain = excerpt(plain, keep, "...")
      end
      if quarto.doc.is_format("html") then
        table.insert(blocks, pandoc.RawBlock("html",
          '<pre class="camdl-run-out"><code>' .. table.concat(html, "\n") .. "</code></pre>"))
      else
        table.insert(blocks, pandoc.CodeBlock(table.concat(plain, "\n"), pandoc.Attr("", { "camdl-run-out" })))
      end
    end
    local run_block = pandoc.Div(blocks, pandoc.Attr("", { "camdl-run" }))
    local figure = block.attributes["figure"]
    if not figure then
      return run_block
    end
    return { run_block, figure_block(figure, block.attributes["caption"]) }
  end
end

-- Three passes in order: make the scratch copy, run the blocks, delete it.
return {
  { Meta = function(meta) setup(meta) end },
  { CodeBlock = code_block },
  { Pandoc = function() teardown() end },
}
