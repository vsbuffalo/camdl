; ── Top-level block keywords ─────────────────────────────────────────────────

[
  "time_unit"
  "description"
  "origin"
  "dimensions"
  "compartments"
  "parameters"
  "tables"
  "functions"
  "forcing"
  "transitions"
  "observations"
  "quantities"
  "contrasts"
  "interventions"
  "reactive_interventions"
  "events"
  "ode"
  "output"
  "simulate"
  "init"
  "timepoints"
  "stratify"
  "let"
  "scenarios"
  "balance"
] @keyword

; ── Scenario verbs / intervention verbs / schedule keywords ──────────────────

[
  "from"
  "to"
  "where"
  "in"
  "by"
  "values"
  "only"
  "at"
  "at_day"
  "every"
  "until"
  "tag"
  "transfer"
  "add"
  "consecutive"
  "extends"
  "set"
  "scale"
  "enable"
  "disable"
  "compose"
  "label"
  "via"
  "action"
  "columns"
  "emit_schedule"
] @keyword.operator

; ── Conditionals / reactive triggers ─────────────────────────────────────────

[
  "if"
  "then"
  "else"
  "when"
] @keyword.conditional

[
  "and"
  "or"
  "not"
] @keyword.operator

"sum" @keyword.function

; ── Attributes — `#[lineage]` and any future #[…] ─────────────────────────────

(attribute name: (identifier) @attribute)
(hash_lbracket) @punctuation.special

; ── Types / kinds ─────────────────────────────────────────────────────────────

(param_kind) @type.builtin

[
  "real"
  "integer"
] @type.builtin

(dim_literal) @type            ; `[1]`, `[P]`, `[T^-1]`, `[P*T^-1]`, etc.

; ── Operators ─────────────────────────────────────────────────────────────────

[
  "-->"
  "@"
  "~"
  "=="
  "!="
  "<"
  ">"
  "<="
  ">="
  "+"
  "-"
  "*"
  "/"
  "×"
  "^"
  "="
] @operator

; ── Punctuation ───────────────────────────────────────────────────────────────

[ "{" "}" ] @punctuation.bracket
[ "[" "]" ] @punctuation.bracket
[ "(" ")" ] @punctuation.bracket
[ "," ":" ] @punctuation.delimiter

; ── Literals ──────────────────────────────────────────────────────────────────

(number) @number
(unit_number value: (number) @number)
(unit_literal) @attribute           ; 'days, 'per_day, 'count, 'ratio, etc.
(string) @string
"null" @constant.builtin

; The ISO date string in `origin = date("YYYY-MM-DD")` gets a more specific
; tag so themes can highlight it distinctly from generic strings.
(origin_decl iso_date: (string) @string.special)

; ── Declarations — names ──────────────────────────────────────────────────────

(compartment_decl name: (identifier) @variable.parameter)
(parameter_decl   name: (identifier) @variable.parameter)
(table_decl       name: (identifier) @variable.parameter)
(function_decl    name: (identifier) @function)
(ode_decl         comp: (identifier) @variable.parameter)
(let_decl         name: (identifier) @variable.parameter)
(timepoint_decl   name: (identifier) @variable.parameter)
(dim_entry        name: (identifier) @type)
(scenario_block   name: (identifier) @variable.parameter)
(balance_block    comp: (identifier) @variable.parameter)

(transition_decl  name: (identifier) @function)
(branch_entry     name: (identifier) @variable.parameter)

(obs_decl          name: (identifier) @function)
(intervention_decl name: (identifier) @function)
(reactive_decl     name: (identifier) @function)

(quantity_decl name: (identifier) @variable.parameter)
(contrast_decl name: (identifier) @variable.parameter)
(obs_column    name: (identifier) @variable.parameter)
(obs_column    role: (identifier) @type)

; The dwell-law name in a `via LAW(...)` clause (erlang / hyper_erlang / …).
(via_call law: (identifier) @function.builtin)

; Dotted run-rooted operands: `observations.<stream>`,
; `<run>.quantities.<member>`, `<run>.observations.<member>`.
(member_access run:    (identifier) @variable)
(member_access stream: (identifier) @property)
(member_access member: (identifier) @property)

; ── Index bindings ────────────────────────────────────────────────────────────

(index_binding var:  (identifier) @variable)
(index_binding dim:  (identifier) @type)
(index_binding next: (identifier) @variable)

(table_dim_entry dim:        (identifier) @type)

(param_index dim: (identifier) @type)
(param_prior dist: (identifier) @function.builtin)
(param_prior pool_over: (identifier) @type)

(dim_inline level: (identifier) @constant)

; ── Expressions — identifiers ─────────────────────────────────────────────────

; Generic identifier (fallback — lower priority than named fields above)
(identifier) @variable

(call_expr func: (identifier) @function.call)
(index_expr name: (identifier) @variable)
(sum_expr   var: (identifier)  @variable)
(sum_expr   dim: (identifier)  @type)

; Known built-in functions — these get the .builtin variant which themes
; can color distinctly from user functions.
((call_expr func: (identifier) @function.builtin)
  (#match? @function.builtin
   "^(date|add_calendar_days|add_calendar_weeks|add_calendar_months|add_calendar_years|date_range|read|read_levels|read_long|defines|incidence|cumulative|prevalence|overdispersed|deterministic|exp|log|min|max|mod|abs|sqrt|floor|ceil|round)$"))

; Distribution names (in priors and likelihoods).
((call_expr func: (identifier) @function.builtin)
  (#match? @function.builtin
   "^(poisson|neg_binomial|normal|binomial|beta_binomial|bernoulli|log_normal|half_normal|beta|gamma|exponential|uniform|log_uniform|truncated_normal|diagnostic_test)$"))

; Generated-quantity reductions and the reactive-trigger inputs. `observed` /
; `sum_observed` are only meaningful inside a `when` predicate; the reductions
; only inside a `quantities {}` body.
((call_expr func: (identifier) @function.builtin)
  (#match? @function.builtin
   "^(final|mean|integral|count_above|count_below|time_of_max|time_of_min|first_above|last_above|first_below|last_below|observed|sum_observed)$"))

; Dwell-law helpers usable in expression position (`branch(...)` inside a
; `hyper_erlang`); the law name itself is captured via (via_call law: …).
((call_expr func: (identifier) @function.builtin)
  (#match? @function.builtin
   "^(erlang|hyper_erlang|branch)$"))

; ── Stoich refs ───────────────────────────────────────────────────────────────

(stoich_ref name: (identifier) @variable.parameter)

; ── Guard expressions ─────────────────────────────────────────────────────────

(guard_atom left:  (identifier) @variable)
(guard_atom right: (identifier) @variable)

; ── Stratify ──────────────────────────────────────────────────────────────────

(stratify_kv (identifier) @type)

; ── Scenario contents ────────────────────────────────────────────────────────

(scenario_field ref: (identifier) @function)             ; enable = [iv_name]
(scenario_kv_item name: (identifier) @variable.parameter)

; ── Comments ──────────────────────────────────────────────────────────────────

(comment) @comment
