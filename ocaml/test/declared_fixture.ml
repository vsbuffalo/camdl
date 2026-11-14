(* One hand-written declared model that uses every type and every variant of
   `Declared` at least once. It is not a model anyone would write: its job is
   to pin the JSON wire format between this codec and the Rust `declared`
   crate.

   `gen_declared_fixture.exe` writes it to
   `rust/crates/declared/tests/fixtures/all_variants.json` (`make
   update-declared-fixture`); `test_declared.ml` checks the committed file
   against it, and `rust/crates/declared/tests/ocaml_fixture.rs` reads the same
   file through the Rust types. *)

open Declared

let fixture_path = "rust/crates/declared/tests/fixtures/all_variants.json"

let nr kind name = { nr_name = name; nr_kind = kind }
let id kind name = TIdent (nr kind name)
let comp = id KCompartment
let param = id KParameter
let num v = TConst v
let bin op l r = TBinOp (op, l, r)
let arg name value = { a_name = name; a_value = value }

(* A top-level expression; `reads` is given by hand. *)
let ex ?(katex = "x") ?(loss = []) ?(reads = []) tree =
  { de_tree = tree; de_katex = katex; de_katex_loss = loss; de_reads = reads }

(* Every way a rendering can fall short, once. *)
let every_katex_loss = List.map fst katex_loss_table

(* A declaration's text as written. The fixture uses one short text for most
   declarations, one text that exercises JSON escaping (a line break, a tab, a
   comment, a quoted string, a backslash and the non-ASCII `×`), and `None`
   twice, so both wire forms are pinned. *)
let written = Some "as written"

let written_transition =
  Some
    "#[lineage] infection[a in age] : S[a] --> I[a] + R[age = adult]\n\
     \t  @ beta × S[a]   # per \"capita\" \\ rate\n\
     \t  where a != b"

let const v = ex ~katex:(Printf.sprintf "%g" v) (num v)

(* Every name kind, once. *)
let all_name_kinds =
  [ nr KCompartment "S"; nr KParameter "beta"; nr KDefinition "N";
    nr KTable "contact"; nr KTimeInput "pop";
    nr KIndexVar "a"; nr KLevel "child"; nr KBuiltin "t";
    nr KDataColumn "tested"; nr KQuantity "peak"; nr KStream "cases";
    nr KTransition "infection"; nr KDimension "age";
    nr KObsAnchor "last_obs"; nr KKeyword "linear";
    nr KUnresolved "mystery" ]

(* Every binary operator, every unary operator, every unit, every expression
   node and both index-item forms, in one tree. *)
let every_expression_node =
  let binops =
    [ BAdd; BSub; BMul; BDiv; BPow; BEq; BNeq; BLt; BGt; BLe; BGe ]
  in
  let unops =
    [ ONeg; OExp; OLog; OSqrt; OAbs; OFloor; OCeil; OSin; OCos; OTanh; OTotal ]
  in
  let units =
    [ UDays; UWeeks; UMonths; UYears; UPerDay; UPerWeek; UPerMonth; UPerYear;
      UCount; URatio ]
  in
  TList
    ([ num 1.0; num 0.25; num (-3.5e-7);
       TIndex (nr KCompartment "S", [ IPositional (id KLevel "child") ]);
       TIndex
         (nr KTable "contact",
          [ INamed ("age", id KIndexVar "a"); IPositional (id KIndexVar "b") ]);
       TSum
         { ts_var = "a"; ts_dim = "age"; ts_guard = Some (GNeq ("a", "b"));
           ts_body = TIndex (nr KCompartment "I", [ IPositional (id KIndexVar "a") ]) };
       TSum
         { ts_var = "b"; ts_dim = "age"; ts_guard = None;
           ts_body = id KIndexVar "b" };
       TCond (bin BGt (comp "I") (num 0.0), param "beta", num 0.0);
       TCall
         ("overdispersed", [ arg "rate" (param "beta"); arg "sigma_sq" (num 0.1) ]);
       TCall ("now", []);
       TRange (num 7.0, num 100.0);
       TObsAccess "cases"; TStr "data/contact.tsv";
       TRunMember
         { tr_run = "baseline"; tr_namespace = NsQuantities; tr_member = "R0" };
       TRunMember
         { tr_run = "sia"; tr_namespace = NsObservations; tr_member = "cases" } ]
     @ List.map (fun op -> bin op (param "beta") (comp "S")) binops
     @ List.map (fun op -> TUnOp (op, param "gamma")) unops
     @ List.map (fun u -> TUnit (7.0, u)) units
     @ List.map (fun r -> TIdent r) all_name_kinds)

(* Every guard form and every relational operator. *)
let every_guard =
  let table op operand =
    GTable
      { gt_table = "adjacent"; gt_indices = [ "p"; "q" ]; gt_op = op;
        gt_operand = operand }
  in
  GOr
    ( GAnd (GEq ("a", "child"), GNeq ("p", "q")),
      GAnd
        ( GAnd (table RLt (GoNum 1.0), table RLe (GoName "threshold")),
          GAnd
            ( GAnd (table RGt (GoNum 0.0), table RGe (GoNum 0.5)),
              GAnd (table REq (GoNum 1.0), table RNe (GoNum 2.0)) ) ) )

let every_binding =
  [ BIn ("a", "age"); BConsecutive ("a", "a_next", "age"); BCompartments "c" ]

let s_ref = { sr_compartment = "S"; sr_indices = [] }
let i_ref = { sr_compartment = "I"; sr_indices = [ IPositional (id KIndexVar "a") ] }
let r_ref = { sr_compartment = "R"; sr_indices = [ INamed ("age", id KLevel "adult") ] }

let dist family args = { di_family = family; di_args = args }

let every_family =
  [ FNegBinomial; FPoisson; FNormal; FBinomial; FBetaBinomial; FBeta;
    FBernoulli; FZeroInflatedNegBinomial ]

let every_param_kind : Ir.param_kind list =
  [ Rate; Probability; Count; Positive; Real; Instant; Duration ]

let parameters =
  List.mapi
    (fun i (kind : Ir.param_kind) ->
      { pd_text = written; pd_name = Printf.sprintf "p%d" i;
        pd_dims = (if i = 0 then [ "age"; "patch" ] else []);
        pd_kind = kind;
        pd_dim =
          (if i = 0 then Some { da_population = -1; da_time = -1 } else None);
        pd_bounds =
          (if i < 2 then Some { b_lo = const 0.001; b_hi = const 2.0 } else None);
        pd_prior =
          (match i with
           | 0 ->
             Some
               { pr_family = "log_normal";
                 pr_args = [ arg "mu" (num 0.0); arg "sigma" (num 1.0) ];
                 pr_pool_over = Some "patch" }
           | 1 ->
             Some { pr_family = "uniform"; pr_args = []; pr_pool_over = None }
           | _ -> None) })
    every_param_kind

let model : declared_model =
  { dm_schema_version = schema_version;
    dm_compile_inputs =
      { ci_set_values =
          [ { sv_name = "beta"; sv_value = 0.3 };
            { sv_name = "N0"; sv_value = 1000.0 } ];
        ci_files =
          [ { fi_path = "data/patches.tsv";
              fi_sha256 =
                Some "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" };
            { fi_path = "data/gone.tsv"; fi_sha256 = None } ] };
    dm_time_unit = Some UDays;
    dm_description = Some "every variant, \"quoted\", and a non-ASCII β";
    dm_origin = Some "2024-01-01";
    dm_dimensions =
      [ { dd_text = written; dd_name = "age"; dd_source = DsInline; dd_levels = [ "child"; "adult" ] };
        { dd_text = written; dd_name = "patch";
          dd_source = DsRead { ds_path = "data/patches.tsv"; ds_column = "name" };
          dd_levels = [ "north"; "south_west" ] } ];
    dm_compartments =
      [ { cd_text = written; cd_name = "S"; cd_kind = CkInteger; cd_dims = [ "age" ];
          cd_expands_to = Some [ "S_child"; "S_adult" ] };
        { cd_text = None; cd_name = "W"; cd_kind = CkReal; cd_dims = []; cd_expands_to = None } ];
    dm_stratify =
      [ { st_text = written; st_dim = "age"; st_only = None };
        { st_text = written; st_dim = "patch"; st_only = Some [ "S"; "I" ] } ];
    dm_parameters = parameters;
    dm_definitions =
      [ { df_text = written; df_name = "N"; df_indices = []; df_shape = None; df_kind = None;
          df_dim = None;
          df_body =
            ex ~katex:"S + I"
              ~reads:[ nr KCompartment "S"; nr KCompartment "I" ]
              (bin BAdd (comp "S") (comp "I")) };
        { df_text = written; df_name = "everything"; df_indices = every_binding;
          df_shape = Some [ "age"; "patch" ]; df_kind = Some Ir.Rate;
          df_dim = Some { da_population = 0; da_time = -1 };
          df_body =
            ex ~katex:"\\frac{a}{b}" ~loss:every_katex_loss ~reads:all_name_kinds
              every_expression_node } ];
    dm_tables =
      [ { tb_text = written; tb_names = [ "contact"; "contact_se" ];
          tb_dims =
            [ { td_dim = "age"; td_unit = None };
              { td_dim = "week"; td_unit = Some UWeeks } ];
          tb_cell_kind = Some Ir.Probability;
          tb_value =
            ex
              (TCall
                 ("read_long", [ arg "path" (id KBuiltin "contact_file") ])) };
        { tb_text = written; tb_names = [ "w" ]; tb_dims = []; tb_cell_kind = None;
          tb_value = ex (TList [ TList [ num 1.0; num 2.0 ]; TList [ num 3.0; num 4.0 ] ]) } ];
    dm_time_inputs =
      [ { ti_text = written; ti_name = "pop"; ti_indices = [ BIn ("p", "patch") ];
          ti_kind = "interpolated"; ti_unit = UCount;
          ti_args =
            [ { ta_name = "file"; ta_value = TvStr "data/pop.tsv" };
              { ta_name = "period"; ta_value = TvExpr (ex (TUnit (1.0, UYears))) } ] } ];
    dm_timepoints = [ { tp_text = written; tp_name = "sia1"; tp_time = ex (TUnit (30.0, UDays)) } ];
    dm_transitions =
      [ { tn_text = written_transition; tn_name = "infection"; tn_indices = [ BIn ("a", "age") ];
          tn_from = [ s_ref ]; tn_to = DstSum [ i_ref; r_ref ];
          tn_dynamics =
            DynRate
              (ex ~katex:"\\beta S"
                 ~reads:[ nr KParameter "beta"; nr KCompartment "S" ]
                 (bin BMul (param "beta") (comp "S")));
          tn_guard = Some every_guard; tn_lineage = true;
          tn_expands_to = Some [ "infection_child"; "infection_adult" ] };
        { tn_text = written; tn_name = "onset"; tn_indices = []; tn_from = [ i_ref ];
          tn_to =
            DstBranch
              [ { ba_to = r_ref; ba_weight = ex (param "p_recover") };
                { ba_to = s_ref;
                  ba_weight = ex (bin BSub (num 1.0) (param "p_recover")) } ];
          tn_dynamics =
            DynVia
              ("erlang", [ arg "stages" (num 3.0); arg "mean" (TUnit (9.0, UDays)) ]);
          tn_guard = None; tn_lineage = false;
          tn_expands_to = None } ];
    dm_ode_equations =
      [ { od_text = written; od_compartment = "W";
          od_derivative = ex (bin BSub (comp "I") (comp "W")) } ];
    dm_balance =
      [ { bl_text = written; bl_compartment = "R"; bl_expr = ex (bin BSub (id KTimeInput "pop") (comp "S")) } ];
    dm_initial =
      { in_text = written; in_compartment = "S"; in_indices = [ IPositional (id KLevel "child") ];
        in_bindings = [ BIn ("p", "patch") ];
        in_value = IvExpr (ex (bin BSub (param "N0") (param "I0"))) }
      :: List.map
           (fun family ->
             { in_text = written; in_compartment = "I"; in_indices = []; in_bindings = [];
               in_value = IvLaw (dist family [ arg "rate" (param "I0") ]) })
           every_family;
    dm_observations =
      [ { ob_text = written; ob_name = "cases"; ob_indices = [ BIn ("p", "patch") ];
          ob_source = Some "surveillance";
          ob_columns =
            { oc_name = "week"; oc_role = CrTime }
            :: { oc_name = "patch"; oc_role = CrDim "patch" }
            :: { oc_name = "start"; oc_role = CrWindowStart }
            :: { oc_name = "stop"; oc_role = CrWindowStop }
            :: List.mapi
                 (fun i kind ->
                   { oc_name = Printf.sprintf "v%d" i; oc_role = CrValue kind })
                 every_param_kind;
          ob_measurement =
            { ms_scored = "v2";
              ms_likelihood =
                dist FNegBinomial
                  [ arg "mean" (bin BMul (param "rho") (id KBuiltin "projected"));
                    arg "dispersion" (param "k") ] };
          ob_projection =
            PjIncidence ("infection", [ IPositional (id KIndexVar "p") ]);
          ob_emit_schedule = Some (ScEvery (ex (TUnit (1.0, UWeeks))));
          ob_covers =
            Some
              { cv_form = "ending_on"; cv_column = "week";
                cv_span = Some (ex (TUnit (7.0, UDays))) };
          ob_expands_to = Some [ "cases_north"; "cases_south_west" ] };
        { ob_text = written; ob_name = "prevalence"; ob_indices = []; ob_source = None;
          ob_columns = [ { oc_name = "t"; oc_role = CrTime } ];
          ob_measurement =
            { ms_scored = "positive";
              ms_likelihood =
                dist FBinomial
                  [ arg "n" (id KDataColumn "tested");
                    arg "p" (id KBuiltin "projected") ] };
          ob_projection = PjDerived (ex (bin BDiv (comp "I") (id KDefinition "N")));
          ob_emit_schedule = Some (ScAt [ const 7.0; const 14.0 ]);
          ob_covers = Some { cv_form = "day"; cv_column = "t"; cv_span = None };
          ob_expands_to = None } ];
    dm_interventions =
      [ { iv_text = written; iv_name = "sia"; iv_indices = [ BIn ("p", "patch") ];
          iv_actions =
            [ AcTransfer
                [ arg "fraction" (param "coverage"); arg "from" (comp "S");
                  arg "to" (comp "R") ];
              AcSet
                { as_compartment = "I"; as_indices = []; as_value = const 0.0 } ];
          iv_schedule = ShAtTimes [ ex (id KObsAnchor "first_obs"); const 60.0 ];
          iv_guard = Some (GEq ("p", "north")) };
        { iv_text = written; iv_name = "routine"; iv_indices = [];
          iv_actions =
            [ AcAdd
                { aa_compartment = "R";
                  aa_indices = [ IPositional (id KLevel "child") ];
                  aa_count = const 10.0 } ];
          iv_schedule =
            ShRecurring
              { sh_every = ex (TUnit (1.0, UMonths)); sh_from = Some (const 0.0);
                sh_to = None };
          iv_guard = None } ];
    dm_events =
      [ { iv_text = written; iv_name = "import"; iv_indices = [];
          iv_actions =
            [ AcAdd { aa_compartment = "I"; aa_indices = []; aa_count = const 1.0 } ];
          iv_schedule =
            ShEveryAtDay { sh_period = ex (TUnit (1.0, UYears)); sh_at_day = const 250.0 };
          iv_guard = None };
        { iv_text = written; iv_name = "import_window"; iv_indices = [];
          iv_actions =
            [ AcAdd { aa_compartment = "I"; aa_indices = []; aa_count = const 2.0 } ];
          iv_schedule =
            ShRecurring
              { sh_every = const 7.0; sh_from = None; sh_to = Some (const 70.0) };
          iv_guard = None } ];
    dm_reactive_interventions =
      [ { rc_text = written; rc_name = "response"; rc_indices = [ BIn ("p", "patch") ];
          rc_when =
            TgOr
              ( TgAnd
                  ( TgAtom (ex (bin BGt (TObsAccess "cases") (num 50.0))),
                    TgNot (TgAtom (ex (bin BLt (id KBuiltin "t") (num 30.0)))) ),
                TgAtom (ex (bin BGe (comp "I") (num 100.0))) );
          rc_after = Some (ex (TUnit (14.0, UDays)));
          rc_once = Some (const 1.0);
          rc_cooldown = Some (ex (TUnit (90.0, UDays)));
          rc_action =
            AcTransfer [ arg "count" (num 500.0); arg "from" (comp "S"); arg "to" (comp "R") ];
          rc_guard = Some (GNeq ("p", "south_west")) };
        { rc_text = written; rc_name = "plain"; rc_indices = [];
          rc_when = TgAtom (ex (bin BGt (comp "I") (num 10.0)));
          rc_after = None; rc_once = None; rc_cooldown = None;
          rc_action =
            AcSet { as_compartment = "I"; as_indices = []; as_value = const 0.0 };
          rc_guard = None } ];
    dm_scenarios =
      [ { sn_text = written; sn_name = "baseline";
          sn_fields =
            [ SfLabel "No campaign"; SfEnable [ "routine" ]; SfDisable [ "sia"; "import" ];
              SfSet [ { an_name = "beta"; an_value = const 0.4 } ];
              SfScale [ { an_name = "coverage"; an_value = const 0.5 } ];
              SfCompose [ "routine_only"; "low_coverage" ];
              SfTEnd (ex (TUnit (2.0, UYears))); SfExtends "default" ] };
        { sn_text = None; sn_name = "empty"; sn_fields = [] } ];
    dm_simulation =
      Some
        { sm_text = written; sm_from = ex (TUnit (0.0, UDays)); sm_to = ex (TUnit (80.0, UDays));
          sm_dt = Some (const 0.5); sm_integrator = Some "rk45";
          sm_atol = Some (const 1e-8); sm_rtol = None };
    dm_output =
      Some
        { ou_text = written; ou_trajectories =
            Some { to_schedule = ScEvery (const 1.0); to_format = "tsv" } };
    dm_quantities =
      [ { qd_text = written; qd_name = "attack_rate"; qd_indices = [ BIn ("a", "age") ];
          qd_body = ex (bin BDiv (comp "R") (id KDefinition "N")) } ];
    dm_contrasts =
      [ { ct_text = written; ct_name = "averted";
          ct_body =
            ex
              (bin BSub
                 (TRunMember
                    { tr_run = "baseline"; tr_namespace = NsObservations;
                      tr_member = "cases" })
                 (TRunMember
                    { tr_run = "sia"; tr_namespace = NsObservations;
                      tr_member = "cases" })) } ] }

(* A second, minimal document: every optional absent and every list empty. *)
let minimal : declared_model =
  { dm_schema_version = schema_version;
    dm_compile_inputs = { ci_set_values = []; ci_files = [] };
    dm_time_unit = None; dm_description = None; dm_origin = None;
    dm_dimensions = []; dm_compartments = []; dm_stratify = [];
    dm_parameters = []; dm_definitions = []; dm_tables = [];
    dm_time_inputs = []; dm_timepoints = []; dm_transitions = [];
    dm_ode_equations = []; dm_balance = []; dm_initial = [];
    dm_observations = []; dm_interventions = []; dm_events = [];
    dm_reactive_interventions = []; dm_scenarios = [];
    dm_simulation = None; dm_output = Some { ou_text = None; ou_trajectories = None };
    dm_quantities = []; dm_contrasts = [] }

(* The fixture file: both documents, under fixed keys. *)
let fixture_json : Yojson.Safe.t =
  `Assoc
    [ ("all_variants", Declared.to_json model);
      ("minimal", Declared.to_json minimal) ]

let fixture_text = Yojson.Safe.pretty_to_string ~std:true fixture_json ^ "\n"
