//! The declared-model types: what counts as the same expression, and that a
//! document survives a JSON round trip.

use declared::*;

fn name(n: &str, kind: NameKind) -> ExprTree {
    ExprTree::Ident { name: NameRef { name: n.into(), kind } }
}
fn param(n: &str) -> ExprTree {
    name(n, NameKind::Parameter)
}
fn comp(n: &str) -> ExprTree {
    name(n, NameKind::Compartment)
}
fn bin(op: BinOp, lhs: ExprTree, rhs: ExprTree) -> ExprTree {
    ExprTree::BinOp { op, lhs: Box::new(lhs), rhs: Box::new(rhs) }
}
fn mul(lhs: ExprTree, rhs: ExprTree) -> ExprTree {
    bin(BinOp::Mul, lhs, rhs)
}
fn div(lhs: ExprTree, rhs: ExprTree) -> ExprTree {
    bin(BinOp::Div, lhs, rhs)
}

/// `beta * S * I`, as the parser associates it: `(beta * S) * I`.
fn beta_s_i() -> ExprTree {
    mul(mul(param("beta"), comp("S")), comp("I"))
}

fn expr(tree: ExprTree, katex: &str, reads: &[(&str, NameKind)]) -> DeclaredExpr {
    DeclaredExpr {
        tree,
        katex: katex.into(),
        reads: reads.iter().map(|(n, k)| NameRef { name: (*n).into(), kind: *k }).collect(),
    }
}

/// A two-compartment model exercising most declaration kinds.
fn small_model() -> DeclaredModel {
    let rate = expr(
        div(beta_s_i(), name("N", NameKind::Definition)),
        r"\beta S I / N",
        &[
            ("beta", NameKind::Parameter),
            ("S", NameKind::Compartment),
            ("I", NameKind::Compartment),
            ("N", NameKind::Definition),
        ],
    );
    let n_body = expr(
        bin(BinOp::Add, comp("S"), comp("I")),
        "S + I",
        &[("S", NameKind::Compartment), ("I", NameKind::Compartment)],
    );
    let lit = |v: f64| expr(ExprTree::Const { value: v }, &v.to_string(), &[]);
    let days = |v: f64| expr(ExprTree::Unit { value: v, unit: UnitLit::Days }, &format!("{v}\\,\\text{{d}}"), &[]);
    DeclaredModel {
        schema_version: SCHEMA_VERSION,
        compile_inputs: CompileInputs {
            set_values: vec![SetValue { name: "beta".into(), value: 0.3 }],
            files: vec![FileInput { path: "ages.tsv".into(), sha256: Some("ab".repeat(32)) }],
        },
        time_unit: Some(UnitLit::Days),
        description: None,
        origin: Some("2024-01-01".into()),
        dimensions: vec![DimensionDecl {
            name: "age".into(),
            source: DimSource::Read { path: "ages.tsv".into(), column: "age".into() },
            levels: vec!["child".into(), "adult".into()],
        }],
        compartments: vec![
            CompartmentDecl {
                name: "S".into(),
                kind: CompartmentKind::Integer,
                dims: vec!["age".into()],
                expands_to: Some(vec!["S_child".into(), "S_adult".into()]),
            },
            CompartmentDecl {
                name: "I".into(),
                kind: CompartmentKind::Integer,
                dims: vec!["age".into()],
                expands_to: Some(vec!["I_child".into(), "I_adult".into()]),
            },
        ],
        stratify: vec![StratifyDecl { dim: "age".into(), only: None }],
        parameters: vec![ParameterDecl {
            name: "beta".into(),
            dims: vec![],
            kind: ParamKind::Rate,
            dim: None,
            bounds: Some(Bounds { lo: lit(0.001), hi: lit(2.0) }),
            prior: Some(PriorDecl {
                family: "log_normal".into(),
                args: vec![
                    Arg { name: "mu".into(), value: ExprTree::Const { value: -1.0 } },
                    Arg { name: "sigma".into(), value: ExprTree::Const { value: 0.5 } },
                ],
                pool_over: None,
            }),
        }],
        definitions: vec![DefinitionDecl {
            name: "N".into(),
            indices: vec![],
            shape: None,
            kind: None,
            dim: None,
            body: n_body,
        }],
        tables: vec![],
        time_inputs: vec![],
        timepoints: vec![],
        transitions: vec![TransitionDecl {
            name: "infection".into(),
            indices: vec![IndexBinding::In { var: "a".into(), dim: "age".into() }],
            from: vec![StoichRef {
                compartment: "S".into(),
                indices: vec![IndexItem::Positional { value: name("a", NameKind::IndexVar) }],
            }],
            to: Destination::Sum {
                compartments: vec![StoichRef {
                    compartment: "I".into(),
                    indices: vec![IndexItem::Positional { value: name("a", NameKind::IndexVar) }],
                }],
            },
            dynamics: Dynamics::Rate { rate },
            guard: None,
            lineage: false,
            expands_to: Some(vec!["infection_child".into(), "infection_adult".into()]),
        }],
        ode_equations: vec![],
        balance: vec![],
        initial: vec![InitialDecl {
            compartment: "I".into(),
            indices: vec![],
            bindings: vec![IndexBinding::In { var: "a".into(), dim: "age".into() }],
            value: InitialValue::Law {
                law: Distribution {
                    family: Family::Poisson,
                    args: vec![Arg { name: "rate".into(), value: ExprTree::Const { value: 5.0 } }],
                },
            },
        }],
        observations: vec![ObservationDecl {
            name: "cases".into(),
            indices: vec![],
            source: None,
            columns: vec![
                ObsColumn { name: "t".into(), role: ColumnRole::Time },
                ObsColumn { name: "cases".into(), role: ColumnRole::Value { kind: ParamKind::Count } },
            ],
            measurement: Measurement {
                scored: "cases".into(),
                likelihood: Distribution {
                    family: Family::Poisson,
                    args: vec![Arg { name: "rate".into(), value: param("beta") }],
                },
            },
            projection: Projection::Incidence { transition: "infection".into(), indices: vec![] },
            emit_schedule: Some(ScheduleCore::Every { every: days(7.0) }),
            covers: Some(Covers { form: "ending_on".into(), column: "t".into(), span: Some(days(7.0)) }),
            expands_to: Some(vec!["cases".into()]),
        }],
        interventions: vec![],
        events: vec![],
        reactive_interventions: vec![],
        scenarios: vec![ScenarioDecl {
            name: "low".into(),
            fields: vec![ScenarioField::Scale {
                values: vec![Assign { name: "beta".into(), value: lit(0.5) }],
            }],
        }],
        simulation: Some(SimulationDecl {
            from: days(0.0),
            to: days(80.0),
            dt: None,
            integrator: None,
            atol: None,
            rtol: None,
        }),
        output: None,
        quantities: vec![],
        contrasts: vec![],
    }
}

#[test]
fn a_model_survives_a_json_round_trip() {
    let model = small_model();
    let json = serde_json::to_string(&model).expect("serialize");
    let back: DeclaredModel = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(model, back);
}

#[test]
fn every_expression_node_survives_a_json_round_trip() {
    let g = Guard::And {
        lhs: Box::new(Guard::Neq { lhs: "a".into(), rhs: "b".into() }),
        rhs: Box::new(Guard::Table {
            table: "dist".into(),
            indices: vec!["a".into(), "b".into()],
            op: RelOp::Lt,
            operand: GuardOperand::Num { value: 50.0 },
        }),
    };
    let nodes = vec![
        ExprTree::Const { value: 0.1 },
        ExprTree::Unit { value: 7.0, unit: UnitLit::PerWeek },
        param("gamma"),
        ExprTree::Index {
            name: NameRef { name: "S".into(), kind: NameKind::Compartment },
            indices: vec![
                IndexItem::Positional { value: name("child", NameKind::Level) },
                IndexItem::Named { dim: "patch".into(), value: name("p", NameKind::IndexVar) },
            ],
        },
        beta_s_i(),
        ExprTree::UnOp { op: UnOp::Exp, arg: Box::new(param("kappa")) },
        ExprTree::Sum { var: "b".into(), dim: "age".into(), guard: Some(g), body: Box::new(comp("I")) },
        ExprTree::Cond {
            pred: Box::new(bin(BinOp::Gt, name("t", NameKind::Builtin), ExprTree::Const { value: 10.0 })),
            then_: Box::new(param("beta")),
            else_: Box::new(ExprTree::Const { value: 0.0 }),
        },
        ExprTree::Call {
            function: "overdispersed".into(),
            args: vec![Arg { name: "rate".into(), value: param("beta") }],
        },
        ExprTree::List {
            items: vec![ExprTree::Range {
                lo: Box::new(ExprTree::Const { value: 7.0 }),
                hi: Box::new(ExprTree::Const { value: 100.0 }),
            }],
        },
        ExprTree::ObsAccess { stream: "cases".into() },
        ExprTree::RunMember {
            run: "fitted".into(),
            namespace: RunNamespace::Quantities,
            member: "total_cases".into(),
        },
    ];
    for node in nodes {
        let json = serde_json::to_string(&node).expect("serialize");
        let back: ExprTree = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(node, back, "{json}");
    }
}

/// Whitespace, comments, redundant parentheses and source positions have no
/// place to live in a tree: the same tokens in the same grouping are one value,
/// and the serialized form names nothing but the nodes.
#[test]
fn a_tree_holds_no_location_and_no_parenthesis() {
    // `beta * S * I`, `beta*S*I  # comment` and `(beta * S) * I` all parse to
    // this one tree, so they are one value here.
    let json = serde_json::to_value(beta_s_i()).expect("serialize");
    fn keys(v: &serde_json::Value, out: &mut Vec<String>) {
        match v {
            serde_json::Value::Object(m) => {
                for (k, v) in m {
                    out.push(k.clone());
                    keys(v, out);
                }
            }
            serde_json::Value::Array(a) => a.iter().for_each(|v| keys(v, out)),
            _ => {}
        }
    }
    let mut seen = Vec::new();
    keys(&json, &mut seen);
    seen.sort();
    seen.dedup();
    assert_eq!(seen, ["bin_op", "ident", "kind", "lhs", "name", "op", "rhs"]);
}

#[test]
fn the_spelling_of_a_number_is_not_part_of_the_tree() {
    let one: ExprTree = serde_json::from_str(r#"{"const":{"value":1}}"#).expect("integer spelling");
    let one_point_zero: ExprTree = serde_json::from_str(r#"{"const":{"value":1.0}}"#).expect("decimal spelling");
    assert_eq!(one, one_point_zero);
}

#[test]
fn operand_order_is_part_of_the_tree() {
    let s_beta_i = mul(mul(comp("S"), param("beta")), comp("I"));
    assert_ne!(beta_s_i(), s_beta_i);
}

#[test]
fn grouping_is_part_of_the_tree() {
    // `beta * S * (I / N)` against `beta * S * I / N`.
    let n = || name("N", NameKind::Definition);
    let grouped = mul(mul(param("beta"), comp("S")), div(comp("I"), n()));
    let flat = div(beta_s_i(), n());
    assert_ne!(grouped, flat);
}

#[test]
fn units_are_part_of_the_tree() {
    let week_in_days = ExprTree::Unit { value: 7.0, unit: UnitLit::Days };
    let one_week = ExprTree::Unit { value: 1.0, unit: UnitLit::Weeks };
    assert_ne!(week_in_days, one_week);
    assert_ne!(week_in_days, ExprTree::Const { value: 7.0 });
}

#[test]
fn what_a_name_resolved_to_is_part_of_the_tree() {
    assert_ne!(param("N"), name("N", NameKind::Definition));
}

#[test]
fn a_bound_variable_name_is_part_of_the_tree() {
    let sum = |v: &str| ExprTree::Sum {
        var: v.into(),
        dim: "age".into(),
        guard: None,
        body: Box::new(ExprTree::Index {
            name: NameRef { name: "I".into(), kind: NameKind::Compartment },
            indices: vec![IndexItem::Positional { value: name(v, NameKind::IndexVar) }],
        }),
    };
    assert_ne!(sum("a"), sum("b"));
}

#[test]
fn an_unknown_field_is_refused() {
    let err = serde_json::from_str::<ExprTree>(r#"{"const":{"value":1.0,"loc":{"line":3}}}"#)
        .expect_err("a location field must not be accepted");
    assert!(err.to_string().contains("unknown field"), "{err}");
}
