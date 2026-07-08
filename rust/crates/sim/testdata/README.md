# sim test fixtures

- `seir_seasonal_closed.ir.json` — `ocaml/golden/seir_vaccine_seasonal.camdl`
  compiled fresh (carries `rate_state_grad`, which the committed `ir/golden/`
  copies predate — gh#275 emission postdates their last regeneration). Used by
  the `ode_equilibrium` warm-start oracles, which need the ∂rate/∂compartment
  Jacobian. The SIA intervention is cleared in-test → a closed seasonal SEIRS
  (N conserved). Regenerate: `camdlc ocaml/golden/seir_vaccine_seasonal.camdl`.
