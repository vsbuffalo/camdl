# Doctest self-test fixture

Block order is load-bearing: `test_doctest.ml` asserts the verdict of each
block by position. Do not reorder without updating the test.

<!-- camdl-doctest-preamble: demo
compartments { S, I, R }
parameters {
  N0    : positive
  I0    : positive
  gamma : rate
}
transitions {
  recovery : I --> R @ gamma * I
}
-->

<!-- camdl-doctest-data: data/pop.tsv
patch	pop
north	50000
south	30000
-->

## 1. clean complete model → pass

```camdl
compartments { S, I, R }
parameters {
  beta  : rate
  gamma : rate
}
let N = S + I + R
transitions {
  infection : S --> I @ beta * S * I / N
  recovery  : I --> R @ gamma * I
}
```

## 2. bare expression / legend → skip:parse (E001)

```camdl
5 'days + 3 'days   → 8 'days
```

## 3. construct fragment, no compartments → skip:fragment

```camdl
transitions {
  infection : S --> I @ beta * S * I / N
}
```

## 4. external-data dependence, no inline data → skip:data

```camdl
dimensions { patch = read("data/missing.tsv", column = "patch") }
```

## 5. complete model with a real dimensional error → fail (E300)

```camdl
compartments { S, I, R }
parameters {
  beta  : rate
  gamma : rate
}
transitions {
  infection : S --> I @ beta
  recovery  : I --> R @ gamma * I
}
```

## 6. explicit ignore → skip:ignore

```camdl ignore
this block is not even valid camdl and must be skipped
```

## 7. fragment + hidden preamble → pass

```camdl preamble=demo
init {
  S = N0 - I0
  I = I0
}
```

## 8. read() resolved by inline data → pass

```camdl
compartments { S, I, R }
dimensions {
  patch = read("data/pop.tsv", column = "patch")
}
stratify(by = patch)
parameters { gamma : rate }
transitions {
  recovery[p in patch] : I[p] --> R[p] @ gamma * I[p]
}
```

## 9. complete model with a syntax error (an unclosed block) → fail (E001)

```camdl
compartments { S, I, R }
parameters {
  beta  : rate
  gamma : rate
}
transitions {
  infection : S --> I @ beta * S * I
  recovery  : I --> R @ gamma * I
```

## 10. fragment whose comment mentions compartments → skip:fragment

```camdl
# the force of infection, summed over all compartments
let foi = beta * I / N
transitions {
  infection : S --> I @ foi * S
}
```

## 11. legend iterating over compartments, not declaring them → skip:parse

```camdl
[c in compartments]     iterate over integer compartment names
```
