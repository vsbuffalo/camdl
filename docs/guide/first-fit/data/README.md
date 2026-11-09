# Boarding-school influenza, 1978

`in_bed.tsv` holds the number of boys confined to bed on each day of an
influenza A (H1N1) outbreak at a boarding school in the north of England.

| column   | meaning                                          |
| -------- | ------------------------------------------------ |
| `time`   | days since Sunday 22 January 1978                |
| `in_bed` | boys confined to bed that day (763 boys at risk) |

## Source

Anonymous (1978). "Influenza in a boarding school." _British Medical Journal_
1(6112): 587. The note reports the population at risk (763), the three boys in
the infirmary on 22 January, and a graph of the daily counts.

The counts were read from that graph and are distributed as the
`influenza_england_1978_school` dataset of the R package
[`outbreaks`](https://cran.r-project.org/package=outbreaks). This file keeps its
`date` and `in_bed` columns, with `date` replaced by days since 22 January.
