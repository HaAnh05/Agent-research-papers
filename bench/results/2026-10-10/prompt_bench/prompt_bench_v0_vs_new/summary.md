# Prompt bench summary

- variants: v0, new | stages: pmrl, cards, section, overview | n=3 | concurrency=8 | dry_run=False
- provider/model: zai / glm-5.3-flash | created: 2026-10-10T21:00:44
- calls: 84 (0 failed). Percentiles use linear interpolation; failed calls without usage are excluded from token statistics.

## pmrl

| variant | ok/calls | out tok med | out tok p90 | reasoning med | reasoning p90 | sec med | sec p90 | in tok med | out spread (max/min) |
|---|---|---|---|---|---|---|---|---|---|
| new | 12/12 | 4398 | 13374 | 3705.5 | 12670.2 | 67.23 | 221.91 | 1913 | 1.51 |
| v0 | 12/12 | 6448.5 | 12204.7 | 5294 | 11134.2 | 103.32 | 225.25 | 2119 | 2.15 |
| **new vs v0** | | -31.8% | +9.6% | -30% | +13.8% | -34.9% | -1.5% | -9.7% | |

Quality:

- new: parse_rate=1, all_nonempty_rate=1, brief_summary_ok_rate=0, words_problem_median=109, words_method_median=141, words_result_median=125.5, words_limitation_median=89
- v0: parse_rate=1, all_nonempty_rate=1, brief_summary_ok_rate=0.1667, words_problem_median=169, words_method_median=204, words_result_median=180.5, words_limitation_median=141
- new vs v0 (abs for rates/grounding, % otherwise): parse_rate=+0, all_nonempty_rate=+0, brief_summary_ok_rate=-0.1667, words_problem_median=-35.5%, words_method_median=-30.9%, words_result_median=-30.5%, words_limitation_median=-36.9%

## cards

| variant | ok/calls | out tok med | out tok p90 | reasoning med | reasoning p90 | sec med | sec p90 | in tok med | out spread (max/min) |
|---|---|---|---|---|---|---|---|---|---|
| new | 12/12 | 9441.5 | 22921.9 | 8935 | 22411.4 | 134.53 | 305.42 | 3493 | 2.69 |
| v0 | 12/12 | 11538.5 | 15894.8 | 11012 | 15330.6 | 157.71 | 229.19 | 3590 | 1.97 |
| **new vs v0** | | -18.2% | +44.2% | -18.9% | +46.2% | -14.7% | +33.3% | -2.7% | |

Quality:

- new: parse_rate=1, complete_rate=0.9167, cards_complete_mean=4.9167, status_counts={'partial': 1, 'complete': 11}
- v0: parse_rate=1, complete_rate=0.75, cards_complete_mean=4.75, status_counts={'partial': 3, 'complete': 9}
- new vs v0 (abs for rates/grounding, % otherwise): parse_rate=+0, complete_rate=+0.1667, cards_complete_mean=+3.5%

## section

| variant | ok/calls | out tok med | out tok p90 | reasoning med | reasoning p90 | sec med | sec p90 | in tok med | out spread (max/min) |
|---|---|---|---|---|---|---|---|---|---|
| new | 12/12 | 17142 | 27550.9 | 16334 | 26758.5 | 212.94 | 357.08 | 1782.5 | 5.06 |
| v0 | 12/12 | 7961.5 | 10921.6 | 6477.5 | 9452.9 | 110.5 | 131.98 | 1822.5 | 1.43 |
| **new vs v0** | | +115.3% | +152.3% | +152.2% | +183.1% | +92.7% | +170.6% | -2.2% | |

Quality:

- new: parse_rate=1, words_median=545, equations_mean=0.75, grounding_mean=0.9843
- v0: parse_rate=1, words_median=1052.5, equations_mean=2.0833, grounding_mean=0.9896
- new vs v0 (abs for rates/grounding, % otherwise): parse_rate=+0, grounding_mean=-0.0053, words_median=-48.2%, equations_mean=-64%

## overview

| variant | ok/calls | out tok med | out tok p90 | reasoning med | reasoning p90 | sec med | sec p90 | in tok med | out spread (max/min) |
|---|---|---|---|---|---|---|---|---|---|
| new | 6/6 | 12136 | 24785 | 11166 | 23424 | 184.87 | 398.88 | 2189 | 1.78 |
| v0 | 6/6 | 6657.5 | 9817.5 | 5129.5 | 7485 | 114.7 | 160.22 | 1932.5 | 1.42 |
| **new vs v0** | | +82.3% | +152.5% | +117.7% | +212.9% | +61.2% | +149% | +13.3% | |

Quality:

- new: parse_rate=1, words_median=580.5, equations_mean=0.8333, grounding_mean=1, headings_ok_rate=1
- v0: parse_rate=1, words_median=1111.5, equations_mean=0, grounding_mean=0.9583, headings_ok_rate=1
- new vs v0 (abs for rates/grounding, % otherwise): parse_rate=+0, grounding_mean=+0.0417, headings_ok_rate=+0, words_median=-47.8%, equations_mean=-
