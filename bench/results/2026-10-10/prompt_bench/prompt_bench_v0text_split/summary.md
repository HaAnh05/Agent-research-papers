# Prompt bench summary

- variants: v0, new | stages: section, overview | n=3 | concurrency=8 | dry_run=False
- provider/model: zai / glm-5.3-flash | created: 2026-10-10T21:19:51
- calls: 36 (0 failed). Percentiles use linear interpolation; failed calls without usage are excluded from token statistics.

## section

| variant | ok/calls | out tok med | out tok p90 | reasoning med | reasoning p90 | sec med | sec p90 | in tok med | out spread (max/min) |
|---|---|---|---|---|---|---|---|---|---|
| new | 12/12 | 4180 | 6528.8 | 2841 | 5155.7 | 68.12 | 100.23 | 1872.5 | 1.62 |
| v0 | 12/12 | 7988.5 | 12128.3 | 6323.5 | 10732.8 | 119.39 | 193.94 | 1822.5 | 1.65 |
| **new vs v0** | | -47.7% | -46.2% | -55.1% | -52% | -42.9% | -48.3% | +2.7% | |

Quality:

- new: parse_rate=1, words_median=883.5, equations_mean=1.75, grounding_mean=0.9893
- v0: parse_rate=1, words_median=1063, equations_mean=2.5, grounding_mean=0.9901
- new vs v0 (abs for rates/grounding, % otherwise): parse_rate=+0, grounding_mean=-0.0008, words_median=-16.9%, equations_mean=-30%

## overview

| variant | ok/calls | out tok med | out tok p90 | reasoning med | reasoning p90 | sec med | sec p90 | in tok med | out spread (max/min) |
|---|---|---|---|---|---|---|---|---|---|
| new | 6/6 | 5870 | 9516 | 4590.5 | 7445 | 102.84 | 159.42 | 2184.5 | 1.54 |
| v0 | 6/6 | 8125.5 | 14998 | 6647 | 12571.5 | 136.12 | 242.11 | 1932.5 | 1.44 |
| **new vs v0** | | -27.8% | -36.6% | -30.9% | -40.8% | -24.4% | -34.2% | +13% | |

Quality:

- new: parse_rate=1, words_median=930.5, equations_mean=0, grounding_mean=0.9583, headings_ok_rate=1
- v0: parse_rate=1, words_median=1104, equations_mean=0, grounding_mean=0.9583, headings_ok_rate=1
- new vs v0 (abs for rates/grounding, % otherwise): parse_rate=+0, grounding_mean=+0, headings_ok_rate=+0, words_median=-15.7%, equations_mean=-
