# Prompt bench summary

- variants: v0, new | stages: cards | n=3 | concurrency=8 | dry_run=False
- provider/model: zai / glm-5.3-flash | created: 2026-10-10T21:48:52
- calls: 24 (0 failed). Percentiles use linear interpolation; failed calls without usage are excluded from token statistics.

## cards

| variant | ok/calls | out tok med | out tok p90 | reasoning med | reasoning p90 | sec med | sec p90 | in tok med | out spread (max/min) |
|---|---|---|---|---|---|---|---|---|---|
| new | 12/12 | 15661 | 21719.5 | 15102 | 21138 | 207.79 | 283.01 | 3538 | 1.62 |
| v0 | 12/12 | 6693.5 | 10966.7 | 6079.5 | 10416.2 | 97.02 | 146.05 | 3590 | 2.75 |
| **new vs v0** | | +134% | +98% | +148.4% | +102.9% | +114.2% | +93.8% | -1.4% | |

Quality:

- new: parse_rate=1, complete_rate=0.8333, cards_complete_mean=4.8333, status_counts={'complete': 10, 'partial': 2}
- v0: parse_rate=1, complete_rate=0.5833, cards_complete_mean=4.5833, status_counts={'complete': 7, 'partial': 5}
- new vs v0 (abs for rates/grounding, % otherwise): parse_rate=+0, complete_rate=+0.25, cards_complete_mean=+5.5%
