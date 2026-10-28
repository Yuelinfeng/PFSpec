# Position Rescue Width Analysis

Round: `widths_main`; complete six-arm pairs: 60; incomplete groups: 0; sources: 30.

Throughput ratios use source-equal geometric means. Brackets give the 95% interval from 10,000 source resamples. Every complete six-arm pair is retained, including slower cases.

| Comparison | All complete | Full 512 | No new graph | New graph | Source direction (all complete) |
|---|---:|---:|---:|---:|---:|
| `w1/target` | 4.8545 [4.3534, 5.2568] | 5.0937 [4.8891, 5.2727] | 4.9575 [4.4719, 5.2869] | 4.4271 [2.9680, 5.7342] | 30 faster / 0 equal / 0 slower |
| `rescue1/target` | 4.9059 [4.3864, 5.3381] | 5.1769 [4.9669, 5.3582] | 5.0468 [4.5568, 5.3797] | 4.2524 [2.6781, 5.7279] | 30 faster / 0 equal / 0 slower |
| `rescue2/target` | 4.8700 [4.3367, 5.3170] | 5.1481 [4.9124, 5.3413] | 5.0262 [4.5419, 5.3521] | 4.1626 [2.4535, 5.8271] | 30 faster / 0 equal / 0 slower |
| `rescue3/target` | 4.8346 [4.2990, 5.2965] | 5.1005 [4.8368, 5.3181] | 5.0383 [4.5497, 5.3647] | 3.7743 [2.2610, 5.4297] | 30 faster / 0 equal / 0 slower |
| `rescue4/target` | 4.8139 [4.2775, 5.2724] | 5.0752 [4.8080, 5.2964] | 5.0449 [4.5579, 5.3708] | 3.5255 [2.2470, 4.7385] | 30 faster / 0 equal / 0 slower |
| `rescue1/w1` | 1.0106 [0.9989, 1.0184] | 1.0163 [1.0119, 1.0204] | 1.0180 [1.0152, 1.0210] | 0.9605 [0.9023, 1.0004] | 28 faster / 0 equal / 2 slower |
| `rescue2/w1` | 1.0032 [0.9829, 1.0162] | 1.0107 [1.0020, 1.0182] | 1.0139 [1.0088, 1.0188] | 0.9403 [0.8261, 1.0162] | 23 faster / 0 equal / 7 slower |
| `rescue3/w1` | 0.9959 [0.9702, 1.0156] | 1.0013 [0.9831, 1.0163] | 1.0163 [1.0122, 1.0203] | 0.8526 [0.7677, 0.9625] | 25 faster / 0 equal / 5 slower |
| `rescue4/w1` | 0.9916 [0.9630, 1.0149] | 0.9964 [0.9738, 1.0153] | 1.0176 [1.0125, 1.0224] | 0.7963 [0.7506, 0.8433] | 24 faster / 0 equal / 6 slower |
| `rescue2/rescue1` | 0.9927 [0.9828, 0.9999] | 0.9944 [0.9870, 1.0005] | 0.9959 [0.9916, 1.0000] | 0.9789 [0.9111, 1.0232] | 15 faster / 0 equal / 15 slower |
| `rescue3/rescue1` | 0.9855 [0.9700, 0.9983] | 0.9852 [0.9697, 0.9980] | 0.9983 [0.9947, 1.0017] | 0.8876 [0.8412, 0.9621] | 14 faster / 0 equal / 16 slower |
| `rescue4/rescue1` | 0.9812 [0.9617, 0.9975] | 0.9804 [0.9607, 0.9966] | 0.9996 [0.9954, 1.0035] | 0.8291 [0.7991, 0.8583] | 16 faster / 0 equal / 14 slower |
| `rescue3/rescue2` | 0.9927 [0.9795, 1.0031] | 0.9908 [0.9773, 1.0018] | 1.0024 [0.9988, 1.0062] | 0.9067 [0.8393, 0.9796] | 12 faster / 0 equal / 18 slower |
| `rescue4/rescue3` | 0.9957 [0.9852, 1.0034] | 0.9950 [0.9846, 1.0027] | 1.0013 [0.9979, 1.0046] | 0.9341 [0.8306, 1.0162] | 17 faster / 0 equal / 13 slower |

## Output and Acceptance

| Comparison | Exact sequence equal | Sequence different | Length different | EOS different | Acceptance fields with differences |
|---|---:|---:|---:|---:|---|
| `rescue1_vs_w1` | 60/60 known | 0 | 0 | 0 | none observed |
| `rescue2_vs_w1` | 60/60 known | 0 | 0 | 0 | none observed |
| `rescue3_vs_w1` | 60/60 known | 0 | 0 | 0 | none observed |
| `rescue4_vs_w1` | 60/60 known | 0 | 0 | 0 | none observed |
| `rescue2_vs_rescue1` | 60/60 known | 0 | 0 | 0 | none observed |
| `rescue3_vs_rescue2` | 60/60 known | 0 | 0 | 0 | none observed |
| `rescue4_vs_rescue3` | 60/60 known | 0 | 0 | 0 | none observed |

## Actual Backup Count Frequencies

Requested count is the fixed capacity of each arm: `rescue1`=1, `rescue2`=2, `rescue3`=3, `rescue4`=4. Actual count is derived from cumulative rank presence; every `0..k` bin is retained. A lower bin records a round where fewer than the requested nodes were available, so a 6/7/8-node layout does not mean that many nodes appeared every round. Extra-node values are cumulative rank appearances, not a per-round promise.

| Arm | Requested | Requested fixed | Valid records / total; valid rounds | Actual count frequency | Below requested | Extra nodes by rank | Rescue hits by rank |
|---|---:|:---:|---:|---|---:|---|---|
| `rescue1` | 1 | yes | 60/60; 11473 | 0:7421, 1:4052 | 7421 | 1:4052 | 1:342 |
| `rescue2` | 2 | yes | 60/60; 11473 | 0:7421, 1:2611, 2:1441 | 10032 | 1:4052, 2:1441 | 1:342, 2:53 |
| `rescue3` | 3 | yes | 60/60; 11473 | 0:7421, 1:2611, 2:1241, 3:200 | 11273 | 1:4052, 2:1441, 3:200 | 1:342, 2:53, 3:9 |
| `rescue4` | 4 | yes | 60/60; 11473 | 0:7421, 1:2611, 2:1241, 3:197, 4:3 | 11470 | 1:4052, 2:1441, 3:200, 4:3 | 1:342, 2:53, 3:9, 4:0 |

## Recovery Calls and Node Hits

| Width | Recovery calls | Backup hits | Saved calls vs W1 | Pairs with exact difference/hit match |
|---|---:|---:|---:|---:|
| `rescue1` | 709 | 342 | 342 | 60/60 |
| `rescue2` | 656 | 395 | 395 | 60/60 |
| `rescue3` | 647 | 404 | 404 | 60/60 |
| `rescue4` | 647 | 404 | 404 | 60/60 |

Node verification cost records are reported separately from decode throughput.

No per-node verification cost records were supplied.

Incomplete groups do not enter any throughput, parity, or recovery comparison. Output differences are counted as observed; no equality claim is made for divergent pairs.
