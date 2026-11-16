# Teams report -- valid.parquet

Split valid, 58 clips, oracle boxes: False. Unit: a track. Intervals: clips resampled.

| rule | unit | n | coverage | accuracy (covered) | macro-F1, unknown = error [95 % CI] |
|---|---|---|---|---|---|
| frame-level | detection | 122790 | 88.3% | 0.943 | 0.885 [0.876, 0.894] |
| track-level | detection | 122790 | 91.1% | 0.987 | 0.942 [0.934, 0.950] |
| track-level | track | 2349 | 87.9% | 0.981 | 0.917 [0.910, 0.925] |
| majority of frame decisions | track | 2349 | 96.8% | 0.952 | 0.936 [0.930, 0.943] |

Track-level decisions per role (a referee should be unknown):

| role | tracks | left | right | unknown |
|---|---|---|---|---|
| goalkeeper | 116 | 16 | 6 | 94 |
| no match | 333 | 75 | 52 | 206 |
| other | 1 | 1 | 0 | 0 |
| player | 2233 | 1018 | 1024 | 191 |
| referee | 230 | 29 | 20 | 181 |

Confusion per track (rows: ground truth, columns: track-level prediction):

| gt_team   |   left |   right |   unknown |
|:----------|-------:|--------:|----------:|
| left      |   1014 |      20 |       141 |
| none      |    105 |      72 |       387 |
| right     |     20 |    1010 |       144 |
