## Updated at: 02:24 - 07/07/2022

| Name                       | Params                  | Ped1 (%AUC)          | Ped2 (%AUC)          | Avenue (%AUC)                        |
| -------------------------- | ----------------------- | -------------------- | -------------------- | ------------------------------------ |
| papers                     | x                       | x                    | 96.97                | 88.52                                |
| defaults                   | x                       | 81.96 (77.37) (80.7) | 95.11 (93.97)        | 86.55 (82.96) (training - panhhuu)   |
| inframes changed           | inframes = 3            | <b>80.63             | <b>95.05             | <b>83.89                             |
| inframes changed           | inframes = 5            |                      | 90.30 (missed model) |                                      |
| msize changed              | msize = 9               | 74.92                | <b>96.51             | 81.69                                |
| msize changed              | msize = 11              |                      | 90.16                |                                      |
| inframes and msize changed | inframes = 3 msize = 9  |                      | 92.94 (missed model) | (training - huupma)                  |
| inframes and msize changed | inframes = 3 msize = 11 | <b>80.69             | <b>96.50             | <b>84.53                             |
| inframes and msize changed | inframes = 5 msize = 9  | 75.19                | <b>96.21             | (need to evaluate - model on huupma) |
| inframes and msize changed | inframes = 5 msize = 11 | <b>78.77             | <b>95.14             | <b>83.56                             |
