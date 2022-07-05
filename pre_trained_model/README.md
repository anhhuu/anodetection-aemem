| Name                       | Params                  | Ped1 (%AUC)                      | Ped2 (%AUC)          | Avenue (%AUC)     |
| -------------------------- | ----------------------- | -------------------------------- | -------------------- | ----------------- |
| papers                     | x                       | x                                | 96.97                | 88.52             |
| defaults                   | x                       | 81.96 (77.37) training (panhhuu) | 95.11                | <b>86.55          |
| inframes changed           | inframes = 3            | <b>80.63                         | 95.05                | 83.89             |
| inframes changed           | inframes = 5            |                                  | 90.30 (missed model) |                   |
| msize changed              | msize = 9               | 74.92                            | <b>96.51             | training (huupma) |
| msize changed              | msize = 11              |                                  | 90.16                |                   |
| inframes and msize changed | inframes = 3 msize = 9  |                                  | 92.94 (missed model) |                   |
| inframes and msize changed | inframes = 3 msize = 11 | <b>80.69                         | <b>96.50             | 84.53             |
| inframes and msize changed | inframes = 5 msize = 9  | 75.19                            | <b>96.21             | (plan priority 2) |
| inframes and msize changed | inframes = 5 msize = 11 | <b>78.77                         | 95.14                | 83.56             |
