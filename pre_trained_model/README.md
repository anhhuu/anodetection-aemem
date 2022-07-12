| Name                        | Params                        | Ped1 (%AUC) | Ped2 (%AUC) | Avenue (%AUC) |
| --------------------------- | ----------------------------- | ----------- | ----------- | ------------- |
| papers                      | `inframes = 4; msize = 10`    | x           | 96.97       | 88.52         |
| <i> defaults <b>(milestone) | <i>`inframes = 4; msize = 10` | <i>80.70    | <i>93.97    | <i>82.96      |
| inframes changed            | `inframes = 3`                | 80.64       | <b>95.05    | <b>83.89      |
| inframes changed            | `inframes = 5`                | <b>82.89    | 92.98       | <b>85.23      |
| msize changed               | `msize = 9`                   | 74.92       | <b>96.51    | 81.69         |
| msize changed               | `msize = 11`                  | 79.55       | 90.16       | <b>84.58      |
| inframes and msize changed  | `inframes = 3; msize = 9`     | <b>82.27    | 90.47       | <b>83.72      |
| inframes and msize changed  | `inframes = 3; msize = 11`    | 80.69       | <b>96.50    | <b>84.53      |
| inframes and msize changed  | `inframes = 5; msize = 9`     | 75.19       | <b>96.21    | <b>85.41      |
| inframes and msize changed  | `inframes = 5; msize = 11`    | 78.77       | <b>95.14    | <b>83.56      |
