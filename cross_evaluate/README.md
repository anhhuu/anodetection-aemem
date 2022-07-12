# Cross evaluate

| Model/Dataset | Ped1 (%AUC) | Ped2 (%AUC) | Avenue (%AUC) |
| ------------- | ----------- | ----------- | ------------- |
| Ped1 Model    | <b>80.70    | 81.20       | 81.63         |
| Ped2 Model    | 72.69       | <b>93.97    | 78.36         |
| Avenue Model  | 69.84       | 83.64       | <b>82.96      |

## 1. Ped1

#### With Ped2 default model

```
python3 Evaluate.py \
--dataset_type ped1 \
--model_dir ./pre_trained_model/defaults/ped2_prediction_model.pth \
--m_items_dir ./pre_trained_model/defaults/ped2_prediction_keys.pt
```

#### With Avenue default model

```
python3 Evaluate.py \
--dataset_type ped1 \
--model_dir ./pre_trained_model/defaults/avenue_prediction_model.pth \
--m_items_dir ./pre_trained_model/defaults/avenue_prediction_keys.pt
```

## 1. Ped2

#### With Ped1 default model

```
python3 Evaluate.py \
--dataset_type ped2 \
--model_dir ./pre_trained_model/defaults/ped1_prediction_model.pth \
--m_items_dir ./pre_trained_model/defaults/ped1_prediction_keys.pt
```

#### With Avenue default model

```
python3 Evaluate.py \
--dataset_type ped2 \
--model_dir ./pre_trained_model/defaults/avenue_prediction_model.pth \
--m_items_dir ./pre_trained_model/defaults/avenue_prediction_keys.pt
```

## 1. Avenue

#### With Ped1 default model

```
python3 Evaluate.py \
--dataset_type avenue \
--model_dir ./pre_trained_model/defaults/ped1_prediction_model.pth \
--m_items_dir ./pre_trained_model/defaults/ped1_prediction_keys.pt
```

#### With Ped2 default model

```
python3 Evaluate.py \
--dataset_type avenue \
--model_dir ./pre_trained_model/defaults/ped2_prediction_model.pth \
--m_items_dir ./pre_trained_model/defaults/ped2_prediction_keys.pt
```
