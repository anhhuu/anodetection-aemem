# Defaults

| Model name             | Ped1 (%AUC) | Ped2 (%AUC) | Avenue (%AUC) |
| ---------------------- | ----------- | ----------- | ------------- |
| <i> papers             | x           | <i> 96.97   | <i>88.52      |
| <i> re-trained default | <i>80.70    | <i> 93.97   | <i>82.96      |

# Combine (pred and recon)

| Model name                                   | Ped1 (%AUC)                 | Ped2 (%AUC)              | Avenue (%AUC)                          |
| -------------------------------------------- | --------------------------- | ------------------------ | -------------------------------------- |
| papers + rand                                | x                           | <b> 95.37                | <b> 88.40                              |
| <i> re-trained default + rand <b>(milestone) | <i> 80.05                   | <i> 92.33                | <i> 82.83                              |
| re-trained default + recon                   | <b> 80.36                   | <b> 93.49                | <b> 83.64                              |
| (`inframes = 3; msize = 11`) + recon         | <b> 80.38                   | <b> 96.01                | <b> 83.64                              |
| highest + recon                              | (`inframes = 5`): <b> 82.45 | (`msize = 9`): <b> 95.96 | (`inframes = 5; msize = 9`): <b> 85.30 |

NOTE: For avenue dataset, anomalies rarely appear in the early frames.

## 1. papers + rand

#### Ped2

```
python3 EvaluatePredFullFrame.py \
--dataset_type ped2 \
--model_dir ./pre_trained_model/papers/ped2_prediction_model.pth \
--m_items_dir ./pre_trained_model/papers/ped2_prediction_keys.pt
```

#### Avenue

```
python3 EvaluatePredFullFrame.py \
--dataset_type avenue \
--model_dir ./pre_trained_model/papers/avenue_prediction_model.pth \
--m_items_dir ./pre_trained_model/papers/avenue_prediction_keys.pt
```

## 2. re-trained default + rand

#### Ped1

```
python3 EvaluatePredFullFrame.py \
--dataset_type ped1 \
--model_dir ./pre_trained_model/defaults/ped1_prediction_model.pth \
--m_items_dir ./pre_trained_model/defaults/ped1_prediction_keys.pt
```

#### Ped2

```
python3 EvaluatePredFullFrame.py \
--dataset_type ped2 \
--model_dir ./pre_trained_model/defaults/ped2_prediction_model.pth \
--m_items_dir ./pre_trained_model/defaults/ped2_prediction_keys.pt
```

#### Avenue

```
python3 EvaluatePredFullFrame.py \
--dataset_type avenue \
--model_dir ./pre_trained_model/defaults/avenue_prediction_model.pth \
--m_items_dir ./pre_trained_model/defaults/avenue_prediction_keys.pt
```

## 3. re-trained default + recon

#### Ped1

```
python3 EvaluateCombine.py \
--dataset_type ped1 \
--pred_model_dir ./pre_trained_model/defaults/ped1_prediction_model.pth \
--pred_m_items_dir ./pre_trained_model/defaults/ped1_prediction_keys.pt \
--recon_model_dir ./pre_trained_model/recon/ped1_reconstruction_model.pth \
--recon_m_items_dir ./pre_trained_model/recon/ped1_reconstruction_keys.pt
```

#### Ped2

```
python3 EvaluateCombine.py \
--dataset_type ped2 \
--pred_model_dir ./pre_trained_model/defaults/ped2_prediction_model.pth \
--pred_m_items_dir ./pre_trained_model/defaults/ped2_prediction_keys.pt \
--recon_model_dir ./pre_trained_model/recon/ped2_reconstruction_model.pth \
--recon_m_items_dir ./pre_trained_model/recon/ped2_reconstruction_keys.pt
```

#### Avenue

```
python3 EvaluateCombine.py \
--dataset_type avenue \
--pred_model_dir ./pre_trained_model/defaults/avenue_prediction_model.pth \
--pred_m_items_dir ./pre_trained_model/defaults/avenue_prediction_keys.pt \
--recon_model_dir ./pre_trained_model/recon/avenue_reconstruction_model.pth \
--recon_m_items_dir ./pre_trained_model/recon/avenue_reconstruction_keys.pt
```

## 4. (`inframes = 3; msize = 11`) + recon

#### Ped1

```
python3 EvaluateCombine.py \
--dataset_type ped1 \
--t_length 4 \
--pred_model_dir ./pre_trained_model/inframes-and-msize_changed/03-and-11/ped1_prediction_model.pth \
--pred_m_items_dir ./pre_trained_model/inframes-and-msize_changed/03-and-11/ped1_prediction_keys.pt \
--recon_model_dir ./pre_trained_model/recon/ped1_reconstruction_model.pth \
--recon_m_items_dir ./pre_trained_model/recon/ped1_reconstruction_keys.pt
```

#### Ped2

```
python3 EvaluateCombine.py \
--dataset_type ped2 \
--t_length 4 \
--pred_model_dir ./pre_trained_model/inframes-and-msize_changed/03-and-11/ped2_prediction_model.pth \
--pred_m_items_dir ./pre_trained_model/inframes-and-msize_changed/03-and-11/ped2_prediction_keys.pt \
--recon_model_dir ./pre_trained_model/recon/ped2_reconstruction_model.pth \
--recon_m_items_dir ./pre_trained_model/recon/ped2_reconstruction_keys.pt
```

#### Avenue

```
python3 EvaluateCombine.py \
--dataset_type avenue \
--t_length 4 \
--pred_model_dir ./pre_trained_model/inframes-and-msize_changed/03-and-09/avenue_prediction_model.pth \
--pred_m_items_dir ./pre_trained_model/inframes-and-msize_changed/03-and-09/avenue_prediction_keys.pt \
--recon_model_dir ./pre_trained_model/recon/avenue_reconstruction_model.pth \
--recon_m_items_dir ./pre_trained_model/recon/avenue_reconstruction_keys.pt
```

## 5. highest + recon

#### Ped1

-   Highest is model: `inframes = 5`

```
python3 EvaluateCombine.py \
--dataset_type ped1 \
--t_length 6 \
--pred_model_dir ./pre_trained_model/inframes_changed/05/ped1_prediction_model.pth \
--pred_m_items_dir ./pre_trained_model/inframes_changed/05/ped1_prediction_keys.pt \
--recon_model_dir ./pre_trained_model/recon/ped1_reconstruction_model.pth \
--recon_m_items_dir ./pre_trained_model/recon/ped1_reconstruction_keys.pt
```

#### Ped2

-   Highest is model: `msize = 9`

```
python3 EvaluateCombine.py \
--dataset_type ped2 \
--pred_model_dir ./pre_trained_model/msize_changed/09/ped2_prediction_model.pth \
--pred_m_items_dir ./pre_trained_model/msize_changed/09/ped2_prediction_keys.pt \
--recon_model_dir ./pre_trained_model/recon/ped2_reconstruction_model.pth \
--recon_m_items_dir ./pre_trained_model/recon/ped2_reconstruction_keys.pt
```

#### Avenue

-   Highest is model: `inframes = 5; msize = 9`

```
python3 EvaluateCombine.py \
--t_length 6 \
--dataset_type avenue \
--pred_model_dir ./pre_trained_model/inframes-and-msize_changed/05-and-09/avenue_prediction_model.pth \
--pred_m_items_dir ./pre_trained_model/inframes-and-msize_changed/05-and-09/avenue_prediction_keys.pt \
--recon_model_dir ./pre_trained_model/recon/avenue_reconstruction_model.pth \
--recon_m_items_dir ./pre_trained_model/recon/avenue_reconstruction_keys.pt
```
