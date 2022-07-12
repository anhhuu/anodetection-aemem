## Template commands

### Train

```bash
python3 Train.py --method <method> --dataset_type <dataset_type>

```

### Evaluate

```
python3 Evaluate.py --method <method> --dataset_type <dataset_type> --model_dir ./pre_trained_model/papers/<model>.pth --m_items_dir ./pre_trained_model/papers/<keys>.pt
```

python3 EvaluatePredFullFrame.py --dataset_type ped2 --model_dir ./pre_trained_model/papers/ped2_prediction_model.pth --m_items_dir ./pre_trained_model/papers/ped2_prediction_keys.pt

### PED2

```
python3 Train.py \
--method pred  \
--dataset_type ped2

-----------------------------------------------------------------------------
python3 Evaluate.py \
--method pred  \
--dataset_type ped2 \
--model_dir ./pre_trained_model/papers/ped2_prediction_model.pth \
--m_items_dir ./pre_trained_model/papers/ped2_prediction_keys.pt


```

### AVENUE

```
python3 Train.py \
--method pred  \
--dataset_type avenue

-----------------------------------------------------------------------------
python3 Evaluate.py \
--method pred  \
--dataset_type avenue \
--model_dir ./pre_trained_model/papers/avenue_prediction_model.pth \
--m_items_dir ./pre_trained_model/papers/avenue_prediction_keys.pt
```
