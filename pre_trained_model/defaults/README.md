## Template commands

### Train

```bash
python3 Train.py --method <method> --dataset_type <dataset_type>

```

### Evaluate

```
python3 Evaluate.py --method <method> --dataset_type <dataset_type> --model_dir ./pre_trained_model/defaults/<model>.pth --m_items_dir ./pre_trained_model/defaults/<keys>.pt
```

### PED1

```
python3 Train.py \
--method pred \
--dataset_type ped1

-----------------------------------------------------------------------------
python3 Evaluate.py --method pred \
--dataset_type ped1 \
--model_dir ./pre_trained_model/defaults/ped1_prediction_model.pth \
--m_items_dir ./pre_trained_model/defaults/ped1_prediction_keys.pt

```

### PED2

```
python3 Train.py \
--method pred  \
--dataset_type ped2

-----------------------------------------------------------------------------
python3 Evaluate.py \
--method pred  \
--dataset_type ped2 \
--model_dir ./pre_trained_model/defaults/ped2_prediction_model.pth \
--m_items_dir ./pre_trained_model/defaults/ped2_prediction_keys.pt


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
--model_dir ./pre_trained_model/defaults/avenue_prediction_model.pth \
--m_items_dir ./pre_trained_model/defaults/avenue_prediction_keys.pt
```
