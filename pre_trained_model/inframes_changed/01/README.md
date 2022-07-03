## Template commands

### Train

```bash
python3 Train.py --method <method> --t_length 4 --dataset_type <dataset_type>

```

### Evaluate

```
python3 Evaluate.py --method <method> --dataset_type <dataset_type> --model_dir ./pre_trained_model/inframes_changed/01/<model>.pth --m_items_dir ./pre_trained_model/inframes_changed/01/<keys>.pt
```

## Result

### PED1

```
#TBU
```

### PED2

```
python3 Train.py --method pred --t_length 4 --dataset_type ped2

Start time: 29/06/2022 15:04:45
Dataset: ped2
Method: pred
Loading dataset...
Loading dataset is finished
Model setting...
Setting up model is finished
Start training and logging into file
Training is finished
Training is taken: 1:58:53.648046

-----------------------------------------------------------------------------
python3 Evaluate.py --method pred --dataset_type ped2 --t_length 4 --model_dir ./pre_trained_model/inframes_changed/01/ped2_prediction_model.pth --m_items_dir ./pre_trained_model/inframes_changed/01/ped2_prediction_keys.pt

Start time: 29/06/2022 17:15:25
Start Evaluation of: ped2, method: pred, trained model used: ped2
Number of frames: 2010
len of anomaly score: 1974
The result of ped2
AUC: 95.04808092544931 %
Evaluate is taken: 0:17:33.001924

```

### AVENUE

```
python3 Train.py --method pred --t_length 4 --dataset_type avenue

Start time: 29/06/2022 17:46:56
Dataset:  avenue
Method:  pred
Loading dataset...
Loading dataset is finished
Model setting...
Setting up model is finished
Start training and logging into file
Training is finished
Training is taken:  16:49:34.465922

-----------------------------------------------------------------------------
python3 Evaluate.py --method pred --dataset_type avenue --t_length=4 --model_dir ./pre_trained_model/inframes_changed/01/avenue_prediction_model.pth --m_items_dir ./pre_trained_model/inframes_changed/01/avenue_prediction_keys.pt

Start time: 30/06/2022 10:53:16
Start Evaluation of: avenue, method: pred, trained model used: avenue
Number of frames: 15324
len of anomaly score: 15261
The result of avenue
AUC: 83.88865716257415 %
Evaluate is taken:  2:12:31.125297
```
