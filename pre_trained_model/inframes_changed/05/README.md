## Template commands

### Train

```bash
python3 Train.py --method <method> --t_length 6 --dataset_type <dataset_type>
```

### Evaluate

```
python3 Evaluate.py --method <method> --t_length 6 --dataset_type <dataset_type> --model_dir ./pre_trained_model/inframes_changed/05/<model>.pth --m_items_dir ./pre_trained_model/inframes_changed/05/<keys>.pt
```

## Result

### PED1

```
#TBU
```

### PED2

```
python3 Train.py --method pred --t_length 6 --dataset_type ped2

Start time: 07/07/2022 10:41:18
Dataset:  ped2
Method:  pred
Loading dataset...
Loading dataset is finished
Model setting...
Setting up model is finished
Start training and logging into file
Training is finished
Training is taken:  1:55:19.751631

-----------------------------------------------------------------------------
python3 Evaluate.py --method pred --t_length 6 --dataset_type ped2 --model_dir ./pre_trained_model/inframes_changed/05/ped2_prediction_model.pth --m_items_dir ./pre_trained_model/inframes_changed/05/ped2_prediction_keys.pt

Start time: 07/07/2022 13:02:17
Start Evaluation of: ped2, method: pred, trained model used: ped2
Number of frames: 2010
len of anomaly score: 1950
The result of ped2
AUC: 92.98227863004777 %
Evaluate is taken:  0:07:26.765750
```

### AVENUE

```
python3 Train.py --method pred --t_length 6 --dataset_type avenue

Start time: 07/07/2022 10:57:45
Dataset:  avenue
Method:  pred
Loading dataset...
Loading dataset is finished
Model setting...
Setting up model is finished
Start training and logging into file
Training is finished
Training is taken:  12:34:46.315178

-----------------------------------------------------------------------------
python3 Evaluate.py --method pred --t_length 6 --dataset_type avenue --model_dir ./pre_trained_model/inframes_changed/05/avenue_prediction_model.pth --m_items_dir ./pre_trained_model/inframes_changed/05/avenue_prediction_keys.pt

Start time: 08/07/2022 02:02:29
Start Evaluation of: avenue, method: pred, trained model used: avenue
Number of frames: 15324
len of anomaly score: 15219
The result of avenue
AUC: 85.23050065123424 %
Evaluate is taken:  1:37:15.275950
```
