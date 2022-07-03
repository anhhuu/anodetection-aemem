## Template commands

### Train

```bash
python3 Train.py --method <method> --t_length 6 --msize 11 --dataset_type <dataset_type>

```

### Evaluate

```
python3 Evaluate.py --method <method> --dataset_type <dataset_type> --t_length 6 --model_dir ./pre_trained_model/inframes-and-msize_changed/04/<model>.pth --m_items_dir ./pre_trained_model/inframes-and-msize_changed/04/<keys>.pt
```

## Result

### PED1

```
python3 Train.py --method pred --t_length 6 --msize 11 --dataset_type ped1

Start time: 30/06/2022 18:14:04
Dataset:  ped1
Method:  pred
Loading dataset...
Loading dataset is finished
Model setting...
Setting up model is finished
Start training and logging into file
Training is finished
Training is taken:  5:30:18.345175

-----------------------------------------------------------------------------
python3 Evaluate.py --method pred --dataset_type ped1 --t_length 6 --model_dir./pre_trained_model/inframes-and-msize_changed/04/ped1_prediction_model.pth --m_items_dir./pre_trained_model/inframes-and-msize_changed/04/ped1_prediction_keys.pt

Start time: 03/07/2022 16:14:19
Start Evaluation of: ped1, method: pred, trained model used: ped1
Number of frames: 7200
len of anomaly score: 7092
The result of ped1
AUC: 80.68558348102147 %
Evaluate is taken:  0:04:21.960905
```

### PED2

```
python3 Train.py --method pred --t_length 6 --msize 11 --dataset_type ped2

Start time: 30/06/2022 13:13:07
Dataset:  ped2
Method:  pred
Loading dataset...
Loading dataset is finished
Model setting...
Setting up model is finished
Start training and logging into file
Training is finished
Training is taken:  1:58:58.595654

-----------------------------------------------------------------------------
python3 Evaluate.py --method pred --dataset_type ped2 --t_length 6 --model_dir ./pre_trained_model/inframes-and-msize_changed/04/ped2_prediction_model.pth --m_items_dir ./pre_trained_model/inframes-and-msize_changed/04/ped2_prediction_keys.pt

Start time: 30/06/2022 15:30:07
Start Evaluation of: ped2, method: pred, trained model used: ped2
Number of frames: 2010
len of anomaly score: 1950
The result of ped2
AUC: 95.1418463853593 %
Evaluate is taken:  0:01:08.968564
```

### AVENUE

```
#TBU
```
