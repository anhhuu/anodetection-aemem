## Template commands

### Train

```bash
python3 Train.py --method <method> --msize 9 --dataset_type <dataset_type>

```

### Evaluate

```
python3 Evaluate.py --method <method> --dataset_type <dataset_type> --model_dir ./pre_trained_model/msize_changed/09/<model>.pth --m_items_dir ./pre_trained_model/msize_changed/09/<keys>.pt
```

## Result

### PED1

```
python3 Train.py --method pred --msize 9 --dataset_type ped1

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
python3 Evaluate.py --method pred --dataset_type ped1 --model_dir ./pre_trained_model/msize_changed/09/ped1_prediction_model.pth --m_items_dir ./pre_trained_model/msize_changed/09/ped1_prediction_keys.pt

Start time: 03/07/2022 15:53:06
Start Evaluation of: ped1, method: pred, trained model used: ped1
Number of frames: 7200
len of anomaly score: 7056
The result of ped1
AUC: 74.92010363129027 %
Evaluate is taken:  0:04:16.206267
```

### PED2

```
python3 Train.py --method pred --msize 9 --dataset_type ped2

Start time: 30/06/2022 15:49:15
Dataset:  ped2
Method:  pred
Loading dataset...
Loading dataset is finished
Model setting...
Setting up model is finished
Start training and logging into file
Training is finished
Training is taken:  1:58:51.011772

-----------------------------------------------------------------------------
python3 Evaluate.py --method pred --dataset_type ped2 --model_dir ./pre_trained_model/msize_changed/09/ped2_prediction_model.pth --m_items_dir ./pre_trained_model/msize_changed/09/ped2_prediction_keys.pt

Start time: 30/06/2022 17:53:36
Start Evaluation of: ped2, method: pred, trained model used: ped2
Number of frames: 2010
len of anomaly score: 1962
The result of ped2
AUC: 96.50638943036604 %
Evaluate is taken:  0:00:59.558563
```

### AVENUE

```
#TBU
```
