## Template commands

### Train

```bash
python3 Train.py --method <method> --msize 11 --dataset_type <dataset_type>

```

### Evaluate

```
python3 Evaluate.py --method <method> --dataset_type <dataset_type> --model_dir ./pre_trained_model/msize_changed/11/<model>.pth --m_items_dir ./pre_trained_model/msize_changed/11/<keys>.pt
```

## Result

### PED1

```
#TBU
```

### PED2

```
python3 Train.py --method pred --msize 11 --dataset_type ped2

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
python3 Evaluate.py --method pred --dataset_type ped2 --model_dir ./pre_trained_model/msize_changed/11/ped2_prediction_model.pth --m_items_dir ./pre_trained_model/msize_changed/11/ped2_prediction_keys.pt

Start time: 03/07/2022 16:11:26
Start Evaluation of: ped2, method: pred, trained model used: ped2
Number of frames: 2010
len of anomaly score: 1962
The result of ped2
AUC: 90.15522344957043 %
Evaluate is taken:  0:01:09.742093
```

### AVENUE

```
python3 Train.py --method pred --msize 11 --dataset_type avenue

Start time: 07/07/2022 13:45:09
Dataset:  avenue
Method:  pred
Loading dataset...
Loading dataset is finished
Model setting...
Setting up model is finished
Start training and logging into file
Training is finished
Training is taken:  19:45:25.600032

-----------------------------------------------------------------------------
python3 Evaluate.py --method pred --dataset_type avenue --model_dir ./pre_trained_model/msize_changed/11/avenue_prediction_model.pth --m_items_dir ./pre_trained_model/msize_changed/11/avenue_prediction_keys.pt

Start time: 08/07/2022 09:36:14
Start Evaluation of: avenue, method: pred, trained model used: avenue
Number of frames: 15324
len of anomaly score: 15240
The result of avenue
AUC: 84.58320511987081 %
Evaluate is taken:  1:32:57.848815
```
