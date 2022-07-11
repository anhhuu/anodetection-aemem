## Template commands

### Train

```bash
python3 Train.py --method <method> --t_length 6 --msize 11 --dataset_type <dataset_type>

```

### Evaluate

```
python3 Evaluate.py --method <method> --dataset_type <dataset_type> --t_length 6 --model_dir ./pre_trained_model/inframes-and-msize_changed/05-and-11/<model>.pth --m_items_dir ./pre_trained_model/inframes-and-msize_changed/05-and-11/<keys>.pt
```

## Result

### PED1

```
python3 Train.py --method pred --t_length 6 --msize 11 --dataset_type ped1

Start time: 02/07/2022 18:14:04
Dataset: ped1
Method: pred
Loading dataset...
Loading dataset is finished
Model setting...
Setting up model is finished
Start training and logging into file
Training is finished
Training is taken: 5:30:18.345175

-----------------------------------------------------------------------------
python3 Evaluate.py --method pred --dataset_type ped1 --t_length 6 --model_dir ./pre_trained_model/inframes-and-msize_changed/05-and-11/ped1_prediction_model.pth --m_items_dir ./pre_trained_model/inframes-and-msize_changed/05-and-11/ped1_prediction_keys.pt

Start time: 03/07/2022 23:18:00
Start Evaluation of: ped1, method: pred, trained model used: ped1
Number of frames: 7200
len of anomaly score: 7020
The result of ped1
AUC: 78.7721995869913 %
Evaluate is taken:  0:04:17.710137
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
python3 Evaluate.py --method pred --dataset_type ped2 --t_length 6 --model_dir ./pre_trained_model/inframes-and-msize_changed/05-and-11/ped2_prediction_model.pth --m_items_dir ./pre_trained_model/inframes-and-msize_changed/05-and-11/ped2_prediction_keys.pt

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
python3 Train.py --method pred --t_length 6 --msize 11 --dataset_type avenue

Start time: 04/07/2022 05:22:26
Dataset:  avenue
Method:  pred
Loading dataset...
Loading dataset is finished
Model setting...
Setting up model is finished
Start training and logging into file
Training is finished
Training is taken:  17:29:02.114667

-----------------------------------------------------------------------------
python3 Evaluate.py --method pred --dataset_type avenue --t_length 6 --model_dir ./exp/avenue/pred/log/avenue_prediction_model.pth --m_items_dir ./exp/avenue/pred/log/avenue_prediction_keys.pt

Start time: 05/07/2022 01:24:14
Start Evaluation of: avenue, method: pred, trained model used: avenue
Number of frames: 15324
len of anomaly score: 15219
The result of avenue
AUC: 83.56389412372958 %
Evaluate is taken:  3:49:14.479455
```
