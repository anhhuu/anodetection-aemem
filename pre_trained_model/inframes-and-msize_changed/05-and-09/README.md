## Template commands

### Train

```bash
python3 Train.py --method <method> --t_length 6 --msize 9 --dataset_type <dataset_type>

```

### Evaluate

```
python3 Evaluate.py --method <method> --dataset_type <dataset_type> --t_length 6 --model_dir ./pre_trained_model/inframes-and-msize_changed/05-and-09/<model>.pth --m_items_dir ./pre_trained_model/inframes-and-msize_changed/05-and-09/<keys>.pt
```

## Result

### PED1

```
python3 Train.py --method pred --t_length 6 --msize 9 --dataset_type ped1

Start time: 03/07/2022 18:54:07
Dataset:  ped1
Method:  pred
Loading dataset...
Loading dataset is finished
Model setting...
Setting up model is finished
Start training and logging into file
Training is finished
Training is taken:  5:29:09.692874

-----------------------------------------------------------------------------
python3 Evaluate.py --method pred --dataset_type ped1 --t_length 6 --model_dir ./pre_trained_model/inframes-and-msize_changed/05-and-09/ped1_prediction_model.pth --m_items_dir ./pre_trained_model/inframes-and-msize_changed/05-and-09/ped1_prediction_keys.pt

Start time: 04/07/2022 12:00:47
Start Evaluation of: ped1, method: pred, trained model used: ped1
Number of frames: 7200
len of anomaly score: 7020
The result of ped1
AUC: 75.19436513810858 %
Evaluate is taken:  0:04:32.522070
```

### PED2

```
python3 Train.py --method pred --t_length 6 --msize 9 --dataset_type ped2

Start time: 03/07/2022 14:55:45
Dataset:  ped2
Method:  pred
Loading dataset...
Loading dataset is finished
Model setting...
Setting up model is finished
Start training and logging into file
Training is finished
Training is taken:  1:59:10.936742

-----------------------------------------------------------------------------
python3 Evaluate.py --method pred --dataset_type ped2 --t_length 6 --model_dir ./pre_trained_model/inframes-and-msize_changed/05-and-09/ped2_prediction_model.pth --m_items_dir ./pre_trained_model/inframes-and-msize_changed/05-and-09/ped2_prediction_keys.pt

Start time: 03/07/2022 17:30:09
Start Evaluation of: ped2, method: pred, trained model used: ped2
Number of frames: 2010
len of anomaly score: 1950
The result of ped2
AUC: 96.20682106254634 %
Evaluate is taken:  0:16:23.659972
```

### AVENUE

```
python3 Train.py --method pred --t_length 6 --msize 9 --dataset_type avenue

Start time: 05/07/2022 05:22:26
Dataset:  avenue
Method:  pred
Loading dataset...
Loading dataset is finished
Model setting...
Setting up model is finished
Start training and logging into file
Training is finished
Training is taken:  18:29:02.114667

-----------------------------------------------------------------------------
python3 Evaluate.py --method pred --dataset_type avenue --t_length 6 --model_dir ./pre_trained_model/inframes-and-msize_changed/05-and-09/avenue_prediction_model.pth --m_items_dir ./pre_trained_model/inframes-and-msize_changed/05-and-09/avenue_prediction_keys.pt

Start time: 07/07/2022 07:45:05
Start Evaluation of: avenue, method: pred, trained model used: avenue
Number of frames: 15324
len of anomaly score: 15219
The result of avenue
AUC: 85.41317335587276 %
Evaluate is taken:  1:41:01.854575
```
