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
#TBU
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
!python Evaluate.py --method pred --dataset_type ped2 --t_length 6 --model_dir ./pre_trained_model/inframes-and-msize_changed/05-and-09/ped2_prediction_model.pth --m_items_dir ./pre_trained_model/inframes-and-msize_changed/05-and-09/ped2_prediction_keys.pt

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
#TBU
```
