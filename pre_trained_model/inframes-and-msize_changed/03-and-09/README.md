## Template commands

### Train

```bash
python3 Train.py --method <method> --t_length 4 --msize 9 --dataset_type <dataset_type>

```

### Evaluate

```
python3 Evaluate.py --method <method> --dataset_type <dataset_type> --t_length 4 --msize 9 --model_dir ./pre_trained_model/inframes-and-msize_changed/03-and-09/<model>.pth --m_items_dir ./pre_trained_model/inframes-and-msize_changed/03-and-09/<keys>.pt
```

## Result

### PED1

```
#TBU
```

### PED2

```
python3 Train.py --method pred --t_length 4 --msize 9 --dataset_type ped2

Start time: 07/07/2022 07:49:59
Dataset:  ped2
Method:  pred
Loading dataset...
Loading dataset is finished
Model setting...
Setting up model is finished
Start training and logging into file
Training is finished
Training is taken:  1:56:35.334922

-----------------------------------------------------------------------------
python3 Evaluate.py --method pred --dataset_type ped2 --t_length 4 --msize 9 --model_dir ./pre_trained_model/inframes-and-msize_changed/03-and-09/ped2_prediction_model.pth --m_items_dir ./pre_trained_model/inframes-and-msize_changed/03-and-09/ped2_prediction_keys.pt

Start time: 07/07/2022 10:32:37
Start Evaluation of: ped2, method: pred, trained model used: ped2
DONE: 0 frames
DONE: 1000 frames
Number of frames: 2010
len of anomaly score: 1974
The result of ped2
AUC: 90.46918977131227 %
Evaluate is taken:  0:05:52.415435
```

### AVENUE

```
python3 Train.py --method pred --t_length 4 --msize 9 --dataset_type avenue

Start time: 06/07/2022 19:26:30
Dataset:  avenue
Method:  pred
Loading dataset...
Loading dataset is finished
Model setting...
Setting up model is finished
Start training and logging into file
Training is finished
Training is taken:  12:05:04.272251

-----------------------------------------------------------------------------
python3 Evaluate.py --method pred --dataset_type avenue --t_length 4 --msize 9 --model_dir ./pre_trained_model/inframes-and-msize_changed/03-and-09/avenue_prediction_model.pth --m_items_dir ./pre_trained_model/inframes-and-msize_changed/03-and-09/avenue_prediction_keys.pt

Start time: 07/07/2022 10:29:39
Start Evaluation of: avenue, method: pred, trained model used: avenue
Number of frames: 15324
len of anomaly score: 15261
The result of avenue
AUC: 83.71869124427226 %
Evaluate is taken:  0:12:44.032690
```
