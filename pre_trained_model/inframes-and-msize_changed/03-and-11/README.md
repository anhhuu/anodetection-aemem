## Template commands

### Train

```bash
python3 Train.py --method <method> --t_length 4 --msize 11 --dataset_type <dataset_type>

```

### Evaluate

```
python3 Evaluate.py --method <method> --dataset_type <dataset_type> --t_length 4 --model_dir ./pre_trained_model/inframes-and-msize_changed/03-and-11/<model>.pth --m_items_dir ./pre_trained_model/inframes-and-msize_changed/03-and-11/<keys>.pt
```

## Result

### PED1

```
python3 Train.py --method pred --t_length 4 --msize 11 --dataset_type ped1

Start time: 30/06/2022 18:14:04
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
python3 Evaluate.py --method pred --dataset_type ped1 --t_length 4 --model_dir ./pre_trained_model/inframes-and-msize_changed/03-and-11/ped1_prediction_model.pth --m_items_dir ./pre_trained_model/inframes-and-msize_changed/03-and-11/ped1_prediction_keys.pt

Start time: 03/07/2022 16:14:19
Start Evaluation of: ped1, method: pred, trained model used: ped1
Number of frames: 7200
len of anomaly score: 7092
The result of ped1
AUC: 80.68558348102147 %
Evaluate is taken: 0:04:21.960905
```

### PED2

```
python3 Train.py --method pred --t_length 4 --msize 11 --dataset_type ped2

Start time: 27/06/2022 08:25:14
Dataset:  ped2
Method:  pred
Loading dataset...
Loading dataset is finished
Model setting...
Setting up model is finished
Start training and logging into file
Training is finished
Training is taken:  1:52:16.362414

-----------------------------------------------------------------------------
python3 Evaluate.py --method pred --dataset_type ped2 --t_length=4 --model_dir ./pre_trained_model/inframes-and-msize_changed/03-and-11/ped2_prediction_model.pth --m_items_dir ./pre_trained_model/inframes-and-msize_changed/03-and-11/ped2_prediction_keys.pt

Start time: 27/06/2022 11:09:23
Start Evaluation of: ped2, method: pred, trained model used: ped2
Number of frames: 2010
len of anomaly score: 1974
The result of ped2
AUC: 96.49785942905118 %
Evaluate is taken:  0:11:37.709379
```

### AVENUE

```
python3 Train.py --method pred --t_length 4 --msize 11 --dataset_type avenue

Start time: 27/06/2022 15:33:34
Dataset: avenue
Method: pred
Loading dataset...
Loading dataset is finished
Model setting...
Setting up model is finished
Start training and logging into file
Training is finished
Training is taken: 12:30:34.931892

-----------------------------------------------------------------------------
python3 Evaluate.py --method pred --dataset_type avenue --t_length 4 --model_dir ./pre_trained_model/inframes-and-msize_changed/03-and-11/avenue_prediction_model.pth --m_items_dir ./pre_trained_model/inframes-and-msize_changed/03-and-11/avenue_prediction_keys.pt

Start time: 28/06/2022 04:48:10
Start Evaluation of: avenue, method: pred, trained model used: avenue
Number of frames: 15324
len of anomaly score: 15261
The result of avenue
AUC: 84.53195755536498 %
Evaluate is taken: 1:32:23.771134
```
