# USCD PED2

```
!python Train.py --method pred --lr 1e-4 --dataset_type ped2 & L1Loss

Start time: 28/06/2022 10:32:02
Dataset:  ped2
Method:  pred
Loading dataset...
Loading dataset is finished
Model setting...
Setting up model is finished
Start training and logging into file
Training is finished
Training is taken:  3:16:21.048370

-----------------------------------------------------------------------------
Start time: 28/06/2022 20:54:01
Start Evaluation of: ped2, method: pred, trained model used: ped2
Number of frames: 2010
len of anomaly score: 1962
The result of ped2
AUC: 93.65190238971915 %
Evaluate is taken:  0:01:26.719431
```

```
!python Train.py --method pred --lr 1e-4 --dataset_type ped2

Start time: 28/06/2022 14:30:30
Dataset:  ped2
Method:  pred
Loading dataset...
Loading dataset is finished
Model setting...
Setting up model is finished
Start training and logging into file
Training is finished
Training is taken:  3:01:58.494832

-----------------------------------------------------------------------------
!python Evaluate.py --method pred --dataset_type ped2 --model_dir ./exp/ped2/pred/log/ped2_prediction_model.pth --m_items_dir ./exp/ped2/pred/log/ped2_prediction_keys.pt

Start time: 28/06/2022 17:37:10
Start Evaluation of: ped2, method: pred, trained model used: ped2
Number of frames: 2010
len of anomaly score: 1962
The result of ped2
AUC: 94.91282217890405 %
Evaluate is taken:  0:31:19.581718
```

```
!python Train.py --method pred --t_length 6 --dataset_type ped2

Start time: 29/06/2022 08:46:21
Dataset:  ped2
Method:  pred
Loading dataset...
Loading dataset is finished
Model setting...
Setting up model is finished
Start training and logging into file
Training is finished
Training is taken:  3:16:22.344766

-----------------------------------------------------------------------------
!python Evaluate.py --method pred --dataset_type ped2 --t_length 6 --model_dir ./exp/ped2/pred/log/ped2_prediction_model.pth --m_items_dir ./exp/ped2/pred/log/ped2_prediction_keys.pt

Start time: 29/06/2022 12:42:40
Start Evaluation of: ped2, method: pred, trained model used: ped2
Number of frames: 2010
len of anomaly score: 1950
The result of ped2
AUC: 90.3002496408079 %
Evaluate is taken:  0:09:43.650188
```

```
!python Train.py --method pred --msize 9 --t_length 6 --dataset_type ped2

Start time: 01/07/2022 07:26:32
Dataset:  ped2
Method:  pred
Loading dataset...
Loading dataset is finished
Model setting...
Setting up model is finished
Start training and logging into file
Training is finished
Training is taken:  1:58:54.278047

-----------------------------------------------------------------------------
!python Evaluate.py --method pred --dataset_type ped2 --t_length 6 --model_dir ./exp/ped2/pred/log/ped2_prediction_model.pth --m_items_dir ./exp/ped2/pred/log/ped2_prediction_keys.pt

Start time: 01/07/2022 09:53:02
Start Evaluation of: ped2, method: pred, trained model used: ped2
Number of frames: 2010
len of anomaly score: 1950
The result of ped2
AUC: 94.92513535241297 %
Evaluate is taken:  0:01:08.334841
```
