# USCD PED2

## RUN ON COLAB (Tesla P100-PCIE-16GB) WITH MY TRAINED MODEL

```
!python Train.py --method pred --t_length 4 --msize 11 --dataset_type ped2

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
!python Evaluate.py --method pred --dataset_type ped2 --t_length=4 --model_dir ./exp/ped2/pred/log/ped2_prediction_model.pth --m_items_dir ./exp/ped2/pred/log/ped2_prediction_keys.pt

Start time: 27/06/2022 11:09:23
Start Evaluation of: ped2, method: pred, trained model used: ped2
Number of frames: 2010
len of anomaly score: 1974
The result of ped2
AUC: 96.49785942905118 %
Evaluate is taken:  0:11:37.709379
```

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
!python Train.py --method pred --t_length 4 --dataset_type ped2

Start time: 29/06/2022 15:04:45
Dataset:  ped2
Method:  pred
Loading dataset...
Loading dataset is finished
Model setting...
Setting up model is finished
Start training and logging into file
Training is finished
Training is taken:  1:58:53.648046

-----------------------------------------------------------------------------
!python Evaluate.py --method pred --dataset_type ped2 --t_length 4 --model_dir ./exp/ped2/pred/log/ped2_prediction_model.pth --m_items_dir ./exp/ped2/pred/log/ped2_prediction_keys.pt

Start time: 29/06/2022 17:15:25
Start Evaluation of: ped2, method: pred, trained model used: ped2
Number of frames: 2010
len of anomaly score: 1974
The result of ped2
AUC: 95.04808092544931 %
Evaluate is taken:  0:17:33.001924
```

# CUHK Avenue

## RUN ON COLAB (Tesla P100-PCIE-16GB) WITH MY TRAINED MODEL

```
!python Train.py --method pred --t_length 3 --msize 11 --dataset_type avenue

Start time: 26/06/2022 18:25:07
Dataset:  avenue
Method:  pred
Loading dataset...
Loading dataset is finished
Model setting...
Setting up model is finished
Start training and logging into file
Training is finished
Training is taken:  11:23:56.984436

-----------------------------------------------------------------------------
!python Evaluate.py --method pred --dataset_type avenue t_length 3 --model_dir ./exp/avenue/pred/log/avenue_prediction_model.pth --m_items_dir ./exp/avenue/pred/log/avenue_prediction_keys.pt

Start time: 27/06/2022 06:21:18
Start Evaluation of: avenue, method: pred, trained model used: avenue
Number of frames: 15324
len of anomaly score: 15282
The result of avenue
AUC: 82.09489167921429 %
Evaluate is taken:  1:39:49.617078
```

```
!python Train.py --method pred --t_length 4 --msize 11 --dataset_type avenue

Start time: 27/06/2022 15:33:34
Dataset:  avenue
Method:  pred
Loading dataset...
Loading dataset is finished
Model setting...
Setting up model is finished
Start training and logging into file
Training is finished
Training is taken:  12:30:34.931892

-----------------------------------------------------------------------------
!python Evaluate.py --method pred --dataset_type avenue --t_length=4 --model_dir ./exp/avenue/pred/log/avenue_prediction_model.pth --m_items_dir ./exp/avenue/pred/log/avenue_prediction_keys.pt

Start time: 28/06/2022 04:48:10
Start Evaluation of: avenue, method: pred, trained model used: avenue
Number of frames: 15324
len of anomaly score: 15261
The result of avenue
AUC: 84.53195755536498 %
Evaluate is taken:  1:32:23.771134
```

```
!python Train.py --method pred --t_length 4 --dataset_type avenue

Start time: 29/06/2022 17:46:56
Dataset:  avenue
Method:  pred
Loading dataset...
Loading dataset is finished
Model setting...
Setting up model is finished
Start training and logging into file
Training is finished
Training is taken:  16:49:34.465922

-----------------------------------------------------------------------------
!python Evaluate.py --method pred --dataset_type avenue --t_length 4 --model_dir ./exp/avenue/pred/log/avenue_prediction_model.pth --m_items_dir ./exp/avenue/pred/log/avenue_prediction_keys.pt

Start time: 30/06/2022 10:53:16
Start Evaluation of: avenue, method: pred, trained model used: avenue
Number of frames: 15324
len of anomaly score: 15261
The result of avenue
AUC: 83.88865716257415 %
Evaluate is taken:  2:12:31.125297
```

# USCD PED1

```
!python Train.py --method pred --lr 15e-5 --epoch 70 --dataset_type ped1

Start time: 28/06/2022 18:43:55
Dataset:  ped1
Method:  pred
Loading dataset...
Loading dataset is finished
Model setting...
Setting up model is finished
Start training and logging into file
Training is finished
Training is taken:  9:28:09.051179

-----------------------------------------------------------------------------
!python Evaluate.py --method pred --dataset_type ped1 --model_dir ./exp/ped1/pred/log/ped1_prediction_model.pth --m_items_dir ./exp/ped1/pred/log/ped1_prediction_keys.pt

Start time: 29/06/2022 04:18:37
Start Evaluation of: ped1, method: pred, trained model used: ped1
Number of frames: 7200
len of anomaly score: 7056
The result of ped1
AUC: 75.83966399516184 %
Evaluate is taken:  2:02:43.131632
```

## RUN ON COLAB (Tesla P100-PCIE-16GB) WITH MY TRAINED MODEL
