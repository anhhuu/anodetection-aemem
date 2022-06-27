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
!!python Evaluate.py --method pred --dataset_type ped2 --t_length=4 --model_dir ./exp/ped2/pred/log/ped2_prediction_model.pth --m_items_dir ./exp/ped2/pred/log/ped2_prediction_keys.pt

Start time: 27/06/2022 11:09:23
Start Evaluation of: ped2, method: pred, trained model used: ped2
Number of frames: 2010
len of anomaly score: 1974
The result of ped2
AUC: 96.49785942905118 %
Evaluate is taken:  0:11:37.709379
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

# USCD PED1

## RUN ON COLAB (Tesla P100-PCIE-16GB) WITH MY TRAINED MODEL
