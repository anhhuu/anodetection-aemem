| Params                                                                              | Ped1 (%AUC) | Ped2 (%AUC) |
| ----------------------------------------------------------------------------------- | ----------- | ----------- |
| `loss_compact = 0.05, loss_separate = 0.05` <br /> `lr = 15e5, epochs = 100` <br /> | 68.53       | 90.00       |

## Template commands

### Train

```bash
python3 TrainWithSSIMLoss.py --method <method> --dataset_type <dataset_type>

```

### Evaluate

```
python3 EvaluateWithSSIMLoss.py --method <method> --dataset_type <dataset_type> --model_dir ./pre_trained_model/ssim/<model>.pth --m_items_dir ./pre_trained_model/ssim/<keys>.pt
```

### PED1

```
python3 TrainWithSSIMLoss.py \
--method pred  \
--dataset_type ped1 \
--loss_compact 0.05 \
--loss_separate 0.05 \
--lr 15e5 \
--epochs 100

-----------------------------------------------------------------------------
python3 EvaluateWithSSIMLoss.py \
--method pred  \
--dataset_type ped1 \
--model_dir ./pre_trained_model/ssim/ped1_prediction_model.pth \
--m_items_dir ./pre_trained_model/ssim/ped1_prediction_keys.pt
```

### PED2

```
python3 TrainWithSSIMLoss.py \
--method pred  \
--dataset_type ped2 \
--loss_compact 0.05 \
--loss_separate 0.05 \
--lr 15e5 \
--epochs 100

-----------------------------------------------------------------------------
python3 EvaluateWithSSIMLoss.py \
--method pred  \
--dataset_type ped2 \
--model_dir ./pre_trained_model/ssim/ped2_prediction_model.pth \
--m_items_dir ./pre_trained_model/ssim/ped2_prediction_keys.pt
```

### PED2

```
python3 TrainWithSSIMLoss.py \
--method pred  \
--dataset_type avenue \
--loss_compact 0.05 \
--loss_separate 0.05 \
--lr 15e5 \
--epochs 100

-----------------------------------------------------------------------------
python3 EvaluateWithSSIMLoss.py \
--method pred  \
--dataset_type avenue \
--model_dir ./pre_trained_model/ssim/avenue_prediction_epoch_60_model.pth \
--m_items_dir ./pre_trained_model/ssim/avenue_prediction_epoch_60_keys.pt
```
