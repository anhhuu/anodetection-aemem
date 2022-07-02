## Template commands

### Train

```bash
!python Train.py --method <method> --msize 11 --dataset_type <dataset_type>

```

### Evaluate

```
!python Evaluate.py --method <method> --dataset_type <dataset_type> --model_dir ./pre_trained_model/msize_changed/02/<model>.pth --m_items_dir ./pre_trained_model/msize_changed/02/<keys>.pt
```
