## Template commands

### Train

```bash
!python Train.py --method <method> --msize 9 --dataset_type <dataset_type>

```

### Evaluate

```
!python Evaluate.py --method <method> --dataset_type <dataset_type> --model_dir ./pre_trained_model/msize_changed/01/<model>.pth --m_items_dir ./pre_trained_model/msize_changed/01/<keys>.pt
```
