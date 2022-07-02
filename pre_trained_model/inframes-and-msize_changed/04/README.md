## Template commands

### Train

```bash
!python Train.py --method <method> --t_length 6 --msize 11 --dataset_type <dataset_type>

```

### Evaluate

```
!python Evaluate.py --method <method> --dataset_type <dataset_type> --t_length 6 --model_dir ./pre_trained_model/inframes-and-msize_changed/04/<model>.pth --m_items_dir ./pre_trained_model/inframes-and-msize_changed/04/<keys>.pt
```
