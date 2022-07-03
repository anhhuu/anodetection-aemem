## Template commands

### Train

```bash
python3 Train.py --method <method> --t_length 4 --msize 11 --dataset_type <dataset_type>

```

### Evaluate

```
python3 Evaluate.py --method <method> --dataset_type <dataset_type> --t_length 4 --model_dir ./pre_trained_model/inframes-and-msize_changed/03-and-11/<model>.pth --m_items_dir ./pre_trained_model/inframes-and-msize_changed/03-and-11/<keys>.pt
```
