# anodetection-aemem

Anomaly detection using autoencoder and memory module.

## Dataset

-   [[download directly]](http://101.32.75.151:8181/dataset/)
-   [[description]](https://github.com/StevenLiuWen/ano_pred_cvpr2018)

## How to run

Use this folder structure, copy dataset into folder `./dataset/`. For example, `./dataset/ped2/`.

1. Training: You can freely define parameters with your own settings like

```
python3 Train.py --dataset_type dataset_type
```

Example for `avenue`:

```
python3 Train.py --dataset_type avenue
```

2. Evaluation

```
python3 Evaluate.py --dataset_type dataset_type --model_dir your_model.pth --m_items_dir your_m_items.pt
```

Example for `avenue`:

```
python3 Evaluate.py --dataset_type avenue --model_dir ./my_trained_model/avenue_prediction_model.pth --m_items_dir ./my_trained_model/avenue_prediction_keys.pt
```

2. Run demo app

```
python3 app.py --method pred --dataset_type dataset_type
```

Example for `avenue`:

```
python3 app.py --method pred --dataset_type avenue
```

## Model

### My trained model

#### Prediction

-   [[ped1]](https://drive.google.com/file/d/1qMFZ2umfqJTh6vw6KjrW9dbj0fRy0A-Z/view?usp=sharing)
-   [[ped2]](https://drive.google.com/file/d/1luwmkFoFFJNqgLGJEA2MoUod71EfTuHf/view?usp=sharing)
-   [[avenue]](https://drive.google.com/file/d/1_scFKFs-pNlUsQ76t35206YiYzmM-izU/view?usp=sharing)

### Paper pre-trained model

-   [[ped2 - pred]](https://drive.google.com/file/d/14RHewQ1VtEpVmo4d9b5U0OgwL8PF2VYa/view)
-   [[ped2 - recons]](https://drive.google.com/file/d/1zsqKv0jZMejsuA-JuZoWwn_pg2fwxTW7/view)
-   [[avenue - pred]](https://drive.google.com/file/d/1sSntCNvgSzdHSsSGCbDmb49PemJ0K5p1/view)
-   [[avenue - recons]](https://drive.google.com/file/d/19UDRv-8JtClX4prParZRkLvGwYbLuGvc/view)

## Works Cited

-   [[Learning Memory-guided Normality for Anomaly Detection]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Park_Learning_Memory-Guided_Normality_for_Anomaly_Detection_CVPR_2020_paper.pdf) - [[Github]](https://github.com/cvlab-yonsei/MNAD/tree/master)
