# Pneumothorax Segmentation in Chest X-ray Images

This is code for my part of solution of my team for [SIIM-ACR Pneumothorax Segmentation](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation), where we ranked 50/1475 and got a silver medal. My part of solution itself ranks in the silver range.

## Data Structure
Download [data](https://www.kaggle.com/iafoss/siimacr-pneumothorax-segmentation-data-1024) and put it in ```size1024``` directory.
```
size1024/
├── mask/
│   ├── 1.2.276.0.7230010.3.1.4.8323329.300.1517875162.258081.png
│   ├── 1.2.276.0.7230010.3.1.4.8323329.301.1517875162.280319.png
│   └── ...
├── train/train
│   ├── 1.2.276.0.7230010.3.1.4.8323329.300.1517875162.258081.png
│   ├── 1.2.276.0.7230010.3.1.4.8323329.301.1517875162.280319.png
│   └── ...
├── test
│   ├── ID_0a0adf93f.png
│   ├── ID_0a4f3c934.png
│   └── ...

```
## Train
To train models run the following commands:<br/>
```$ python train.py --fold 0```<br/>
```$ python train.py --fold 1```<br/>
```$ python train.py --fold 2```<br/>
```$ python train.py --fold 3```<br/>
```$ python train.py --fold 4```<br/>
#### Flags
- `--gpus`: IDs of GPUs to train on.
- `--train_batch_size`: Batch size of train loader.
- `--epochs`: Number of epochs to train the model.
- `--fold`: Whcih fold to train.
- `--lr`: Learning rate.
- `--img_size`: Image size.
- `--num_worker`: Positive integer will turn on multi-process data loading with the specified number of loader worker processes (Check PyTorch [docs](https://pytorch.org/docs/stable/data.html)).
To monitor metrics in Tensorboard run:</br>
```tensorboard --logdir pn_logs --port=2222```
## Weights preparation
To convert weight files from PyTorch Lightning to vanilla Pytorch run the following command.<br/>
```$ python convert.py```
## Inference 
```$ python inference_ensemble.py```
#### Flags
- `--gpu`: ID of GPU to inference on.
- `--batch_size`: Batch size of test loader.
- `--epochs`: Number of epochs to train the model.
- `--num_workers`: Number of workers of test loader. 

You can [upload the result to Kaggle](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/submissions) for scoring either manually or do it through a Kaggle API (registered account is required):
~~~
$  kaggle competitions submit -c siim-acr-pneumothorax-segmentation -f subs/submission.scv -m "my submission"
