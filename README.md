## pretrain ResNet
```
python pretrain_resnet.py --lr 0.01
```

## learning the scale
```
python scale_search.py --iter 1
python scale_training.py --iter 1 --max_epoch 200

```

## ScaleNet with RL
```
python scale_finetuning.py --data imagenet --lr 1e-2 --max_epoch 60
```
