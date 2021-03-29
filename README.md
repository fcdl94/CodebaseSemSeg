# Codebase for Semantic Segmentation
## Starting code

# Implementations
* Deeplab v2/v3 with Resnet and Resnext
* Cityscapes, VOC and ADE-20K dataset

Network and BN implementation are taken from [in-placeABN](https://github.com/mapillary/inplace_abn).

# How to run the training
Check the `run.sh` file for an example.
All the parameters are available in `argparser.py`

The default folder for the logs is `logs/<opts.dataset>/<opts.name>`. 
The log is in the tensorboard format.

# Project Tree
```
.
├── README.md
├── argparser.py
├── data
│   ├── Cityscapes -> PUT here your dataset!
├── dataset -> List of implemented datasets
│   ├── __init__.py
│   ├── ade.py
│   ├── cityscapes.py
│   ├── sampler.py
│   ├── transform.py
│   ├── utils.py
│   └── voc.py
├── metrics -> Class to compute metrics (Acc, mIoU, etc)
│   ├── __init__.py
│   └── stream_metrics.py
├── models  -> DeepNN backbones (Resnet, Resnext, WiderResnet) 
│   ├── __init__.py
│   ├── densenet.py
│   ├── resnet.py
│   ├── resnext.py
│   ├── util.py
│   └── wider_resnet.py
├── modules -> Modules to build networks (DeepLab and Residual Blocks)
│   ├── __init__.py
│   ├── deeplab.py
│   ├── dense.py
│   ├── misc.py
│   └── residual.py
├── pretrained -> PUT here pretrained models (get them from https://github.com/mapillary/inplace_abn#training-on-imagenet-1k)
├── requirements.txt  
├── run.py -> Main file
├── run.sh -> Run example
├── segmentation_module.py -> Helper to create Segmentation Network
├── train.py -> Instance model and perform training/validation
└── utils
    ├── __init__.py
    ├── logger.py  -> Logger class
    ├── loss.py    -> Custom Losses
    ├── scheduler.py -> Custom Scheduler
    └── utils.py
```

For any issue, contact me or open an issue here!