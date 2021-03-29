#!/bin/bash

alias exp='python -m torch.distributed.launch --nproc_per_node=2 run.py --batch_size 12 --opt_level O1'
shopt -s expand_aliases

exp --dataset cts --name os8_iabn-hyper --lr 2.5e-3 --epochs 360