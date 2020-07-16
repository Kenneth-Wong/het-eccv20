#!/usr/bin/env bash

# Train HET 

export CUDA_VISIBLE_DEVICES=1

# Traditional training: PredCls/SGCls/SGDet:
# add -vg200 to train on vg200 dataset.
# add -vg200_kr to train on vg200 dataset. 

if [ $1 == "0" ]; then
    echo "TRAINING HET SGDet"
    python models/train_rels.py -m sgdet -model het -order leftright -nl_obj 1 -nl_edge 1 -b 1 -clip 5 \
        -p 2000 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/vg200det/vg200-frcnn-42.tar\
        -save_dir checkpoints/het-sgdet -nepoch 50 -use_bias -use_encoded_box
elif [ $1 == "1" ]; then
    echo "TRAINING HET SGCls"
    python models/train_rels.py -m sgcls -model het -order leftright -nl_obj 1 -nl_edge 1 -b 2 -clip 3 \
        -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/vg200det/vg200-frcnn-42.tar\
        -save_dir checkpoints/het-sgcls -nepoch 50 -use_bias -use_encoded_box
elif [ $1 == "2" ]; then
    echo "TRAINING HET PredCls"
    python models/train_rels.py -m predcls -model het -order leftright -nl_obj 1 -nl_edge 1 -b 2 -clip 3 \
        -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/vg200det/vg200-frcnn-42.tar\
        -save_dir checkpoints/het-sgcls -nepoch 50 -use_bias -use_encoded_box

# following are some samples

elif [ $1 == "4" ]; then
    echo "TRAINING HET PredCls, relrank:"
    python models/train_rels.py -m predcls -model het -order leftright -nl_obj 1 -nl_edge 1 -b 2 -clip 3 \
        -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/vg200det/vg200-frcnn-42.tar\
        -save_dir checkpoints/het-sgcls -nepoch 50 -use_bias -use_encoded_box -vg200_kr -relrank -sal_input sal


fi
