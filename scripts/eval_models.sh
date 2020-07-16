#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

if [ $1 == "0" ]; then
    python models/eval_rels.py -m predcls -model het -order leftright -nl_obj 1 -nl_edge 1 -b 6 -clip 5 \
        -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/het-predcls.tar -nepoch 50 -use_bias -use_encoded_box

elif [ $1 == "1" ]; then
    python models/eval_rels.py -m sgcls -model het -order leftright -nl_obj 1 -nl_edge 1 -b 6 -clip 5 \
        -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/het-sgcls.tar -nepoch 50 -use_bias -use_encoded_box

elif [ $1 == "2" ]; then
    python models/eval_rels.py -m sgdet -model het -order leftright -nl_obj 1 -nl_edge 1 -b 6 -clip 5 \
        -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/het-sgdet.tar -nepoch 50 -use_bias -use_encoded_box
fi
