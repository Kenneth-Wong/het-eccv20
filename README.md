# HET: Sketching Image Gist
Code for the ECCV 2020 paper: "[Sketching Image Gist: Human-Mimetic Hierarchical Scene Graph Generation][0]" (accepted).

The code is partly referred to the project [rowanz/neural-motifs][1] and [KaihuaTang/VCTree-Scene-Graph-Generation][6]. If you get any problem that cause you unable to run the project, you can check the issues under [rowanz/neural-motifs][1] first. 

# Dependencies
- You may follow these commands to establish the environments under Ubuntu system
```
Install Anaconda
conda update -n base conda
conda create -n motif pip python=3.6
conda install pytorch=0.3 torchvision cuda90 -c pytorch
bash install_package.sh
```

# Prepare Dataset

Please refer to [DATA.md](data/DATA.md) for data preparation. 

# Set up

1. Update the config file with the dataset paths. Follow the steps in `DATA.md`.
    - You'll also need to fix your PYTHONPATH: ```export PYTHONPATH=/home/YourName/ThePathOfYourProject``` 

2. Compile everything. run ```make``` in the main directory: this compiles the Bilinear Interpolation operation for the RoIs.

3. Pretrain VG detection. The old version involved pretraining COCO as well, but we got rid of that for simplicity. Run `./scripts/pretrain_detector.sh`
Note: You might have to modify the learning rate and batch size, particularly if you don't have 3 Titan X GPUs (which is what I used). 

    - Download the [VG150 pretrained detector checkpoint](https://drive.google.com/open?id=11zKRr2OF5oclFL47kjFYBOxScotQzArX). You need to change the "-ckpt THE_PATH_OF_INITIAL_CHECKPOINT_MODEL" under `./scripts/train_het.sh`
    - Download the [VG200 pretrained detector checkpoint](https://pan.baidu.com/s/1nYtWiLOxsDm7fbquXsbiBA) (code: fmhf). 


# How to Train / Evaluation
0. Note that, most of the parameters are under config.py. The training stages and settings are manipulated through `./scripts/train_het.sh` Each line of command in train_vctreenet.sh needs to manually indicate "-ckpt" model (initial parameters) and "-save_dir" the path to save model. We list some of our checkpoints here. 

    - HetH, PredCls/SgCls, VG150: [checkpoint](https://pan.baidu.com/s/1KoybpA_qxkgj5cSJy8s55Q) (code: yvt1)
    - HetH, SgDet, VG150: [checkpoint](https://pan.baidu.com/s/1xFwQNgRHoOuEZM4fT8ng4w) (code: n964)
    - HetH-RRM, SgCls, VG200_KR, AAP=area+sal: [checkpoint](https://pan.baidu.com/s/10xLm5RndrbKBHYEMx5Mk3w) (code: 7i9w)

# Other Things You Need To Know
- When you evaluate your model, you will find 3 metrics are printed: 1st, "R@20/50/100" is what we use to report R@20/50/100 in our paper, 2nd, "cls avg" is corresponding mean recall mR@20/50/100 proposed by our paper, "total R" is another way to calculate recall that used in some previous papers/projects, which is quite tricky and unfair, because it almost always get higher recall. 
- The tuple match rule and triplet match rule will be applied only if the RRM module is applied. 

# If this paper/project inspires your work, pls cite our work:
```
arXiv comming soon. 
```
- For more information, please visit the homepage: [kennethwong.tech](http://www.kennethwong.tech/)

[0]: https://arxiv.org/abs/xxxxx
[1]: https://github.com/rowanz/neural-motifs
[2]: https://github.com/rowanz/neural-motifs/tree/master/data/stanford_filtered
[3]: https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21768059&authkey=APvRgmSUEvf4h8s
[4]: https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21768060&authkey=ADI-fKq10g-niGk
[5]: https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21768063&authkey=ADOyKfb6MGR5seI
[6]: https://github.com/KaihuaTang/VCTree-Scene-Graph-Generation


