# Stanford Filtered data (VG150)
Adapted from [Danfei Xu](https://github.com/danfeiX/scene-graph-TF-release/blob/master/data_tools/README.md).

Follow the steps to get the dataset set up.
1. Download the VG images [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip). Extract these images to a folder and link to them in `config.py` (eg. currently I have `VG_IMAGES=data/stanford_filtered/Images`, and the extracted `VG_100K` and `VG_100K_2` are in this folder). 

2. Download the [VG metadata](http://cvgl.stanford.edu/scene-graph/VG/image_data.json). I recommend extracting it to this directory (e.g. `data/stanford_filtered/image_data.json`), or you can edit the path in `config.py`.

3. Download the [scene graphs](http://cvgl.stanford.edu/scene-graph/dataset/VG-SGG.h5) and extract them to `data/stanford_filtered/VG-SGG.h5`. 

4. Download the [scene graph dataset metadata](http://cvgl.stanford.edu/scene-graph/dataset/VG-SGG-dicts.json) and extract it to `data/stanford_filtered/VG-SGG-dicts.json`.

5. (Optional) The saliency map: We use [DSS](https://github.com/Andrew-Qibin/DSS) to generate the saliency map. Please refer to the DSS and follow their setup. We provide the script to use it, see the `data/stanford_filtered/saliencymap.py`. In this script, we use the `imdb_512.h5` as the input image dataset. You can also load the images directly with opencv or PIL, etc.


# VG200 and VG-KR
1. Download the **VG200** and **VG-KR** annotation. It contains two files: *VG200-SGG-dicts.json* and *VG200-SGG.h5*. In the *VG200-SGG.h5*, there exist indicative key relation annotations. You can obtain them on [Google Drive](https://drive.google.com/drive/folders/1g7Fmfm64Ja1cXCo1Pv0ZvreaLQpdkBQ3?usp=sharing) or [Baidu](https://pan.baidu.com/s/1DL7geH0XXlzu3UZGIbgzbg) (code: kapn).

2. Create a folder `data/vg200` and Setup the paths in `vg200/utils/config.py`. You may use the soft links to put the `Images` and `saliency_512.h5` in this folder. 

3. (Optional) You can also create the VG200/VG-KR yourself. We provide the scripts and raw data. We briefly list the necessary data here. You can refer to `data/vg200/utils/config.py` and properly set the paths. Before running the scripts, remember to fix your PYTHONPATH: ```export PYTHONPATH=/home/YourName/ThePathOfYourProject``` . All the scripts should be run from the project root. 

    1. Prepare the additional raw VG data (all of them can be found on the Visual Genome site), including:

        - *imdb\_1024.h5*, *imdb\_512.h5* (you can also use the raw images).

        - *object\_alias.txt*, *relationship\_alias.txt*.

        - *objects.json*, *relationships.json*.

    2. Prepare the word embedding vectors from GloVe. Put the data files under the folder `data/GloVe`.

    3. Download our provided raw data directly([Baidu](https://pan.baidu.com/s/1e0IhT95pmRPDxq3ay3oqSQ) (code: 8wz4)), OR, run the script to generate them yourself:

        - *captions\_to\_sg.json*. We use the Stanford Scene Graph Parser to generate it. Please refer to their project site.

        - *cleanse\_objects.jsonO*, *cleanse_relationships.json*. OR, run the `cleanse_raw_vg.py` script. 

        - *cleanse\_triplet\_match.json*. OR, run the `triplet_match.py` script. 

        - *object\_list.txt*, *predicate\_list.txt*, *predicate\_stem.txt*.

    4. Run the `vg_to_roidb.py` scriplt. It finally creates the *VG200-SGG-dicts.json* and *VG200-SGG.h5*. 


