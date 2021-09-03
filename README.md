# Detection in Crowded Scenes: One Proposal, Multiple Predictions

This is the pytorch implementation of our paper "[Detection in Crowded Scenes: One Proposal, Multiple Predictions](https://openaccess.thecvf.com/content_CVPR_2020/html/Chu_Detection_in_Crowded_Scenes_One_Proposal_Multiple_Predictions_CVPR_2020_paper.html)", https://arxiv.org/abs/2003.09163, published in CVPR 2020.

Our method aiming at detecting highly-overlapped instances in crowded scenes. 

The key of our approach is to let each proposal predict a set of instances that might be highly overlapped rather than a single one in previous proposal-based frameworks. With this scheme, the predictions of nearby proposals are expected to infer the **same set** of instances, rather than **distinguishing individuals**, which is much easy to be learned. Equipped with new techniques such as EMD Loss and Set NMS, our detector can effectively handle the difficulty of detecting highly overlapped objects.

The network structure and results are shown here:

<img width=60% src="https://github.com/Purkialo/images/blob/master/CrowdDet_arch.jpg"/>
<img width=90% src="https://github.com/Purkialo/images/blob/master/CrowdDet_demo.jpg"/>

# Citation

If you use the code in your research, please cite:
```
@InProceedings{Chu_2020_CVPR,
author = {Chu, Xuangeng and Zheng, Anlin and Zhang, Xiangyu and Sun, Jian},
title = {Detection in Crowded Scenes: One Proposal, Multiple Predictions},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

# Run
1) Setup environment by docker
   - Requirements: Install [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
   - Create docker image:
   ```shell
   sudo docker build . -t crowddet
   ```
   - Run docker image:
   ```shell
   sudo docker run --gpus all --shm-size=8g -it --rm crowddet
   ```

2. CrowdHuman data:
    * CrowdHuman is a benchmark dataset to better evaluate detectors in crowd scenarios. The dataset can be downloaded from http://www.crowdhuman.org/. The path of the dataset is set in `config.py`.

3. Steps to run:
    * Step1:  training. More training and testing settings can be set in `config.py`.
	```
	cd tools
	python3 train.py -md rcnn_fpn_baseline
	```
    
	* Step2:  testing. If you have four GPUs, you can use ` -d 0-3 ` to use all of your GPUs.
			  The result json file will be evaluated automatically.
	```
	cd tools
	python3 test.py -md rcnn_fpn_baseline -r 40
	```
    
	* Step3:  evaluating json, inference one picture and visulization json file.
			  ` -r ` means resume epoch, ` -n ` means number of visulization pictures.
	```
	cd tools
	python3 eval_json.py -f your_json_path.json
	python3 inference.py -md rcnn_fpn_baseline -r 40 -i your_image_path.png 
	python3 visulize_json.py -f your_json_path.json -n 3
	```

# Models

We use MegEngine in the research (https://github.com/megvii-model/CrowdDetection), this proiect is a re-implementation based on Pytorch.
<!-- We use pre-trained model from Detectron2 Model Zoo: https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl. (or [R-50.pkl](https://drive.google.com/open?id=1qWAwY8QOhYRazxRuIhRA55b8YDxdOR8_)) -->
We use pre-trained model from [MegEngine Model Hub](https://megengine.org.cn/model-hub) and convert this model to pytorch. You can get this model from [here](https://drive.google.com/file/d/1lfYQHC63oM2Dynbfj6uD7XnpDIaA5kNr/view?usp=sharing).
These models can also be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1U3I-qNIrXuYQzUEDDdISTw)(code:yx46).
| Model | Top1 acc | Top5 acc |
| --- | --- | --- |
| ResNet50 | 76.254 | 93.056 |

All models are based on ResNet-50 FPN.
| | AP | MR | JI | Model
| --- | --- | --- | --- | --- |
| RCNN FPN Baseline (convert from MegEngine) | 0.8718 | 0.4239 | 0.7949 | [rcnn_fpn_baseline_mge.pth](https://drive.google.com/file/d/19LBc_6vizKr06Wky0s7TAnvlqP8PjSA_/view?usp=sharing) |
| RCNN EMD Simple (convert from MegEngine) | 0.9052 | 0.4196 | 0.8209 | [rcnn_emd_simple_mge.pth](https://drive.google.com/file/d/1f_vjFrjTxXYR5nPnYZRrU-yffYTGUnyL/view?usp=sharing) |
| RCNN EMD with RM (convert from MegEngine) | 0.9097 | 0.4102 | 0.8271 | [rcnn_emd_refine_mge.pth](https://drive.google.com/file/d/1qYJ0b7QsYZsP5_8yIjya_kj_tu90ALDJ/view?usp=sharing) |
| RCNN FPN Baseline (trained with PyTorch) | 0.8665 | 0.4243 | 0.7949 | [rcnn_fpn_baseline.pth](https://drive.google.com/file/d/10poBJ1qwlV0iS6i_lnbw9cbdt4tpTvh1/view?usp=sharing) |
| RCNN EMD Simple (trained with PyTorch) | 0.8997 | 0.4167 | 0.8225 | [rcnn_emd_simple.pth](https://drive.google.com/file/d/1Rryeqz5sMWTTm3epEfqDpK1H8EsPvlLe/view?usp=sharing) |
| RCNN EMD with RM (trained with PyTorch) | 0.9030 | 0.4128 | 0.8263 | [rcnn_emd_refine.pth](https://drive.google.com/file/d/1jk_b7Ws528uCfEgOesLS_iBsqXcHl2Ju/view?usp=sharing) |
| RetinaNet FPN Baseline | 0.8188 | 0.5644 | 0.7316 | [retina_fpn_baseline.pth](https://drive.google.com/file/d/1w1CmE4MfYB4NT5Uyx85dPkR87gEFXhBJ/view?usp=sharing) |
| RetinaNet EMD Simple | 0.8292 | 0.5481 | 0.7393 | [retina_emd_simple.pth](https://drive.google.com/file/d/1LwUlTf4YAH3wp-HXAAuXyTD11SeDgwhE/view?usp=sharing) |
<!-- | RCNN FPN Baseline | --- | --- | --- | --- |
| RCNN EMD Simple | --- | --- | --- | --- |
| RCNN EMD with RM | --- | --- | --- | --- | -->

# Contact

If you have any questions, please do not hesitate to contact Xuangeng Chu (xg_chu@pku.edu.cn).
