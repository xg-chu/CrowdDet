# Detection in Crowded Scenes: One Proposal, Multiple Predictions

This is the pytorch implementation of our paper "Detection in Crowded Scenes: One Proposal, Multiple Predictions", https://arxiv.org/abs/2003.09163, published in CVPR 2020.

Our method aiming at detecting highly-overlapped instances in crowded scenes. 

The key of our approach is to let each proposal predict a set of instances that might be highly overlapped rather than a single one in previous proposal-based frameworks. With this scheme, the predictions of nearby proposals are expected to infer the **same set** of instances, rather than **distinguishing individuals**, which is much easy to be learned. Equipped with new techniques such as EMD Loss and Set NMS, our detector can effectively handle the difficulty of detecting highly overlapped objects.

The network structure and results are shown here:

<img width=60% src="https://github.com/Purkialo/images/blob/master/CrowdDet_arch.jpg"/>
<img width=90% src="https://github.com/Purkialo/images/blob/master/CrowdDet_demo.jpg"/>

# Citation

If you use the code in your research, please cite:
```
@article{chu2020detection,
	title={Detection in Crowded Scenes: One Proposal, Multiple Predictions},
	author={Chu, Xuangeng and Zheng, Anlin and Zhang, Xiangyu and Sun, Jian},
	journal={arXiv preprint arXiv:2003.09163},
	year={2020}
}
```

# Run
1. Requirements:
    * python3.6.8, pytorch 1.4.0, cuda10.0

2. CrowdHuman data:
    * CrowdHuman is a benchmark dataset to better evaluate detectors in crowd scenarios. The dataset can be downloaded from http://www.crowdhuman.org/. The path of the dataset is set in `config.py`.

3. Install some cuda tools:
	```
	cd utils/det_tools_cuda
	python3 setup.py install
	```
	* Please note that it must be compiled with cuda!

4. Steps to run:
    * Step1:  training. More training and testing settings can be set in `config.py`.
	```
	python3 train.py
	```
    
	* Step2:  testing. If you have multiple GPUs, you can use ` -d 0-1 ` to use more GPUs.
	```
	python3 test.py -r 30
	```
    
	* Step3:  evaluating.
	```
	python3 .evaluate/compute_APMR.py --detfile ./model/crowd_emd_simple/outputs/eval_dump/dump-30.json --target_key 'box'
	python3 .evaluate/compute_JI.py --detfile ./model/crowd_emd_simple/outputs/eval_dump/dump-30.json --target_key 'box'
	```

# Models

The pre-trained model is come from Detectron2 Model Zoo: https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl

More models will be released in a few days!

# Contact

If you have any questions, please do not hesitate to contact Xuangeng Chu (xg_chu@pku.edu.cn).
