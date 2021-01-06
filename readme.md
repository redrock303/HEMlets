# HEMlets Pose: Learning Part-Centric Heatmap Triplets for Accurate 3D Human Pose Estimation
## Kun Zhou, Xiaoguang Han, Nianjuan Jiang, Kui Jia, Jiangbo Lu

This is the office implement of **HEMlets PoSh: Learning Part-Centric Heatmap Triplets for 3D Human Pose and Shape Estimation**. Please follow the links to read the [paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Zhou_HEMlets_Pose_Learning_Part-Centric_Heatmap_Triplets_for_Accurate_3D_Human_ICCV_2019_paper.pdf) and visit the corresponding [project page](https://sites.google.com/site/hemletspose/).

We provide quick inference code to validate and visualize our results on [Human3.6M](http://vision.imar.ro/human3.6m/description.php). Brief runing instructions are given below.
1. Pre installation\
 create a new conda vitual environment\
 conda/pip install -r requirement.txt
2. Download the pre-trained model and the tiny dataset(a pre-processed testing video from Human3.6M) at [Baidu Cloud](https://pan.baidu.com/s/1pg35KvvqUK5jX8UMRk_emQ) [code:HEMs] or [Google Cloud](https://drive.google.com/drive/folders/1z8Jj0xx4SvHC-YKuw_M_c_Z4vA4HpzID).
3. Visualization and evaluation on a single video from Human3.6M.
   We implement a script for visualization and evaluation of the predicted results on Human3.6M by running the command:\
   ```bash inference.sh```\
  if 'visualize' is set to 1, it means the visualization is activated, and you will get an additional video file in the root path which records all the rendering frames. Otherwise, it will only print the P1/P2 result on the screen.\
[figure](./inference/temp.png)\
We will update this repository with the training code.


```

### Citing
If you find this code useful for your research, please consider citing the following paper:
	@Inproceedings{zhou2019hemlets,
	  Title          = {HEMlets Pose: Learning Part-Centric Heatmap Triplets for Accurate 3D Human Pose Estimation},
	  Author         = {Kun Zhou, Xiaoguang Han, Nianjuan Jiang, Kui Jia, Jiangbo Lu},
	  Booktitle      = {International Conference on Computer Vision (ICCV)},
	  Year           = {2019}
	}

	@article{zhou2020hemlets,
    title           = {HEMlets PoSh: Learning Part-Centric Heatmap Triplets for 3D Human Pose and Shape  Estimation},
    Author          = {Kun Zhou, Xiaoguang Han, Nianjuan Jiang, Kui Jia, Jiangbo Lu},
    Journal         = {arXiv preprint arXiv:2003.04894},
    Year            = {2020}
    }

### Notices
> This code implementation is only for research or educational purposes, and not freely available for commercial use or redistribution. 
If you have any problems or suggestions, please feel free to contact Kun Zhou at 1039557638@qq.com.
