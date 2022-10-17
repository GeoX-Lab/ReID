# ReID
## Vehicle Re-identification based on UAV Viewpoint: Dataset and Method

In GASNet, it is Global Attention and full-Scale Network (GASNet) for the vehicle ReID task based on UAV images.
In VRU, it is a large-scale vehicle ReID dataset named VRU (the abbreviation of Vehicle Re-identification based on UAV), which consists of 172,137 images of 15,085 vehicles captured by UAVs, through which each vehicle has multiple images from various viewpoints.
## GASNet
### Abstract:
High-resolution remote sensing images bring a large amount of data as well as challenges to traditional vision tasks. Vehicle re-identification (ReID), as an essential vision task that can utilize remote sensing images, has been widely used in suspect vehicle search, cross-border vehicle tracking, traffic behavior analysis, and automatic toll collection systems. Although there have been a large number of studies on vehicle ReID, most of them are based on fixed surveillance cameras and do not take full advantage of high-resolution remote sensing images. Compared with images collected by fixed surveillance cameras, high-resolution remote sensing images based on Unmanned Aerial Vehicles (UAVs) have the characteristics of rich viewpoints and a wide range of scale variations. These characteristics bring richer information to vehicle ReID tasks and have the potential to improve the performance of vehicle ReID models. However, to the best of our knowledge, there is a shortage of large open-source datasets for vehicle ReID based on UAV views, which is not conducive to promoting UAV-view-based vehicle ReID research. To address this issue, we construct a large-scale vehicle ReID dataset named VRU (the abbreviation of Vehicle Re-identification based on UAV), which consists of 172,137 images of 15,085 vehicles captured by UAVs, through which each vehicle has multiple images from various viewpoints. Compared with the existing vehicle ReID datasets based on UAVs, the VRU dataset has a larger volume and is fully open source. Since most of the existing vehicle ReID methods are designed for fixed surveillance cameras, it is difficult for these methods to adapt to UAV-based vehicle ReID images with multi-viewpoint and multi-scale characteristics. Thus, this work proposes a Global Attention and full-Scale Network (GASNet) for the vehicle ReID task based on UAV images. To verify the effectiveness of our GASNet, GASNet is compared with the baseline models on the VRU dataset. The experiment results show that GASNet can achieve 97.45% Rank-1 and 98.51% mAP, which outperforms those baselines by 3.43%/2.08% improvements in term of Rank-1/mAP. Thus, our major contributions can be summarized as follows: (1) the provision of an open-source UAV-based vehicle ReID dataset, (2) the proposal of a state-of-art model for UAV-based vehicle ReID.

### Examples
Please download the pre-train model from this link and put it in the ./weights/pre_train/ folder.
https://pan.baidu.com/s/1XPSgZI92ClK8lcas_v9sRg?pwd=hqj0

Please download the VRU dataset from this link and put it in the ./datasets/ folder.
https://github.com/GeoX-Lab/ReID/tree/main/VRU

train
```
python main_gasnet.py
```

test

```
python main_gasnet.py --evaluate

```
### Citation
If you find this code or dataset useful for your research, please cite our paper.

```Bibtex
@Article{rs14184603,
	AUTHOR = {Lu, Mingming and Xu, Yongchuan and Li, Haifeng},
	TITLE = {Vehicle Re-Identification Based on UAV Viewpoint: Dataset and Method},
	JOURNAL = {Remote Sensing},
	VOLUME = {14},
	YEAR = {2022},
	NUMBER = {18},
	ARTICLE-NUMBER = {4603},
	URL = {https://www.mdpi.com/2072-4292/14/18/4603},
	ISSN = {2072-4292},
	DOI = {10.3390/rs14184603}
```}
