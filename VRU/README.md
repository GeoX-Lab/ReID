## VRU dataset 

We use UAVs to construct a vehicle image dataset, named VRU, for the vehicle ReID task. To collect vehicle image data under various scenes, 5 `DJI Mavic 2 Pro' UAVs are deployed. a total of 172,137 images containing 15,085 vehicle instances were obtained. The comparison between the VRU dataset and other vehicle re-identification datasets collected based on UAV is as follows:

| Datasets    | VRU     | UAV-VeID | VRAI     |
| ----------- | ------- | -------- | -------- |
| Identities  | 15085   | 4601     | 13022    |
| Images      | 172137  | 41917    | 137613   |
| Multi-view  | $\surd$ | $\surd$  | $\surd$  |
| Multi-scale | $\surd$ | $\surd$  | $\surd$  |
| Weather     | $\surd$ | $\surd$  | $\times$ |
| Lighting    | $\surd$ | $\surd$  | $\times$ |
| full-open   | $\surd$ | $\times$ | $\times$ |

Now, the VRU dataset has been open sourced and can be downloaded from the [Baidu network disk](https://pan.baidu.com/s/1s5RcJK0wAfg3INYuRjG5zw?pwd=382t) or [Google Driver](https://drive.google.com/file/d/1ESeeYeqbf1TIUChXNcevJK_0fyVGQpgZ/view?usp=share_link).
Contact: xuyongchuan@csu.edu.cn

## Citation

If you find this dataset useful for your research, please cite our paper.

```
Bibtex
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
}
```

