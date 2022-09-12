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

Now, the VRU dataset has been open sourced and can be downloaded from the Baidu network disk [link](https://pan.baidu.com/s/1MQZRksYKM-WOh9l9W7R5tQ?pwd=).

## Citation

If you find this dataset useful for your research, please cite our paper.

```
Bibtex
@article{
	title={Vehicle Re-identification based on UAV Viewpoint: Dataset and Method},
	author={Lu, Mingming and Xu Yongchuan and Li Haifeng},
	journal={Remote Sensing},
	DOI={}
	year={2022},
	type={Journal Article}

}
```

