# **Enhancing Building Footprint Extraction with Partial Occlusion by Exploring Building Integrity**

**Introduction**

![image](https://github.com/user-attachments/assets/3342724c-68ab-400e-ba7e-9e8e759be157)

**Installation**
 
Please follow the installation guide for MMSegmentation to set up the required environment. For specific steps and details, please refer to the following link: [mmseg installation](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/get_started.md#installation)


**DataPreprocess**

[WHU Building Dataset](https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html)

[Massachusetts buildings dataset](https://www.kaggle.com/datasets/balraj98/massachusetts-buildings-dataset)

[Inria aerial image dataset](https://project.inria.fr/aerialimagelabeling/)

[Aerial imagery for roof segmentation (AIRS) datasett](https://www.airs-dataset.com/)


Getting enhanced images:  

```python image_esbi_process.py ```

**Training**

```
CUDA_VISIBLE_DEVICES=0,1,2 bash tools/dist_train.sh **/bienet/RS_configs/whu_bienet_512-160k.py 3
```

```
CUDA_VISIBLE_DEVICES=0,1,2 bash tools/dist_train.sh **/bienet/RS_configs/mass_bienet_512-160k.py 3
```

```
CUDA_VISIBLE_DEVICES=0,1,2 bash tools/dist_train.sh **/bienet/RS_configs/inria_bienet_512-160k.py 3
```

```
CUDA_VISIBLE_DEVICES=0,1,2 bash tools/dist_train.sh **/bienet/RS_configs/airs_bienet_512-160k.py 3
```

**Acknowledgement**

+ [BuildFormer](https://github.com/WangLibo1995/BuildFormer) 

+ [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) 

