# SOCIS_2019
This project lies on the complexity of detecting whales in satelite images. In particular, only the Mediterranean Sea is considered. 

## Requirements
- PyTorch 1.1.0
- tqdm
- pandas
- opencv

## How to run it
For the vainilla SRGAN (standard loss) 
```
cd SRGAN
CUDA_VISIBLE_DEVICES=0 python train.py --upscale_factor [2,4] --num_epochs 30 --hr_size [322,644] --batch_size [2,4,8,...] --data_aug
```
Then, generate dataset with the trained generator
```
cd SRGAN
CUDA_VISIBLE_DEVICES=0 python whole_pipe.py --upscale_factor [2,4] --generate_dataset 
```
Train ResNet-18 or 50 in the new generated dataset
```
cd multiclass_classification
CUDA_VISIBLE_DEVICES=0 python kfold.py --upscale_factor [2, 4] --classifier_name [resnet18,resnet50] --batch_size [2,4,8,...] 
```
Fine-tune generator with the contribution of the previously trained classifier
```
cd SRGAN
CUDA_VISIBLE_DEVICES=0 python train.py --upscale_factor [2,4] --whole_pipe --lambda_class --num_epochs 30 --hr_size [322,644] --batch_size [2,4,8,...] --data_aug
```
Generate dataset with the modified generator
```
cd SRGAN
CUDA_VISIBLE_DEVICES=0 python whole_pipe.py --upscale_factor [2,4] --generate_dataset --whole_pipe --lambda_class
```

Test classifier in the different datasets
```
cd multiclass_classification
CUDA_VISIBLE_DEVICES=0 python predict.py --upscale_factor [2,4] [--whole_pipe] --lambda_class [2,1,0.1,0.01,0.001] --classifier_name [resnet18,resnet50]
```

## Dataset 
In this project we have used two datasets: a **satellite image dataset** with low-resolution images acquired from Google and Bing Maps and labelled by ourselves, and a **super-resolution (SR) dataset** to train the SRGAN to learn a correct mapping from low-resolution (LR) to high-resolution (HR) images. 

The satellite dataset consists on seven classes: clouds, dynamic ship, rocks, static ship, ship, water and whale. The dataset is unbalanced since whales is a species in danger of extinction and there are not many specimens. What is even harder is finding them on the ocean surface. 
<div align="center">
  <img src="result_images/sat_dat.png" width="850" height="160">
</div>

<div align="center">
  <img src="result_images/proportions.png" width="300" height="200">
</div>

The SR dataset consists of images with high resolution and high frequency information. This is key when training super resolution models as they need to downsample the HR image to get the LR and then learn the mapping to recover high frequencies. This training is not possible with satellite images as there is not high frequency information on them.

<div align="center">
  <img src="result_images/sr_1.jpg" width="350" height="300">
  <img src="result_images/sr_2.jpg" width="350" height="300">
</div>


## Steps
* Image gathering from the first 3.5km margin offshore. Subsequently the area is sampled in 71x71m plots (double of a regular blue whale of 30m). The plot coordinates are then requested to Google Maps and Bing Maps. The former uses the GCP framework and the static maps API. Then with the unique key assigned for the use of the API up to 150000 requests can be made to extract the plot images. The latter uses the code in [this repository](https://github.com/manurare/Satellite-Aerial-Image-Retrieval.git) to read a .csv file where each line indicates the coordinates of a particular plot.

<div align="center">
  <img src="result_images/map_noGrid.png" width="350" height="300">
  <img src="result_images/grid.png" width="350" height="300">
</div>

* Due to the lack of spatial resolution in satellite images a superresolution step is required to increase high frequencies and restore fine details. SRGAN and Pixel Recursive Super Resolution were tested. We verified that the best approach was to use SRGAN. The latter is an old architecture and does not take on account precisely the global image information. Furthermore, it takes more than 100h per image when predicting resulting in unacceptable computational time comsuption. SRGAN on the other hand, presents fast prediction times (15fps) and the resulting super-resolved image is sharper and objectively better than with PRSR.

* Implement a new loss on the generator to generate not only SR images but images belonging to specific classes in which inherent features are enhanced to ease the classification task. The **first** step is to train both the **SRGAN** only with the **SR dataset** thus giving the mapping among LR and HR and the **ResNet-18/50** to on the **satellite dataset** to learn the labelling of the seven different classes. Then, we **fine-tune the SRGAN** using the **satellite dataset** by iteratively adding the loss to the generator coming from the prediction that ResNet-18/50 computes on the specific image. Thus, the generator has a bigger loss than the default one and will have to output images coherent with the minimization of the new loss, i.e., minimize at the same time the ResNet-18/50 error on the classification.

<div align="center">
  <img src="result_images/new_loss.png" width="550" height="280">
  <img src="result_images/train_flow.png" width="550" height="280">
</div>

* With the new generator configurations (variations of &#955;) new versions from the satellite dataset are created to eventually train ResNet-18/50 on them and test their performance with respect to the native satellite dataset. Thus, with the results obtained with the classifiers on the different datasets we check whether the modified generator was succesfull into reducing false positives. 

## Results
In this [link](https://www-iuem.univ-brest.fr/datacube/sample-apps/rshiny_app/) you can check an interactive map where different raster data layers can be uploaded (rds files) and modified to be overlapped, combined and weighted among them with higher and lower intensity to evaluate visually different aspects from satellite data. **Test rds files can be found in SOCIS_2019/rshiny_app/rds_files**

- _whales_reduced.rds_ file shows information about whales around the globe. Each dot represents one whale. They have been reduced as there were 10k points where data was available making the server unable to render such data. Therefore, we have only considered 2k samples.
- _mefts.rds_ depicts marine ecosystem functional types. This data range is [1,64] where each value represents a different type. The way it is represented is that higher types are plot with higher diameter dots and viceversa.
- _mefts_and_whales.rds_ is the combination of whale presence with the mefts by doing a multiplication in the coordinates where it concides to be both data.

### Metric Results for ResNet-18
|Dataset|Accuracy|Whale Score|
| --- | --- | --- |
| Original SD | **0.946** | **149.163** |
| Bicubic x2 | 0.938 | 83.974 |
| SRGAN x2 | 0.932 | 107.034 |
| SRGAN x2 + ResNet-18, &#955;=2 | 0.932 | 100.715 |
| SRGAN x2 + ResNet-18, &#955;=1 | 0.924 | 125.493 |
| SRGAN x2 + ResNet-18, &#955;=0.1 | 0.918 | 119.689 |
| SRGAN x2 + ResNet-18, &#955;=0.01 | 0.922 | 103.630 |
| SRGAN x2 + ResNet-18, &#955;=0.001 | 0.920 | 116.397 |

### Metric Results for ResNet-50
|Dataset|Accuracy|Whale Score|
| --- | --- | --- |
| Original SD | 0.886 | 92.016 |
| Bicubic x2 | 0.890 | 79.944 |
| SRGAN x2 | **0.900** | **121.293** |
| SRGAN x2 + ResNet-50, &#955;=2 | 0.864 | 62.844 |
| SRGAN x2 + ResNet-50, &#955;=1 | 0.868 | 86.003 |
| SRGAN x2 + ResNet-50, &#955;=0.1 | 0.852 | 71.302 |
| SRGAN x2 + ResNet-50, &#955;=0.01 | 0.798 | 53.280 |
| SRGAN x2 + ResNet-50, &#955;=0.001 | **0.874** | **148.354** |

### Visual result of our method
On the top image it is depicted the raw satellite image whereas in the bottom our super resolution method is considered.
<div align="center">
  <img src="result_images/zoomed_result.jpg" width="500" height="500">
</div>
