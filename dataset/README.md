## Dataset

We collected 208 clinical CT images containing 2423 vertebrae. The dataset can be downloaded at [Google Drive](https://drive.google.com/drive/folders/1EM5zSf8OwBav6doZnLRaXGR-UupEyzjP?usp=sharing). 

### Original Images

The `original_dataset.zip` contains the original CT images, segmentation masks of vertebrae and annotation of Genant's Grade. The value of segmentation represent the label of vertebrae, which is encoded as following rules.

> 1-7: cervical spine: C1-C7  
> 8-19: thoracic spine: T1-T12  
> 20-25: lumbar spine: L1-L6  
> 26: sacrum - not labeled in this dataset  
> 27: cocygis - not labeled in this dataset  
> 28: additional 13th thoracic vertebra, T13  

The Genant's Grade annotation is recorded in the `annotation.yaml` file. It is a dict with the following format.

> CT filename:  
> &nbsp;&nbsp;'vertebra label': 'Genant's Grade'  
> &nbsp;&nbsp;'vertebra label': 'Genant's Grade'   
> &nbsp;&nbsp;...  

### Processed Images

Our pipeline assess vertebrae fracture with the patches of vertebrae, and you can download our pre-prosessed patches in the `processed_dataset.zip` file, which can be directly loaded by our training code.

For the detailed information about pre-processing and dataset arrangement, please refer to the `pre-process` folder in the repo.

### Pre-processed Dataset *VerSe*

To validate our method on external dataset, you can download our pre-processed *VerSe* at [Google Drive](https://drive.google.com/drive/folders/1582r45M3xWpqjRLARpLn0sNrO1QHmEDz?usp=share_link).

For the detailed information of *VerSe* dataset, please refer to its repository at [*VerSe*](https://github.com/anjany/verse).

## Prepare Your Own Dataset

If you need to train our pipeline with your own data, please follow the instruction in the `pre-process` folder to arrange the dataset. The **Original CT Images** and **Annotation of Genant's Grade** on each vertebra is needed for data preparation.
