## CT Pre-processing procedure
### 1. Segmentation and Labelling

We ultilize the segmentation tool below to get segmentation masks and labels of given CT images, please refer to its instruction. We recommend using the **docker in the verse2020 folder**, for it can be easily depolyed.

https://github.com/christianpayer/MedicalDataAugmentationTool-VerSe

After running the algorithm, the segmentation masks (end with '*_seg.nii.gz') will be generated in the `result_christian_payer` folder. We rename the segmentation masks to the filenames of input images for the next procedures.

### 2. Image Align

To facilitate training of deep learning model, the input CT images should have an identical direction matrix and isotropic pixel spacings. In this procedure, we resample the CT image to convert its direction matrix to identical matrix.

Running the script `img_align.py` to conduct the alignment. Note that the **CT image** should be resampled by `linear` interpolation and **Segmentation mask** should be resampled by `nearest` interpolation.

### 3. Crop

Running the script `crop.py` to generate patches of vertebrae. The patches will be scaled to 128x128xN arrays.

### 4. Rename

For convenience, we just attatch the Genant's Grade to the filename of each vertebrae patch. Please write your own code to rename filename to the format `uuid_{label}_{genant's grade}.nii.gz`. The label is encoded as the following rule.

> 1-7: cervical spine: C1-C7
> 8-19: thoracic spine: T1-T12
> 20-25: lumbar spine: L1-L6
> 26: sacrum - not labeled in this dataset
> 27: cocygis - not labeled in this dataset
> 28: additional 13th thoracic vertebra, T13

### 5. Dataset Split

We use yaml to record the training and test file list. It is a list with filenames of vertebrae patches. You can write you own code to generate the file lists and dump it to yaml files. You should not split vertebrae of one CT image into different set.

### 6. Dataset Arrangement

Arrange your dataset folder to the following structure.

> dataset
> ├── img
> │   ├── image1.nii.gz
> │   ├── L_Mild_ZX2723628_19_0.nii.gz
> │   ├── L_Mild_ZX2723628_20_0.nii.gz
> │   ├── L_Mild_ZX2723628_21_1.nii.gz
> │    ...
> ├── seg
> │   ├── L_Mild_ZX2723628_18_0.nii.gz
> │   ├── L_Mild_ZX2723628_19_0.nii.gz
> │   ├── L_Mild_ZX2723628_20_0.nii.gz
> │   ├── L_Mild_ZX2723628_21_1.nii.gz
> │    ...
> ├── test_file_list.WA0aVG88.yaml
> └── train_file_list.WA0aVG88.yaml

Now you can use the code in `train` folder to train your own model.