## CT Pre-processing Procedure

### 1. Segmentation and Labelling

We ultilize the segmentation tool below to get segmentation masks and labels of vertebrae, please refer to its instruction. We recommend using the `docker` in the `verse2020` folder, for it can be easily depolyed.

https://github.com/christianpayer/MedicalDataAugmentationTool-VerSe

After running the algorithm, the segmentation masks (end with `*_seg.nii.gz`) will be generated in the `result_christian_payer` folder. We rename them to the filenames of input images in the next procedures.

### 2. Image Align

To facilitate training of deep learning model, the input CT images should have an identical direction matrix and isotropic pixel spacings. In this procedure, we resample the CT image to convert its direction matrix to identical matrix.

Running the script `img_align.py` to conduct the alignment. Note that the **CT image** should be resampled by `linear` interpolation and **Segmentation mask** should be resampled by `nearest` interpolation.

### 3. Crop

Running the script `crop.py` to generate patches of vertebrae. The patches will be scaled to 128x128xN arrays.

### 4. Rename

For convenience, we just attatch the Genant's Grade to the filename of each vertebrae patch. Please write your own code to rename filename to the format `{image id}_{label}_{genant's grade}.nii.gz`. The label is encoded as the following rule.

> 1-7: cervical spine: C1-C7  
> 8-19: thoracic spine: T1-T12  
> 20-25: lumbar spine: L1-L6  
> 26: sacrum - not labeled in this dataset  
> 27: cocygis - not labeled in this dataset  
> 28: additional 13th thoracic vertebra, T13  

### 5. Dataset Split

We use yaml to record the training and test file list. It is a list with filenames of vertebrae patches. You can write you own code to generate the file lists and dump it to yaml files. You should not split vertebrae of same CT image into different sets.

The yaml file is named as `train_file_list.{dataset tag}.yaml` and `test_file_list.{dataset tag}.yaml`. The `dataset tag` is an identifier to the paired train and test list. You can create multiple dataset split with different `dataset tag` to conduct the multi fold cross-validation.

### 6. Dataset Arrangement

Arrange your dataset folder to the following structure.

> dataset_root    
> ├── img  
> │   ├── image1_18_0.nii.gz  
> │   ├── image1_19_0.nii.gz  
> │   ├── image1_20_0.nii.gz  
> │   ├── image1_21_1.nii.gz  
> │    ...  
> ├── seg  
> │   ├── image1_18_0.nii.gz  
> │   ├── image1_19_0.nii.gz  
> │   ├── image1_20_0.nii.gz  
> │   ├── image1_21_1.nii.gz  
> │    ...  
> ├── test_file_list.tag.yaml  
> └── train_file_list.tag.yaml  

Now you can use the code in `train` folder to train your own model, with the argument `--dataset dataset_root --dataset_tag tag` to set the dataset.
