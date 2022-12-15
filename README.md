# Faint Features Tell: Automatic Vertebrae Fracture Screening Assisted by Contrastive Learning

## Overview

This is the code repository of paper **Faint Features Tell: Automatic Vertebrae Fracture Screening Assisted by Contrastive Learning** ([arxiv](https://arxiv.org/abs/2208.10698)). 

In this paper, we propose a supervised contrastive learning based model to estimate Genent’s Grade of vertebral fracture with CT scans. Our method has a specificity of 99% and a sensitivity of 85% in binary classification, and a macro-F1 of 77% in multi-class classification. It can be concluded that forming feature space by contrastive learning could enhance CNN‘s capability of capturing faint feature and improve its performance on vertebrae fracture screening.

![overview](https://raw.githubusercontent.com/wxwxwwxxx/VertebralFractureGrading/main/overview_final.png)

Our work has been accepted by BIBM2022 as short paper.

## Data

To support the research community of medical image analysis, our dataset is **publicly available** at [Google Drive](https://drive.google.com/drive/folders/1EM5zSf8OwBav6doZnLRaXGR-UupEyzjP?usp=sharing). You can find the detailed introduction in the `dataset` folder.

Additionaly, if you are going to train our model with your own data, you can refer to the `pre-process` folder to arrange the dataset.

## Code

Our training and validaion code can be found in the `train` folder, and information about our trained weight can be found in the `model` folder.

## Citation

If you are going to use our code or data in your own work, we would be grateful if you cite our paper.

> @article{wei2022faint,  
> &nbsp;&nbsp;title={Faint Features Tell: Automatic Vertebrae Fracture Screening Assisted by Contrastive Learning},  
> &nbsp;&nbsp;author={Wei, Xin and Cong, Huaiwei and Zhang, Zheng and Peng, Junran and Chen, Guoping and Li, Jinpeng},  
> &nbsp;&nbsp;journal={arXiv preprint arXiv:2208.10698},  
> &nbsp;&nbsp;year={2022}  
> }  

## License

Our dataset is under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) License, and our code is under [MIT](https://github.com/wxwxwwxxx/VertebralFractureGrading/blob/main/LICENSE) Lisence.
