# Faint Features Tell: Automatic Vertebrae Fracture Screening Assisted by Contrastive Learning

## Overview

This is the code repository of paper *Faint Features Tell: Automatic Vertebrae Fracture Screening Assisted by Contrastive Learning* ([arxiv](https://arxiv.org/abs/2208.10698)). In this paper, we propose a supervised contrastive learning based model to estimate Genent’s Grade of vertebral fracture with CT scans. Our method has a specificity of 99% and a sensitivity of 85% in binary classification, and a macro-F1 of 77% in multi-class classification. It can be concluded that forming feature space by contrastive learning facilitate CNN to capture the faint feature in the given images.

![overview](https://raw.githubusercontent.com/wxwxwwxxx/VertebralFractureGrading/main/overview_final.png)

Our work has been accepted by BIBM2022 as short paper.

## Data

To support the research community of medical image analysis, our dataset is publicly available at [Google Drive](https://drive.google.com/drive/folders/1EM5zSf8OwBav6doZnLRaXGR-UupEyzjP?usp=sharing). You can find the detailed information in the `dataset` folder.

Additionaly, if you are going to train our model on your own data, you can refer to `pre-process` folder to arrange the dataset.

## Code

Our training and validaion code can be found at `train` folder, and information about our trained weight can be found at `model` folder.

## Citation

If you use our code or data in your own work, it would be grateful if you could cite our paper.

> @article{wei2022faint,
>   title={Faint Features Tell: Automatic Vertebrae Fracture Screening Assisted by Contrastive Learning},
>   author={Wei, Xin and Cong, Huaiwei and Zhang, Zheng and Peng, Junran and Chen, Guoping and Li, Jinpeng},
>   journal={arXiv preprint arXiv:2208.10698},
>   year={2022}
> }

## License

Our dataset is under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) License, and our code is under [MIT](https://mit-license.org/) Lisence.
