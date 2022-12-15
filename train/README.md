## Model Training

### Environment

Our training environment is based on **CUDA 11.3** and **Pytorch 1.10**. Other python dependencies can be installed via

> pip install -r requirements.txt

### Training

We use [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) to optimize our model on two gpus. You can run the following script to train your own model.

> python -m torch.distributed.run --nproc_per_node=2 main_distributed_contrastive.py --load_ckpt

The arguments are as follows:

`'--tag', '-t'`

Experiment tag, an identifier that is used to name the ckpt subfolder and tensorboard subfolder. If the tag is `debug`, then the ckechpoint and tensorboard won't be generated.

`'--ckpt_root', '-cp'`

Checkpoint root folder, a subfolder with name `tag` will be created, and the model, log and training code will be recorded in it.

`'--tensorboard', '-tb'`

Tensorboard root folder, a subfolder with name `tag` will be created, and tensorboard file will be recorded in it.

`'--dataset', '-d'`
`'--dataset_tag', '-dt'`

Dataset root path and dataset split tag. Please refer to the `pre-process` folder in this repo.

`'--load_ckpt'`

Whether to load the checkpoint. If true, the script will try to load the model in the `ckpt_root/tag` folder, otherwise the model will be trained from scratch.

#### If your script get stuck...

In certain situation, NCCL will cause stuck issue during gpu synchronization. If your script get stuck, you can try to set environment variable `NCCL_P2P_DISABLE=1`.

### Validation

Todo.