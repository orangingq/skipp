# kan-mixer

**MLP-Mixer** (*Mixer* for short) consists of per-patch linear embeddings, Mixer layers, and a classifier head. Mixer layers contain one token-mixing MLP and one channel-mixing MLP, each consisting of two fully-connected layers and a GELU nonlinearity. Other components include: skip-connections, dropout, and linear classifier head.

![alt text](figures/mlp-mixer.png)
## Installation
* `Python>=3.10` 
* Use GPUs.
```bash
pip install -r vit_jax/requirements.txt
```

## Model Training
```python
# S/16
CUDA_VISIBLE_DEVICES='0' python -m train --dataset=cifar-10 --model_name=mlp \
--learning_rate=0.001 --log_freq=100 --optimizer=sam --scheduler=sam \
--gamma=0.4
# B/16
CUDA_VISIBLE_DEVICES='1' python -m train --dataset=cifar-10 --model_name=mlp --batch_size=512 \
--embedding_dim=768 --token_mixing_dim=384 --channel_mixing_dim=3072 --depth=12 \
--learning_rate=0.001 --log_freq=98 --optimizer=sam --gamma=0.4 
# L/16
CUDA_VISIBLE_DEVICES='0' python -m train --dataset=cifar-10 --model_name=mlp --batch_size=512 \
--embedding_dim=1024 --token_mixing_dim=512 --channel_mixing_dim=4096 --depth=24 \
--learning_rate=0.001 --log_freq=98 --optimizer=sam --gamma=0.4

CUDA_VISIBLE_DEVICES='0' python -m train --dataset=cifar-10 --model_name=mlp --batch_size=512 --embedding_dim=768 --token_mixing_dim=384 --channel_mixing_dim=3072 --depth=12 --learning_rate=0.001 --log_freq=98 --optimizer=gsam --gamma=0.4 --checkpoint=mlp_b16_sam04_batch512 --save_dir=mlp_b16_gsam_batch512_from200  --num_epochs=300

CUDA_VISIBLE_DEVICES='0' python -m train --dataset=cifar-10 --model_name=mlp --batch_size=512 --embedding_dim=768 --token_mixing_dim=384 --channel_mixing_dim=3072 --depth=12 --learning_rate=3e-3 --log_freq=98 --optimizer=gsam --gamma=0.4 --checkpoint=mlp_b16_gsam_adamw --save_dir=mlp_b16_gsam_adamw --num_epochs=400 --run_name=mlp_b16_gsam_adamw

CUDA_VISIBLE_DEVICES='1' python -m train --dataset=cifar-10 --model_name=mlp --batch_size=512 --embedding_dim=768 --token_mixing_dim=384 --channel_mixing_dim=3072 --depth=12 --learning_rate=0.001 --log_freq=98 --optimizer=gsam --gamma=0.4 --checkpoint=mlp_b16_gsam --save_dir=mlp_b16_gsam --num_epochs=300 --run_name=mlp_b16_gsam
```
## Model Fine-tuning
```python
python -m main --workdir=/tmp/mix-$(date +%s) \
    --config=$(pwd)/models/configs/mixer_base16_cifar10.py \
    --config.pretrained_dir='gs://mixer_models/imagenet21k'
```

To see a detailed list of all available flags, `run python3 -m train --help`.

### Available Mixer models
We provide the Mixer-B/16 and Mixer-L/16 models pre-trained on the ImageNet and ImageNet-21k datasets. Details can be found in Table 3 of the Mixer paper. All the models can be found at:

https://console.cloud.google.com/storage/mixer_models/

Note that these models are also available directly from TF-Hub: [sayakpaul/collections/mlp-mixer](https://tfhub.dev/sayakpaul/collections/mlp-mixer) (external contribution by Sayak Paul).

### Expected Mixer results
We ran the fine-tuning code on Google Cloud machine with four V100 GPUs with the default adaption parameters from this repository. Here are the results:

|upstream|model|	dataset|	accuracy|	wall_clock_time|	link|
|--------|------|--------|----------|--------------|---------|
|ImageNet   |Mixer-B/16|cifar10|96.72%|3.0h|[tensorboard.dev](https://tensorboard.dev/experiment/j9zCYt9yQVm93nqnsDZayA/)|
|ImageNet   |Mixer-L/16|cifar10|96.59%|3.0h|[tensorboard.dev](https://tensorboard.dev/experiment/Q4feeErzRGGop5XzAvYj2g/)|
|ImageNet-21k   |Mixer-B/16|cifar10|96.82%|9.6h|[tensorboard.dev](https://tensorboard.dev/experiment/mvP4McV2SEGFeIww20ie5Q/)|
|ImageNet-21k   |Mixer-L/16|cifar10|98.34%|10.0h|[tensorboard.dev](https://tensorboard.dev/experiment/dolAJyQYTYmudytjalF6Jg/)|