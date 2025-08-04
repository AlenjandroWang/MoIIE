# MoIIE
MoIIE: Mixture of Intra- and Inter-Modality Experts for Large Vision Language Models


## Contents
- [Install](#install)
- [Train](#train)
- [Evaluation](#evaluation)

### Install
* CUDA and cuDNN

  We use CUDA 11.8 and cuDNN 8.7.0. We actually use the CUDA docker by NVIDIA: `docker pull nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04`. CUDA 12 is fine, too.

* Create a conda virtual environment and activate it:

  ```shell
  conda create -n bunny python=3.10
  conda activate bunny
  ```


  ```shell
  pip install --upgrade pip  # enable PEP 660 support
  ```

* Install apex

  ```shell
  # https://github.com/NVIDIA/apex#from-source
  pip install ninja
  git clone https://github.com/NVIDIA/apex
  cd apex
  # if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key...
  pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
  # otherwise
  pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
  ```

* Install flash-attention

  ```shell
  # https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features
  pip install packaging
  pip install flash-attn --no-build-isolation
  ```

* Install bunny and other requirements

  ```shell
  cd Bunny
  pip install -e .
  ```
### Training

MoIIE training consists of two stages: 

(1) pretrain stage: use data to connect a *frozen pretrained* vision encoder to a *frozen* LLM, and only the connector is trained ()

(2) visual instruction tuning stage&&Sparse training for all model parameters: use data to teach the model to follow multimodal instructions, where the connector, learnable LLM parameters vision encoder and MoE module are updated.

MoIIE is trained on 8 A100 GPUs. Under other circumstances, you can reduce the `per_device_train_batch_size` and increase the `gradient_accumulation_steps` accordingly. Always keep the global batch size the same: `global_batch_size ` = `per_device_train_batch_size` $`\times`$ `gradient_accumulation_steps` $`\times`$ `num_gpus`.

