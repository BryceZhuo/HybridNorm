# HybridNorm: Towards Stable and Efficient Transformer Training via Hybrid Normalization

This repository contains the official implementation of paper **[“HybridNorm: Towards Stable and Efficient Transformer Training via Hybrid Normalization]()”.**


![Framework](./assets/norm.png)
Illustrations of different transformer layer structures: (a) Post-Norm architecture; (b) Pre-Norm architecture; (c) Pre-Norm with QK-Norm architecture; (d) HybridNorm architecture.


## Introduction
In this paper, we propose **HybridNorm**, a straightforward yet effective hybrid normalization strategy that integrates the advantages of both Pre-Norm and Post-Norm approaches. Specifically, HybridNorm employs QKV normalization within the attention mechanism and Post-Norm in the feed-forward network (FFN) of each transformer block. This design not only stabilizes training but also enhances performance.


- **1B Dense Models**: Training and validation performance of 1.2 B dense models with Hybridnorm and HybridNorm<sup>*</sup> under 1T training tokens.
![Dynamics of 1B dense model](./assets/OLMo2_1B_train_dynamics.png)

- **MoE-1B-7B**: Traning and validation performace for MoE models under 500B training tokens.
![Dynamics of MoE model](./assets/OLMoE_1B7B_train_dynamics.svg)



## Pretraining

### Preparation
First, you need to install:

```python
# For dense model
cd dense_model
pip install -e .[all]

# For MoE model
cd MoE_model
pip install -e .[all]
```

Then you need to download the datasets [**OLMo-mix-1124**](https://huggingface.co/datasets/allenai/olmo-mix-1124) and [**OLMoE-mix-0924**](https://huggingface.co/datasets/allenai/OLMoE-mix-0924). Please see OLMo' instructions [README_of_OLMo2.md](./dense_model/README_of_OLMo2.md) and [README_of_OLMoE.md](./MoE_model/README_of_OLMoE.md) for details.



### Training
##### Dense model

```python
cd dense_model

# For HybridNorm
torchrun --nproc_per_node=8 scripts/train.py configs/exps/1.2B-dense-HybriedNorm.yaml 

# For HybridNorm*
torchrun --nproc_per_node=8 scripts/train.py configs/exps/1.2B-dense-HybriedNorm.yaml \
  ----model.norm_pos_special_config="_{0:{'attn_norm':True,'ffn_residual':False}}_" 

```
##### MoE model
```python
cd MoE_model

# For HybridNorm*
torchrun --nproc_per_node=8 scripts/train.py configs/exps/MoE-1B-7B-HybridNorm.yaml \
  ----model.norm_pos_special_config="_{0:{'attn_norm':True,'ffn_residual':False}}_" 
```


## Citing this work
If you find this work helpful or use it in your research, please consider citing our paper:
```bibtex
@article{zhuo2025hybridnorm,
  title={HybridNorm: Towards Stable and Efficient Transformer Training via Hybrid Normalization},
  author={Zhijian Zhuo and Yutao Zeng and Ya Wang and Sijun Zhang and Jian Yang and Xiaoqing Li and Xun Zhou and Jinwen Ma},
  journal={arXiv preprint arXiv:2503.04598},
  year={2025}
}
```

## Acknowledgement
The code is based on [OLMo2](https://github.com/allenai/OLMo) and [OLMoE](https://github.com/allenai/OLMoE)

