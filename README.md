# Graph Prediction with Distribution Shift of Graph Feature

Prediction on graphs has a wide range of applications, such as traffic flow forecasting and atmospheric pollution prediction. However, a common issue with these graph data is that the feature distributions on the graphs change over time. This work considers the problem of making predictions when the feature distribution of a graph changes. Causal inference attributes the distribution changes to variations in environmental variables. When a new environment comes, the performance of the model may degenerate.  Based on graph neural networks, a framework that separates the changing environmental information from the invariant information is proposed. In the feature space, the framework represents the environment as a convex combination of a set of fixed bases, aiming to transform unseen environments close to seen environments as much as possible. 

## Installation

The code is implemented by Python, Pytorch, and PyTorch Geometric. Using the following code to install the required packages:

```bash
pip install -r requirements.txt
```

# Run
To run the code, please use the following commands
```bash
python main.py \
        --dataset AIR_BJ --mode 'train' \
        --batch_size 64 --save_iter 100 --base_lr 1e-4 \
        --input_dim 1 \
        --hid_dim 32 \
        --dropout 0.1 \
        --wo_env False \
        --wo_env_aug False \
        --wo_s_edge False \
        --edge_feat_flag False \
        --depth 10 \
        --n_envs 10 \
        --aug_magnitude 0.2 \
        --K 2 \
        --seed 2020 \
        --beta1 0.6 \
        --beta2 1 \
        --n_exp 0
```