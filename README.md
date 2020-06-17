# Reinforcement learning for Recommendation Systems

Implementation of [Deep Reinforcement Learning based Recommendation with Explicit User-Item Interactions Modeling](https://arxiv.org/pdf/1810.12027.pdf)

TODO:

- [x] Implement validation 
- [x] Change training process (switch to sessions)
- [x] Add Prioritized Experience Replay

Special TODO:

- [x] Add Ornstein–Uhlenbeck noise for better exploration



#### Movielens (1M) results

| Model                          | nDCG@10 | Hit_rate@10 |
| ------------------------------ | :-----: | ----------- |
| **DDPG with OU noise**         |  0.281  | 0.506       |
| DDPG                           |  0.243  | 0.423       |
| Neural Collaborative Filtering |  0.238  | 0.430       |
| Random (for comparison)        |  ~0.05  | ~0.1        |



![viz](img/learning_curve.png)

