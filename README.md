# Robust Data Pruning under Label Noise via Maximizing Re-labeling Accuracy (NeurIPS 2023, [PDF](https://arxiv.org/pdf/2311.01002.pdf))

by [Dongmin Park](https://scholar.google.com/citations?user=4xXYQl0AAAAJ&hl=ko)<sup>1</sup>, [Seola Choi](https://scholar.google.com/citations?hl=ko&user=U6P2mAgAAAAJ)<sup>1</sup>, [Doyoung Kim](https://scholar.google.com/citations?user=vEAbNDYAAAAJ&hl=en)<sup>1</sup>, [Hwanjun Song](https://scholar.google.com/citations?user=Ijzuc-8AAAAJ&hl=en)<sup>1, 2</sup>, [Jae-Gil Lee](https://scholar.google.com/citations?user=h9mbv9MAAAAJ&hl=en)<sup>1</sup>.

<sup>1</sup> KAIST, <sup>2</sup> Amazon AWS AI

* **`Sep 22, 2023`:** **Our work is accepted at NeurIPS 2023.**

# Brief Summary
- **Prune4ReL** is a new data pruning method for **Re-labeling** models (e.g., [DivideMix](https://github.com/LiJunnan1992/DivideMix) & [SOP+](https://github.com/shengliu66/SOP)) showing state-of-the-art performance under label noise.
- Inspired by a re-labeling theory, Prune4ReL finds the desired data subset by **maximizing the total reduced neighborhood confidence**, thereby maximizing re-labeling & generalization performance.
- With a greedy approximation, Prune4ReL is efficient and **scalable to large datasets** including Clothing-1M & ImageNet-1K.
- On four real noisy datasets (e.g., CIFAR-10/100N, WebVision, & Clothing-1M), **Prune4Rel outperforms data pruning baselines with Re-labeling models by 9.1%, and those with a standard model by 21.6%**.

# How to run

### Prune4ReL

Please follow Table 7 for hyperparameters. For CIFAR-10N dataset with SOP+ as Re-labeling model,

```bash
python3 main_label_noise.py --gpu 0 --model 'PreActResNet18' --robust-learner 'SOP' -rc 0.9 -rb 0.1 \
          --dataset CIFAR10 --noise-type $noise_type --n-class 10 --lr-u 10 -se 10 --epochs 300 \
          --fraction $fraction --selection Prune4Rel --save-log True \
          --metric cossim --uncertainty LeastConfidence --tau 0.975 --eta 1 --balance True
```

More detailed scripts for other datasets can be found in [`scripts/`](https://github.com/kaist-dmlab/Prune4Rel/tree/main/scripts) folder.



### Data Pruning Baselines: Uniform, SmallLoss, Margin, Forgetting, GraNd, Moderate, etc

Basically, the script is similar to that of Prune4ReL. For example, 

```bash
python3 main_label_noise.py --gpu 0 --model 'PreActResNet18' --robust-learner 'SOP' -rc 0.9 -rb 0.1 \
          --dataset CIFAR10 --noise-type $noise_type --n-class 10 --lr-u 10 -se 10 --epochs 300 \
          --fraction $fraction --selection *$pruning_algorithm* --save-log True \
```
where \*$pruning_algorithm\* *must* be from [Uniform, SmallLoss, Uncertainty, Forgetting, GraNd, ...], each of which is a class name in `deep_core/methods/~~.py`.


# Citation

```
@article{park2023robust,
  title={Robust Data Pruning under Label Noise via Maximizing Re-labeling Accuracy},
  author={Park, Dongmin and Choi, Seola and Kim, Doyoung and Song, Hwanjun and Lee, Jae-Gil},
  journal={NeurIPS 2023},
  year={2023}
}
```

# References

We thank the DeepCore library, on which we built most of our repo. Hope our project helps extend the open-source library of data pruning.
* DeepCore library \[[code](https://github.com/PatrickZH/DeepCore)\] : DeepCore: A Comprehensive Library for Coreset Selection in Deep Learning, Guo et al. 2022.
 
 
 
 
 
