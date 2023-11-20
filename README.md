# Robust Data Pruning under Label Noise via Maximizing Re-labeling Accuracy (NeurIPS 2023, [PDF](https://arxiv.org/pdf/2311.01002.pdf))

by [Dongmin Park](https://scholar.google.com/citations?user=4xXYQl0AAAAJ&hl=ko)<sup>1</sup>, [Seola Choi](https://scholar.google.com/citations?hl=ko&user=U6P2mAgAAAAJ)<sup>1</sup>, [Doyoung Kim](https://scholar.google.com/citations?user=vEAbNDYAAAAJ&hl=en)<sup>1</sup>, [Hwanjun Song](https://scholar.google.com/citations?user=Ijzuc-8AAAAJ&hl=en)<sup>1, 2</sup>, [Jae-Gil Lee](https://scholar.google.com/citations?user=h9mbv9MAAAAJ&hl=en)<sup>1</sup>.

<sup>1</sup> KAIST, <sup>2</sup> Amazon AWS AI

* **`Sep 22, 2023`:** **Our work is accepted at NeurIPS 2023.**

# How to run

### Prune4ReL

Go to the AL/ folder

* CIFAR10
```bash
python3 main.py --gpu 0 --data_path=$your_data_folder --dataset 'CIFAR10' --n-class 10 --model 'ResNet18' \
                        --method 'Uncertainty' --uncertainty 'Margin' --n-query 1000 --epochs 200 --batch-size 128
```
* CIFAR100
```bash
python3 main.py --gpu 0 --data_path=$your_data_folder --dataset 'CIFAR100' --n-class 100 --model 'ResNet18' \
                        --method 'Uncertainty' --uncertainty 'Margin' --n-query 1000 --epochs 200 --batch-size 128
```
* ImageNet30
```bash
python3 main.py --gpu 0 --data_path=$your_data_folder --dataset 'ImageNet30' --n-class 30 --model 'ResNet18' \
                        --method 'Uncertainty' --uncertainty 'Margin' --n-query 780 --epochs 200 --batch-size 128
```


### Existing Subset Selection Algorithms: Forgetting, GraNd, kCenterGreedy, Glister, etc

Go to the DeepCore/ folder

* CIFAR10, CIFAR100, ImageNet50
```bash
python3 main.py --data_path=$your_data_folder --datset $dataset --model $arch --selection $selection_algorithm --fraction $target_fraction
```
\*$selection_algorithm must be in ['Uniform', 'Uncertainty', 'Forgetting', 'GraNd', ...], each of which is a class name in deep_core/methods/~~.py


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

* DeepCore library \[[code](https://github.com/PatrickZH/DeepCore)\] : DeepCore: A Comprehensive Library for Coreset Selection in Deep Learning, Guo et al. 2022.
