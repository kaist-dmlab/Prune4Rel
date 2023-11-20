# CIFAR10
for data in CIFAR10
do
  for tau in 0.975
  do
    for fraction in 0.2 0.4 0.6 0.8
    do
      # Uniform Uncertainty kCenterGreedy Forgetting GraNd Glister Moderate / SmallLoss / Prune4Rel
      for method in Prune4Rel
      do
        for noise_type in real1, real2, real3, realW
        do
          python3 ../main_label_noise.py --gpu 0 --model 'PreActResNet18' --noise-type $noise_type \
          -se 10 --epochs 300 --fraction $fraction --selection $method --save-log True --pre-warmuped False \
          --metric cossim --uncertainty LeastConfidence \
          --tau $tau --balance True
        done
      done
    done
  done
done