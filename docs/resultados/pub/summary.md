# Resultados agregados entre sementes

Sementes 0, 1, 2 para pub_cifar10_s, pub_imagenette_s (cada semente = outro subconjunto de imagens e outro fine-tuning); pub_imagenet sem sementes (todas as imagens). Valores: média ± desvio-padrão entre sementes.

## Acurácia nas imagens originais

| dataset | model | acurácia original (%) |
|---|---|---|
| cifar10 | cifar10_mobilenetv2_x1_0 | 94.2 ± 0.4 |
| cifar10 | cifar10_repvgg_a0 | 94.4 ± 0.3 |
| cifar10 | cifar10_resnet20 | 92.3 ± 0.1 |
| cifar10 | cifar10_resnet56 | 94.4 ± 0.3 |
| cifar10 | cifar10_vgg16_bn | 94.0 ± 0.5 |
| imagenet_val | convnext_tiny | 82.1 |
| imagenet_val | efficientnet_b0 | 77.6 |
| imagenet_val | resnet18 | 69.8 |
| imagenet_val | resnet50 | 76.1 |
| imagenet_val | swin_t | 81.0 |
| imagenet_val | vgg16_bn | 73.4 |
| imagenet_val | vit_b_16 | 81.1 |
| imagenette | convnext_tiny | 86.7 ± 0.0 |
| imagenette | efficientnet_b0 | 85.0 ± 0.2 |
| imagenette | resnet18 | 77.8 ± 0.2 |
| imagenette | resnet50 | 84.4 ± 0.1 |
| imagenette | swin_t | 83.9 ± 0.6 |
| imagenette | vgg16_bn | 80.2 ± 0.5 |
| imagenette | vit_b_16 | 91.2 ± 0.2 |

## IRE × preditores existentes — R² LOTO do flip por condição (média dos modelos)

| predictor | cifar10 | imagenet_val | imagenette |
|---|---|---|---|
| distância cosseno entre features (requer a rede) | 0.99 ± 0.00 | 0.98 | 0.95 ± 0.00 |
| distância L2 entre features (requer a rede) | 0.96 ± 0.00 | 0.95 | 0.92 ± 0.00 |
| IRE | 0.86 ± 0.00 | 0.91 | 0.86 ± 0.01 |
| ΔE por banda + coerência + cor (ridge) | 0.82 ± 0.01 | 0.79 | 0.74 ± 0.04 |
| LPIPS (Zhang et al., 2018) | 0.90 ± 0.00 | 0.67 | 0.73 ± 0.01 |
| SSIM | 0.59 ± 0.00 | 0.47 | 0.54 ± 0.01 |
| energia por banda (Yin et al., 2019) | 0.49 ± 0.02 | 0.42 | 0.61 ± 0.01 |
| parâmetro nominal | 0.23 ± 0.00 | 0.47 | 0.42 ± 0.00 |
| PSNR | 0.08 ± 0.00 | 0.12 | 0.14 ± 0.00 |

## R² LOTO da queda de acurácia por conjunto de preditores

| features | cifar10 | imagenet_val | imagenette |
|---|---|---|---|
| delta_bands | 0.51 ± 0.01 | 0.35 | 0.44 ± 0.02 |
| dose_rank | 0.21 ± 0.00 | 0.36 | 0.28 ± 0.01 |
| psnr | 0.06 ± 0.00 | 0.05 | 0.05 ± 0.00 |
| spec_bands | 0.68 ± 0.00 | 0.49 | 0.39 ± 0.04 |
| spec_bands+coh | 0.77 ± 0.01 | 0.78 | 0.71 ± 0.03 |
| spec_bands+coh+chroma | 0.80 ± 0.01 | 0.80 | 0.80 ± 0.03 |
| spec_bands_dB | 0.60 ± 0.00 | 0.37 | 0.37 ± 0.03 |
| spec_fine | 0.62 ± 0.02 | -0.89 | -27.03 ± 3.76 |
| spec_fine+coh | 0.57 ± 0.25 | -0.89 | -1.03 ± 0.97 |
| ssim | 0.58 ± 0.00 | 0.38 | 0.38 ± 0.01 |

## Limiar BMD (queda de 1 p.p.)

n < número de sementes: em alguma semente o limiar não cruzou dentro das doses testadas.

| dataset | technique | cifar10_mobilenetv2_x1_0 | cifar10_repvgg_a0 | cifar10_resnet20 | cifar10_resnet56 | cifar10_vgg16_bn | convnext_tiny | efficientnet_b0 | resnet18 | resnet50 | swin_t | vgg16_bn | vit_b_16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cifar10 | bilateral | 1.145 ± 0.174 | 1.241 ± 0.118 | 1.015 ± 0.128 | 1.052 ± 0.045 | 1.174 ± 0.108 | nan | nan | nan | nan | nan | nan | nan |
| cifar10 | clahe | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.029 ± 0.050 | 1.000 ± 0.000 | 1.000 ± 0.000 | nan | nan | nan | nan | nan | nan | nan |
| cifar10 | clahe_blend | 0.522 ± 0.197 | 0.791 ± 0.050 | 0.700 ± 0.163 | 0.738 ± 0.210 | 0.736 ± 0.126 | nan | nan | nan | nan | nan | nan | nan |
| cifar10 | gaussian | 0.462 ± 0.023 | 0.500 ± 0.005 | 0.473 ± 0.023 | 0.485 ± 0.017 | 0.458 ± 0.021 | nan | nan | nan | nan | nan | nan | nan |
| cifar10 | hist_eq | 0.262 ± 0.024 | 0.379 ± 0.027 | 0.259 ± 0.069 | 0.272 ± 0.044 | 0.280 ± 0.019 | nan | nan | nan | nan | nan | nan | nan |
| cifar10 | jpeg | 0.300 ± 0.265 | 1.867 ± 1.650 | 2.476 ± 0.478 | 2.479 ± 0.182 | 0.667 ± 1.155 | nan | nan | nan | nan | nan | nan | nan |
| cifar10 | jpeg444 | 4.886 ± 0.789 | 5.785 ± 0.478 | 5.651 ± 0.346 | 4.660 ± 0.301 | 6.808 ± 0.617 | nan | nan | nan | nan | nan | nan | nan |
| cifar10 | laplacian | 0.107 ± 0.009 | 0.108 ± 0.012 | 0.128 ± 0.019 | 0.125 ± 0.042 | 0.140 ± 0.021 | nan | nan | nan | nan | nan | nan | nan |
| cifar10 | median | 1.151 ± 0.009 | 1.185 ± 0.005 | 1.118 ± 0.009 | 1.152 ± 0.006 | 1.180 ± 0.011 | nan | nan | nan | nan | nan | nan | nan |
| cifar10 | sr_bicubic | 1.013 ± 0.001 | 1.020 ± 0.002 | 1.011 ± 0.001 | 1.015 ± 0.001 | 1.019 ± 0.003 | nan | nan | nan | nan | nan | nan | nan |
| cifar10 | sr_espcn | 2.000 ± 0.000 | 2.000 ± 0.000 | 2.000 ± 0.000 | 2.000 ± 0.000 | 2.000 ± 0.000 | nan | nan | nan | nan | nan | nan | nan |
| cifar10 | sr_fsrcnn | 2.000 ± 0.000 | 2.000 ± 0.000 | 2.000 ± 0.000 | 2.000 ± 0.000 | 2.000 ± 0.000 | nan | nan | nan | nan | nan | nan | nan |
| cifar10 | unsharp | 0.591 ± 0.029 | 0.492 ± 0.022 | 0.651 ± 0.050 | 0.505 ± 0.139 | 0.732 ± 0.057 | nan | nan | nan | nan | nan | nan | nan |
| cifar10 | wiener | 1.136 ± 0.003 | 1.191 ± 0.016 | 1.126 ± 0.010 | 1.166 ± 0.010 | 1.178 ± 0.007 | nan | nan | nan | nan | nan | nan | nan |
| imagenet_val | bilateral | nan | nan | nan | nan | nan | 0.903 (n=1) | 0.869 (n=1) | 1.051 (n=1) | 1.034 (n=1) | 0.732 (n=1) | 0.757 (n=1) | 1.491 (n=1) |
| imagenet_val | clahe | nan | nan | nan | nan | nan | 1.903 (n=1) | 1.178 (n=1) | 1.000 (n=1) | 1.000 (n=1) | 1.919 (n=1) | 1.218 (n=1) | 1.489 (n=1) |
| imagenet_val | clahe_blend | nan | nan | nan | nan | nan | 0.956 (n=1) | 0.624 (n=1) | 0.457 (n=1) | 0.546 (n=1) | 0.968 (n=1) | 0.605 (n=1) | 0.792 (n=1) |
| imagenet_val | gaussian | nan | nan | nan | nan | nan | 0.639 (n=1) | 0.536 (n=1) | 0.527 (n=1) | 0.509 (n=1) | 0.535 (n=1) | 0.547 (n=1) | 0.598 (n=1) |
| imagenet_val | hist_eq | nan | nan | nan | nan | nan | 0.658 (n=1) | 0.488 (n=1) | 0.340 (n=1) | 0.371 (n=1) | 0.645 (n=1) | 0.358 (n=1) | 0.646 (n=1) |
| imagenet_val | jpeg | nan | nan | nan | nan | nan | 4.832 (n=1) | 7.605 (n=1) | 11.381 (n=1) | 8.107 (n=1) | 3.218 (n=1) | 6.928 (n=1) | 12.440 (n=1) |
| imagenet_val | jpeg444 | nan | nan | nan | nan | nan | 8.861 (n=1) | 13.155 (n=1) | 21.270 (n=1) | 18.395 (n=1) | 7.731 (n=1) | 10.409 (n=1) | 33.481 (n=1) |
| imagenet_val | laplacian | nan | nan | nan | nan | nan | 0.287 (n=1) | 0.657 (n=1) | 0.174 (n=1) | 0.156 (n=1) | 0.600 (n=1) | 0.177 (n=1) | 0.337 (n=1) |
| imagenet_val | median | nan | nan | nan | nan | nan | 1.531 (n=1) | 1.491 (n=1) | 1.480 (n=1) | 1.482 (n=1) | 1.413 (n=1) | 1.339 (n=1) | 2.050 (n=1) |
| imagenet_val | sr_bicubic | nan | nan | nan | nan | nan | 1.132 (n=1) | 1.053 (n=1) | 1.131 (n=1) | 1.045 (n=1) | 1.110 (n=1) | 1.096 (n=1) | 1.191 (n=1) |
| imagenet_val | sr_espcn | nan | nan | nan | nan | nan | 2.000 (n=1) | 2.000 (n=1) | 2.000 (n=1) | 2.000 (n=1) | 2.000 (n=1) | 2.000 (n=1) | 2.000 (n=1) |
| imagenet_val | sr_fsrcnn | nan | nan | nan | nan | nan | 2.000 (n=1) | 2.000 (n=1) | 2.000 (n=1) | 2.000 (n=1) | 2.000 (n=1) | 2.000 (n=1) | 2.000 (n=1) |
| imagenet_val | unsharp | nan | nan | nan | nan | nan | 1.183 (n=1) | 1.757 (n=1) | 0.682 (n=1) | 0.699 (n=1) | 2.095 (n=1) | 0.722 (n=1) | 1.094 (n=1) |
| imagenet_val | wiener | nan | nan | nan | nan | nan | 1.644 (n=1) | 1.602 (n=1) | 1.541 (n=1) | 1.548 (n=1) | 1.541 (n=1) | 1.490 (n=1) | 2.272 (n=1) |
| imagenette | bilateral | nan | nan | nan | nan | nan | 2.123 ± 0.223 | 1.040 ± 0.106 | 1.055 ± 0.052 | 0.834 ± 0.063 | 1.321 ± 0.404 | 1.034 ± 0.018 | 1.608 ± 0.668 |
| imagenette | clahe | nan | nan | nan | nan | nan | 2.965 ± 0.439 | 1.236 ± 0.168 | 1.000 ± 0.000 | 1.000 ± 0.000 | 2.177 ± 0.472 | 1.116 ± 0.101 | 1.813 ± 0.039 |
| imagenette | clahe_blend | nan | nan | nan | nan | nan | — (n=0) | 0.694 ± 0.138 | 0.421 ± 0.057 | 0.566 ± 0.111 | 0.949 ± 0.025 (n=2) | 0.528 ± 0.044 | 0.828 ± 0.075 |
| imagenette | gaussian | nan | nan | nan | nan | nan | 0.694 ± 0.029 | 0.479 ± 0.012 | 0.417 ± 0.009 | 0.383 ± 0.014 | 0.679 ± 0.036 | 0.417 ± 0.006 | 0.465 ± 0.034 |
| imagenette | hist_eq | nan | nan | nan | nan | nan | 0.895 ± 0.004 | 0.427 ± 0.029 | 0.300 ± 0.040 | 0.332 ± 0.020 | 0.804 ± 0.090 | 0.353 ± 0.042 | 0.706 ± 0.042 |
| imagenette | jpeg | nan | nan | nan | nan | nan | 15.024 ± 2.495 | 9.278 ± 0.855 | 0.000 ± 0.000 | 0.000 ± 0.000 | 9.440 ± 4.597 | 5.397 ± 2.425 | 7.306 ± 2.379 |
| imagenette | jpeg444 | nan | nan | nan | nan | nan | 14.940 ± 2.115 | 13.692 ± 0.854 | 19.118 ± 2.886 | 24.470 ± 2.149 | 21.592 ± 9.055 | 15.511 ± 1.510 | 31.779 ± 4.870 |
| imagenette | laplacian | nan | nan | nan | nan | nan | — (n=0) | 1.454 ± 0.019 (n=2) | 0.498 ± 0.045 | 0.702 ± 0.026 | 1.140 ± 0.078 | 0.696 ± 0.043 | 0.758 ± 0.099 |
| imagenette | median | nan | nan | nan | nan | nan | 2.611 ± 0.360 | 1.370 ± 0.061 | 1.369 ± 0.027 | 1.328 ± 0.037 | 1.608 ± 0.153 | 1.287 ± 0.032 | 1.598 ± 0.062 |
| imagenette | sr_bicubic | nan | nan | nan | nan | nan | 1.125 ± 0.015 | 1.126 ± 0.044 | 1.023 ± 0.002 | 1.031 ± 0.002 | 1.307 ± 0.113 | 1.033 ± 0.004 | 1.034 ± 0.006 |
| imagenette | sr_espcn | nan | nan | nan | nan | nan | 2.000 ± 0.000 | 2.000 ± 0.000 | 2.000 ± 0.000 | 2.000 ± 0.000 | 2.000 ± 0.000 | 2.000 ± 0.000 | 2.000 ± 0.000 |
| imagenette | sr_fsrcnn | nan | nan | nan | nan | nan | 2.000 ± 0.000 | 2.000 ± 0.000 | 2.000 ± 0.000 | 2.000 ± 0.000 | 2.000 ± 0.000 | 2.000 ± 0.000 | 2.000 ± 0.000 |
| imagenette | unsharp | nan | nan | nan | nan | nan | — (n=0) | 2.243 ± 0.060 | 1.536 ± 0.166 | 2.723 ± 0.134 | 2.385 ± 0.266 | 2.192 ± 0.225 | 1.885 ± 0.233 |
| imagenette | wiener | nan | nan | nan | nan | nan | 3.450 ± 0.112 | 1.627 ± 0.125 | 1.486 ± 0.015 | 1.470 ± 0.057 | 2.352 ± 0.489 | 1.416 ± 0.063 | 1.799 ± 0.113 |

## E4 — adaptação (%, média dos modelos ± sd entre sementes)

| dataset | cond_id | acc_frozen | acc_bn | acc_ft_pp | info_loss |
|---|---|---|---|---|---|
| cifar10 | bilateral@3 | 90.6 ± 0.6 | 89.9 ± 0.8 | 92.0 ± 0.3 | 2.8 ± 0.3 |
| cifar10 | clahe@1 | 92.0 ± 0.6 | 89.8 ± 0.8 | 92.3 ± 0.6 | 2.1 ± 0.4 |
| cifar10 | gaussian@5 | 39.3 ± 0.6 | 85.1 ± 0.9 | 89.5 ± 0.9 | 30.4 ± 0.6 |
| cifar10 | jpeg@7 | 82.8 ± 0.9 | 80.3 ± 1.3 | 87.0 ± 0.8 | 9.8 ± 0.4 |
| cifar10 | jpeg@11 | 68.5 ± 1.3 | 67.8 ± 1.3 | 80.9 ± 0.8 | 21.2 ± 1.2 |
| cifar10 | laplacian@4 | 85.4 ± 0.6 | 89.5 ± 0.7 | 92.7 ± 0.4 | 6.1 ± 0.5 |
| cifar10 | median@0 | 80.8 ± 0.5 | 83.4 ± 0.6 | 87.4 ± 0.4 | 9.7 ± 0.6 |
| cifar10 | unsharp@5 | 91.6 ± 0.5 | 90.3 ± 0.6 | 93.0 ± 0.3 | 1.9 ± 0.3 |
| cifar10 | wiener@1 | 73.5 ± 0.5 | 83.1 ± 0.8 | 87.2 ± 0.4 | 14.9 ± 1.0 |
| imagenette | bilateral@5 | 98.3 ± 0.0 | 98.5 ± 0.1 | 98.9 ± 0.1 | 0.8 ± 0.1 |
| imagenette | clahe@5 | 98.8 ± 0.1 | 98.2 ± 0.1 | 98.9 ± 0.1 | 0.7 ± 0.1 |
| imagenette | gaussian@5 | 99.1 ± 0.1 | 98.8 ± 0.1 | 99.0 ± 0.1 | 0.5 ± 0.0 |
| imagenette | gaussian@9 | 89.2 ± 0.2 | 89.4 ± 0.1 | 95.8 ± 0.1 | 4.7 ± 0.4 |
| imagenette | hist_eq@8 | 99.0 ± 0.1 | 98.0 ± 0.0 | 98.9 ± 0.1 | 0.5 ± 0.1 |
| imagenette | jpeg@13 | 95.7 ± 0.1 | 93.3 ± 0.1 | 97.5 ± 0.0 | 3.0 ± 0.3 |
| imagenette | laplacian@7 | 98.1 ± 0.1 | 98.8 ± 0.0 | 99.2 ± 0.0 | 1.1 ± 0.1 |
| imagenette | median@2 | 94.6 ± 0.0 | 96.6 ± 0.2 | 97.9 ± 0.1 | 2.6 ± 0.2 |
| imagenette | median@3 | 90.4 ± 0.0 | 94.1 ± 0.2 | 96.7 ± 0.3 | 4.4 ± 0.1 |
| imagenette | sr_bicubic@9 | 95.1 ± 0.1 | 94.0 ± 0.2 | 97.2 ± 0.4 | 2.3 ± 0.1 |
| imagenette | sr_espcn@2 | 95.4 ± 0.2 | 93.1 ± 0.1 | 97.0 ± 0.1 | 2.7 ± 0.3 |
| imagenette | unsharp@8 | 99.2 ± 0.0 | 98.9 ± 0.1 | 99.3 ± 0.1 | 0.3 ± 0.1 |
| imagenette | wiener@3 | 97.4 ± 0.1 | 97.3 ± 0.1 | 98.3 ± 0.2 | 1.6 ± 0.2 |
