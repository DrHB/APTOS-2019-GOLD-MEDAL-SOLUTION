# Kaggle-APTOS-2019-Blindness-Detection



##  SETUP
OLD DATA - Diabetic Retinopathy Detection (https://www.kaggle.com/c/diabetic-retinopathy-detection) <br/>
NEW DATA - APTOS 2019 Blindness Detection https://www.kaggle.com/c/aptos2019-blindness-detection/data

# EXP_725
I am gonna first pretrain model on OLD DATA and than fine tune on NEW DATA with 5-fold cross-validaion
Important transformation here is zoom crop to the center from (0.9 to 1.4)

## EXP_725.ipynb
```
MODEL:           EfficientNet-B5
NUM_CLASSES:     1 (5 classes but I am treatign this as a regression problem)
BS:              128
SZ:              224
VALID:           NEW DATA
TFMS:            [flip(p=0.5), 
                  flip_vert(True), 
                  max_rotate(360), 
                  max_lighting(0.1),
                  max_zoom(1.3),
                  p_lighting(0.5), 
                  zoom_crop(scale=(0.9, 1.4), do_rand=True))]
NORMALIZE:       IMAGENET
TRAINING:        fit_one_cycle(10, 1e-3,   wd=1e-2, div_factor=10, pct_start=0.3)-UNF
                 fit_one_cycle(5,  1e-3/5, wd=1e-2, div_factor=10, pct_start=0.3)-UNF
                 fit_one_cycle(30, 1e-3/8, wd=1e-2, div_factor=10, pct_start=0.3)-UNF
MODEL WEIGHTS:   NB_EXP_725_UNFREEZE_P3
MODEL TRN_LOSS:  0.305515
MODEL VAL_LOSS:  0.342098
QUADR KAPPA:     0.887489
LB SCORE:        0.725
SUBMISSION FLN:  EXP_725(version 11/14)
```
Comments: Pretrained model trained just OLD DATA gives pretty good results. Now using best weight to fine tune new data

## [EXP_725-CV_0 - EXP_725-CV_5] .ipynb
Using weights ``` NB_EXP_725_UNFREEZE_P3 ```To train NEW DATA with 5 fold splits. <br/>

Set up for all CV experimetns: 
```
MODEL:           EfficientNet-B5
NUM_CLASSES:     1 (5 classes but I am treatign this as a regression problem)
BS:              128
SZ:              224
VALID:           NEW DATA
TFMS:            [flip(p=0.5), 
                  flip_vert(True), 
                  max_rotate(360), 
                  max_lighting(0.1),
                  max_zoom(1.3),
                  p_lighting(0.5), 
                  zoom_crop(scale=(0.9, 1.4), do_rand=True))]
NORMALIZE:       IMAGENET
TRAINING:        fit_one_cycle(15, 1e-3,   wd=1e-2, div_factor=10, pct_start=0.3)-UNF

