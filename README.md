# Kaggle-APTOS-2019-Blindness-Detection



##  SETUP
OLD DATA - Diabetic Retinopathy Detection (https://www.kaggle.com/c/diabetic-retinopathy-detection)
NEW DATA - APTOS 2019 Blindness Detection https://www.kaggle.com/c/aptos2019-blindness-detection/data

# EXP 725

## EXP_0.ipynb
```
MODEL:           EfficientNet-B5
NUM_CLASSES:     1108
BS:              64
SZ:              224
VALID:           Random split(0.2)
TFMS:            [flip(p=0.5), max_rotate(-10, 10), max_zoom(1.1), max_lighting(0.2), max_wrap(0.2), p_lighting(0.75)]
NORMALIZE:       IMAGENET

TRAINING:        fit_one_cycle(3,  1e-3,   wd=1e-2, div_factor=10, pct_start=0.3)-FRZ
                 fit_one_cycle(15, 1e-3/2, wd=1e-2, div_factor=10, pct_start=0.3)-UNF
                 fit_one_cycle(15, 1e-4, wd=1e-2, div_factor=10, pct_start=0.3)  -UNF

MODEL WEIGHTS:   EXP_00_PHASE_3_UNFREEZE
MODEL TRN_LOSS:  1.510587
MODEL VAL_LOSS:  3.794280
MODEL ACCURACY:  29.837053
LB SCORE:        0.137
SUBMISSION FLN:  RGB_EXP_0.csv
```
Comments: huge gap between trainign and valid, try to StratifiedKFold split

## EXP_0_1.ipynb
