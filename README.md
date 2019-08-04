# Kaggle-APTOS-2019-Blindness-Detection



###  SETUP
OLD DATA - Diabetic Retinopathy Detection (https://www.kaggle.com/c/diabetic-retinopathy-detection) <br/>
NEW DATA - APTOS 2019 Blindness Detection (https://www.kaggle.com/c/aptos2019-blindness-detection/data)

# EXP_725 (LB: 0.808)
I am gonna first pretrain model on OLD DATA and than fine tune on NEW DATA with 5-fold cross-validaion
Important transformation here is zoom crop to the center from (0.9 to 1.4)

### EXP_725.ipynb
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

### [EXP_725-CV_0 - EXP_725-CV_4].ipynb
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
```

Summary:

| Notebook Name  | Train Loss | Valid Loss | Quadratic Kappa | Weights |
| ------------- | ------------- | ---------| --------| --------|
| EXP_725-CV_0 | 0.303028 | 0.198670 | 0.924300 | NB_EXP_725_CV_0_UNFREEZE_P1| 
| EXP_725-CV_1 | 0.259711 | 0.261448 | 0.909077 | NB_EXP_725_CV_1_UNFREEZE_P1| 
| EXP_725-CV_2 | 0.318001 | 0.231378 | 0.914873 | NB_EXP_725_CV_2_UNFREEZE_P1| 
| EXP_725-CV_3 | 0.201959 | 0.182800 | 0.929271 | NB_EXP_725_CV_3_UNFREEZE_P1| 
| EXP_725-CV_4 | 0.195760 | 0.204893 | 0.937953 | NB_EXP_725_CV_4_UNFREEZE_P1| 

Submission (Average all the predictions)
```
LB SCORE:        0.808
SUBMISSION FLN:  EXP_725(version 12/14)
```


# EXP_725_352 (LB: 0.785)
Same as ``` EXP_725``` but increased image size to 352 and added more robust center zoom crop (1.1 - 1.45x). Trained using weights from ```EXP_725```, ``` NB_EXP_725_UNFREEZE_P3 ```

### EXP_725_352.ipynb
```
MODEL:           EfficientNet-B5
NUM_CLASSES:     1 (5 classes but I am treatign this as a regression problem)
BS:              56
SZ:              352
VALID:           NEW DATA

TFMS:            [flip(p=0.5), 
                 flip_vert(True), 
                 max_rotate(360), 
                 max_lighting(0.1),
                 max_zoom(1.3),
                 p_lighting(0.5), 
                 zoom_crop(scale=(1.1, 1.45), do_rand=True))]
                 
NORMALIZE:       IMAGENET
TRAINING:        fit_one_cycle(5, 1e-3,    wd=1e-2, div_factor=10, pct_start=0.3)-UNF
                 fit_one_cycle(3, 1e-3/10, wd=1e-2, div_factor=25, pct_start=0.3)-UNF

MODEL WEIGHTS:   NB_EXP_725_352_UNFREEZE_P2
MODEL TRN_LOSS:  0.289148
MODEL VAL_LOSS:  0.335972
QUADR KAPPA:     0.893238
LB SCORE:        0.727
SUBMISSION FLN:  EXP_725_352(version 16/17)
```
Comments: Pretrained model on  ``` NB_EXP_725_UNFREEZE_P3 ``` with image siae ``` 224 ```, Increaseing image size to ``` 352``` and adding extra zoom helped with the validation loss. 

### [EXP_725_352-CV_0 - EXP_725_352-CV_4].ipynb
Using weights ``` NB_EXP_725_UNFREEZE_P3 ```To train NEW DATA with 5 fold splits. <br/>

Set up for all CV experimetns: 
```
MODEL:           EfficientNet-B5
NUM_CLASSES:     1 (5 classes but I am treatign this as a regression problem)
BS:              56
SZ:              352
VALID:           NEW DATA

TFMS:            [flip(p=0.5), 
                 flip_vert(True), 
                 max_rotate(360), 
                 max_lighting(0.1),
                 max_zoom(1.3),
                 p_lighting(0.5), 
                 zoom_crop(scale=(1.1, 1.45), do_rand=True))]
                 
NORMALIZE:       IMAGENET
TRAINING:        fit_one_cycle(15, 1e-3,   wd=1e-2, div_factor=25, pct_start=0.3)-UNF
```

Summary:

| Notebook Name  | Train Loss | Valid Loss | Quadratic Kappa | Weights |
| ------------- | ------------- | ---------| --------| --------|
| EXP_725_352-CV_0| 0.214058 | 0.191476 | 0.926447 | NB_EXP_725_352_CV_0_UNFREEZE_P1| 
| EXP_725_352-CV_1| 0.220699 | 0.232317 | 0.914668 | NB_EXP_725_352_CV_1_UNFREEZE_P1| 
| EXP_725_352-CV_2| 0.216604 | 0.218829 | 0.921627 | NB_EXP_725_352_CV_2_UNFREEZE_P1| 
| EXP_725_352-CV_3| 0.222879 | 0.165061 | 0.931339 | NB_EXP_725_352_CV_3_UNFREEZE_P1| 
| EXP_725_352-CV_4| 0.218691 | 0.189928 | 0.936874 | NB_EXP_725_352_CV_4_UNFREEZE_P1| 

Submission (Average all the predictions)
```
LB SCORE:        0.785
SUBMISSION FLN:  EXP_725_352(version 15/15)
```

# EXP_730_BEN (LB: TBD)
Same as ``` EXP_725``` added more robust center zoom crop (1.02 - 1.35x). Trained using weights from ```EXP_725```, ``` NB_EXP_725_UNFREEZE_P3```. Images this time were proces. with Ben Method. Old ben method was not taking in two condiseration image ratio when resizing, I have added function ```resize_to``` which preserves image ratio when resizing. For proceessing images I used notebook ``` BEN_PROCESS.ipynb ```

### EXP_730.ipynb
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
                 zoom_crop(scale=(1.02, 1.35), do_rand=True))]
                 
NORMALIZE:       IMAGENET
TRAINING:        fit_one_cycle(10, 1e-3,    wd=1e-2, div_factor=25, pct_start=0.3)-UNF

MODEL WEIGHTS:   NB_EXP_725_352_UNFREEZE_P2
MODEL TRN_LOSS:  0.316414
MODEL VAL_LOSS:  0.302961
QUADR KAPPA:     0.896701
LB SCORE:        
SUBMISSION FLN: EXP_725_352(version 17/17)
```
Comments: Model trained using old data and weights from EXP_725, showed good training and loss. 
