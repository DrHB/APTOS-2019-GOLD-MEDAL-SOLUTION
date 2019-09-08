# Kaggle-APTOS-2019-Blindness-Detection
About Challange: 

Imagine being able to detect blindness before it happened.

Millions of people suffer from diabetic retinopathy, the leading cause of blindness among working aged adults. Aravind Eye Hospital in India hopes to detect and prevent this disease among people living in rural areas where medical screening is difficult to conduct. Successful entries in this competition will improve the hospital’s ability to identify potential patients. Further, the solutions will be spread to other Ophthalmologists through the 4th Asia Pacific Tele-Ophthalmology Society (APTOS) Symposium

Currently, Aravind technicians travel to these rural areas to capture images and then rely on highly trained doctors to review the images and provide diagnosis. Their goal is to scale their efforts through technology; to gain the ability to automatically screen images for disease and provide information on how severe the condition may be.

In this synchronous Kernels-only competition, you'll build a machine learning model to speed up disease detection. You’ll work with thousands of images collected in rural areas to help identify diabetic retinopathy automatically. If successful, you will not only help to prevent lifelong blindness, but these models may be used to detect other sorts of diseases in the future, like glaucoma and macular degeneration.



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
VALID:           NEW DATA CV SPLIT

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
VALID:           NEW DATA CV SPLIT

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

# EXP_730_BEN (LB: 0.804)
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
LB SCORE:        0.755
SUBMISSION FLN:  EXP_725_352(version 17/17)
```
Comments: Model trained using old data and weights from EXP_725, showed good training and loss. 

### [EXP_730-CV_0 - EXP_730-CV_4].ipynb
Using weights ``` NB_EXP_730_UNFREEZE_P1 ```To train NEW DATA with 5 fold splits. <br/>

Set up for all CV experimetns: 
```
MODEL:           EfficientNet-B5
NUM_CLASSES:     1 (5 classes but I am treatign this as a regression problem)
BS:              56
SZ:              352
VALID:           NEW DATA CV SPLIT

TFMS:            [flip(p=0.5), 
                 flip_vert(True), 
                 max_rotate(360), 
                 max_lighting(0.1),
                 max_zoom(1.3),
                 p_lighting(0.5), 
                  zoom_crop(scale=(1.01, 1.35), do_rand=True))]
                 
NORMALIZE:       IMAGENET
TRAINING:        fit_one_cycle(15, 1e-3,   wd=1e-2, div_factor=25, pct_start=0.3)-UNF
```

Summary:

| Notebook Name  | Train Loss | Valid Loss | Quadratic Kappa | Weights |
| ------------- | ------------- | ---------| --------| --------|
| EXP_730-CV_0| 0.230880 | 0.214356 | 0.920836 | NB_EXP_730_CV_0_UNFREEZE_P1| 
| EXP_730-CV_1| 0.298317 | 0.251441 | 0.918218 | NB_EXP_730_CV_1_UNFREEZE_P1| 
| EXP_730-CV_2| 0.216604 | 0.299689 | 0.904481 | NB_EXP_730_CV_2_UNFREEZE_P1| 
| EXP_730-CV_3| 0.178483 | 0.176829 | 0.932204 | NB_EXP_730_CV_3_UNFREEZE_P1| 
| EXP_730-CV_4| 0.170133 | 0.235239 | 0.928617 | NB_EXP_730_CV_4_UNFREEZE_P1| 

Submission (Average all the predictions)
```
LB SCORE:        0.804
SUBMISSION FLN:  EXP_730_BEN(version 22/22)
```
# EXP_730_352_BEN (LB: TBD)
Same as ``` EXP_730_BEN``` added more robust center zoom crop (1.02 - 1.35x) and the image size increased to ```352```. Trained using weights from ```EXP_730_BEN```, ``` NB_EXP_730_UNFREEZE_P1```. 

### EXP_730_352.ipynb
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
                 zoom_crop(scale=(1.02, 1.35), do_rand=True))]
                 
NORMALIZE:       IMAGENET
TRAINING:        fit_one_cycle(5, 1e-3/2,    wd=1e-2, div_factor=25, pct_start=0.3)-UNF

MODEL WEIGHTS:   NB_EXP_730_352_UNFREEZE_P1
MODEL TRN_LOSS:  0.224072
MODEL VAL_LOSS:  0.342114
QUADR KAPPA:     0.890448
LB SCORE:        TBD
SUBMISSION FLN:  TBD
```
Comments: This is trained mainly to use for transfer learning



# EXP_735 (LB: 0.793)
Training on old data with image size ```352```, I am using weitts for transfer learning from the notebook ```EXP_730_352```, ```NB_EXP_730_352_UNFREEZE_P1```. Image were first cropped to remove all the black background using script TBD.

### EXP_735.ipynb
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
                 zoom_crop(scale=(1.02, 1.35), do_rand=True))]
                 
NORMALIZE:       IMAGENET
TRAINING:        fit_one_cycle(10, 1e-3,    wd=1e-2, div_factor=25, pct_start=0.3)-UNF

MODEL WEIGHTS:   NB_EXP_735_UNFREEZE_P1
MODEL TRN_LOSS:  0.233254
MODEL VAL_LOSS:  0.332348
QUADR KAPPA:     0.893338
LB SCORE:        TBD
SUBMISSION FLN:  TBD
```
Comments: Looks good move to cv

### [EXP_735-CV_0 - EXP_735-CV_4].ipynb
Using weights ``` NB_EXP_735_UNFREEZE_P1 ```To train NEW DATA with 5 fold splits. <br/>

Set up for all CV experimetns: 
```
MODEL:           EfficientNet-B5
NUM_CLASSES:     1 (5 classes but I am treatign this as a regression problem)
BS:              56
SZ:              352
VALID:           NEW DATA CV SPLIT

TFMS:            [flip(p=0.5), 
                 flip_vert(True), 
                 max_rotate(360), 
                 max_lighting(0.1),
                 max_zoom(1.3),
                 p_lighting(0.5), 
                  zoom_crop(scale=(1.01, 1.35), do_rand=True))]
                 
NORMALIZE:       IMAGENET
TRAINING:        fit_one_cycle(15, 1e-3,   wd=1e-2, div_factor=25, pct_start=0.3)-UNF
```

Summary:

| Notebook Name  | Train Loss | Valid Loss | Quadratic Kappa | Weights |
| ------------- | ------------- | ---------| --------| --------|
| EXP_735-CV_0| 0.250133 | 0.205121 | 0.920307 | NB_EXP_735_CV_0_UNFREEZE_P1| 
| EXP_735-CV_1| 0.135174 | 0.211053 | 0.921838 | NB_EXP_735_CV_1_UNFREEZE_P1| 
| EXP_735-CV_2| 0.176270 | 0.170685 | 0.932106 | NB_EXP_735_CV_2_UNFREEZE_P1| 
| EXP_735-CV_3| 0.230097 | 0.223343 | 0.915626 | NB_EXP_735_CV_3_UNFREEZE_P1| 
| EXP_735-CV_4| 0.146527 | 0.205450 | 0.931362 | NB_EXP_735_CV_4_UNFREEZE_P1| 

Submission (Average all the predictions)
```
LB SCORE:        0.793
SUBMISSION FLN:  EXP_352_crop(version 25/25)
```
# EXP_740 (LB:  0.821, PB: 0.926)
In this experiment I combine OLD DATA with NEW DATA and do ```StratifiedKFold``` 5 Fold CV. Before combining I remove in NEW DATA all the duplicates and confusing label images (see in notebooks function ```get_ign_list```). Moreover images are preprocessed using ```PROCCES.ipynb```. This processing helps to remove extra black baground and center the images. Training is done in 3 phases with graudela increasing image sizes - ```224, 352, 448 ```

Set up for all CV experimetns: 
### IMG SIZE 224
```
MODEL:           EfficientNet-B5
NUM_CLASSES:     1 (5 classes but I am treatign this as a regression problem)
BS:              128
SZ:              224
VALID:           StratifiedKFold split of combined data

TFMS:            [flip(p=0.5), 
                 flip_vert(True), 
                 max_rotate(360), 
                 max_lighting(0.1),
                 max_zoom(1.3),
                 p_lighting(0.5), 
                 zoom_crop(scale=(1.01, 1.35), do_rand=True))]
                 
NORMALIZE:       IMAGENET
TRAINING:        fit_one_cycle(10, 1e-2/7,   wd=1e-2, div_factor=25, pct_start=0.3)-UNF
                 fit_one_cycle(5,  1e-2/7/5, wd=1e-2, div_factor=25, pct_start=0.3)-UNF

```
| Notebook Name  | Train Loss | Valid Loss | Quadratic Kappa | Weights |
| ------------- | ------------- | ---------| --------| --------|
| EXP_740-CV_0| 0.307050 | 0.342956 | 0.757189 | NB_EXP_740_CV_0_UNFREEZE_P2| 
| EXP_740-CV_1| 0.302008 | 0.337840 | 0.756192 | NB_EXP_740_CV_1_UNFREEZE_P2| 
| EXP_740-CV_2| 0.318744 | 0.368400 | 0.740174 | NB_EXP_740_CV_2_UNFREEZE_P2| 
| EXP_740-CV_3| 0.306437 | 0.330385 | 0.772315 | NB_EXP_740_CV_3_UNFREEZE_P2| 
| EXP_740-CV_4| 0.308271 | 0.344059 | 0.758115 | NB_EXP_740_CV_4_UNFREEZE_P2| 

### IMG SIZE 352
```
MODEL:           EfficientNet-B5
NUM_CLASSES:     1 (5 classes but I am treatign this as a regression problem)
BS:              352
SZ:              52
VALID:           StratifiedKFold split of combined data

TFMS:            [flip(p=0.5), 
                 flip_vert(True), 
                 max_rotate(360), 
                 max_lighting(0.1),
                 max_zoom(1.3),
                 p_lighting(0.5), 
                 zoom_crop(scale=(1.01, 1.35), do_rand=True))]
                 
NORMALIZE:       IMAGENET
TRAINING:        fit_one_cycle(5, 1e-3/8,   wd=1e-2, div_factor=25, pct_start=0.3)-UNF

```
| Notebook Name  | Train Loss | Valid Loss | Quadratic Kappa | Weights |
| ------------- | ------------- | ---------| --------| --------|
| EXP_740-CV_0| 0.234084 | 0.252582 | 0.821898 | NB_EXP_740_CV_0_352_UNFREEZE_P1| 
| EXP_740-CV_1| 0.243329 | 0.245724 | 0.825637 | NB_EXP_740_CV_1_352_UNFREEZE_P1| 
| EXP_740-CV_2| 0.249529 | 0.262216 | 0.814094 | NB_EXP_740_CV_2_352_UNFREEZE_P1| 
| EXP_740-CV_3| 0.253743 | 0.242828 | 0.832840 | NB_EXP_740_CV_3_352_UNFREEZE_P1| 
| EXP_740-CV_4| 0.252761 | 0.249456 | 0.823078 | NB_EXP_740_CV_4_352_UNFREEZE_P1| 

### IMG SIZE 448
```
MODEL:           EfficientNet-B5
NUM_CLASSES:     1 (5 classes but I am treatign this as a regression problem)
BS:              448
SZ:              32
VALID:           StratifiedKFold split of combined data

TFMS:            [flip(p=0.5), 
                 flip_vert(True), 
                 max_rotate(360), 
                 max_lighting(0.1),
                 max_zoom(1.3),
                 p_lighting(0.5), 
                 zoom_crop(scale=(1.01, 1.35), do_rand=True))]
                 
NORMALIZE:       IMAGENET
TRAINING:        fit_one_cycle(5, 1e-3/4,   wd=1e-2, div_factor=25, pct_start=0.3)-UNF

```

| Notebook Name  | Train Loss | Valid Loss | Quadratic Kappa | Weights |
| ------------- | ------------- | ---------| --------| --------|
| EXP_740-CV_0| 0.242496 | 0.246118 | 0.815219 | NB_EXP_740_CV_0_448_UNFREEZE_P1| 
| EXP_740-CV_1| 0.231280 | 0.231828 | 0.822380 | NB_EXP_740_CV_1_448_UNFREEZE_P1| 
| EXP_740-CV_2| 0.248360 | 0.249534 | 0.813643 | NB_EXP_740_CV_2_448_UNFREEZE_P1| 
| EXP_740-CV_3| 0.265476 | 0.231223 | 0.829990 | NB_EXP_740_CV_3_448_UNFREEZE_P1| 
| EXP_740-CV_4| 0.238991 | 0.235502 | 0.826658 | NB_EXP_740_CV_4_448_UNFREEZE_P1| 

```
CV SCORE:        0.821578 
LB SCORE:        0.818
SUBMISSION FLN:  EXP_740(version 32/32)
```

### IMG SIZE 448
```
MODEL:           EfficientNet-B5
NUM_CLASSES:     1 (5 classes but I am treatign this as a regression problem)
SZ:              456
BS:              32
VALID:           StratifiedKFold split of combined data

TFMS:            [flip(p=0.5), 
                 flip_vert(True), 
                 max_rotate(360), 
                 max_lighting(0.1),
                 max_zoom(1.3),
                 p_lighting(0.5), 
                 zoom_crop(scale=(1.01, 1.35), do_rand=True))]
                 
NORMALIZE:       IMAGENET
TRAINING:        fit_one_cycle(5, 1e-3/4,   wd=1e-2, div_factor=25, pct_start=0.3)-UNF

```

| Notebook Name  | Train Loss | Valid Loss | Quadratic Kappa | Weights |
| ------------- | ------------- | ---------| --------| --------|
| EXP_740-CV_0| 0.232561 | 0.247312 | 0.813309 | NB_EXP_740_CV_0_448_UNFREEZE_P1_rs1| 
| EXP_740-CV_1| 0.212561 | 0.240302 | 0.816792 | NB_EXP_740_CV_1_448_UNFREEZE_P1_rs1| 
| EXP_740-CV_2| 0.219508 | 0.258105 | 0.805388 | NB_EXP_740_CV_2_448_UNFREEZE_P1_rs1| 
| EXP_740-CV_3| 0.251555 | 0.237465 | 0.822387 | NB_EXP_740_CV_3_448_UNFREEZE_P1_rs1| 
| EXP_740-CV_4| 0.235883 | 0.242834 | 0.819575 | NB_EXP_740_CV_4_448_UNFREEZE_P1_rs1| 

```
CV SCORE:        0.823
LB SCORE:        0.821
SUBMISSION FLN:  XP_740_448_rs(version 53/53)
```


# EXP_765 (LB:  0.816: PB: 0.927)
Exactly like EXP_740, except training is done in 2 phases with graudela increasing image sizes - ```224, 380 ``` with the model ```EfficientNet-B5```. I have used Lookahead with Radam as optimizers. See Notebooks for more details

Set up for all CV experimetns: 

### IMG SIZE 224

```
MODEL:           EfficientNet-B4
NUM_CLASSES:     1 (5 classes but I am treatign this as a regression problem)
BS:              128
SZ:              224
VALID:           StratifiedKFold split of combined data

TFMS:            [flip(p=0.5), 
                 flip_vert(True), 
                 max_rotate(360), 
                 max_lighting(0.1),
                 p_lighting(0.5), 
                 zoom_crop(scale=(1.01, 1.45), do_rand=True))]
                 
NORMALIZE:       IMAGENET
TRAINING:        fit_one_cycle(20, 1e-3,   wd=1e-2, div_factor=25, pct_start=0.3)-UNF
```

| Notebook Name  | Train Loss | Valid Loss | Quadratic Kappa | Weights |
| ------------- | ------------- | ---------| --------| --------|
| EXP_740-CV_0| 0.284535 | 0.349184 | 0.754262 | NB_EXP_765_CV_0_UNFREEZE_P1| 
| EXP_740-CV_1| 0.276863 | 0.350727 | 0.757114 | NB_EXP_765_CV_1_UNFREEZE_P1| 
| EXP_740-CV_2| 0.279066 | 0.361914 | 0.745415 | NB_EXP_765_CV_2_UNFREEZE_P1| 
| EXP_740-CV_3| 0.271024 | 0.332245 | 0.777068 | NB_EXP_765_CV_3_UNFREEZE_P1| 
| EXP_740-CV_4| 0.282613 | 0.351242 | 0.756870 | NB_EXP_765_CV_4_UNFREEZE_P1| 

### IMG SIZE 380

```
MODEL:           EfficientNet-B4
NUM_CLASSES:     1 (5 classes but I am treatign this as a regression problem)
BS:              380
SZ:              64
VALID:           StratifiedKFold split of combined data

TFMS:            [flip(p=0.5), 
                 flip_vert(True), 
                 max_rotate(360), 
                 max_lighting(0.1),
                 p_lighting(0.5), 
                 zoom_crop(scale=(1.01, 1.35), do_rand=True))]
                 
NORMALIZE:       IMAGENET
TRAINING:        fit_one_cycle(5, 1e-3,   wd=1e-2, div_factor=25, pct_start=0.3)-UNF
```

| Notebook Name  | Train Loss | Valid Loss | Quadratic Kappa | Weights |
| ------------- | ------------- | ---------| --------| --------|
| EXP_740-CV_0| 0.209359 | 0.250411 | 0.823727 | NB_EXP_765_CV_0_380_UNFREEZE_P1| 
| EXP_740-CV_1| 0.219378 | 0.244317 | 0.827123 | NB_EXP_765_CV_1_380_UNFREEZE_P1| 
| EXP_740-CV_2| 0.213898 | 0.256796 | 0.822892 | NB_EXP_765_CV_2_380_UNFREEZE_P1| 
| EXP_740-CV_3| 0.213808 | 0.235958 | 0.841296 | NB_EXP_765_CV_3_380_UNFREEZE_P1| 
| EXP_740-CV_4| 0.228376 | 0.240436 | 0.837558 | NB_EXP_765_CV_4_380_UNFREEZE_P1| 


```
CV SCORE:        0.831
LB SCORE:        0.816
SUBMISSION FLN:  EXP_740(version 32/32)
```
