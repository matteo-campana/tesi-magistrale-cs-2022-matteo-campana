# Image Segmentation with Swin-Transformers


## Usage

### To reproduce the original environment
- create a virtualenv and install the packages required to reproduce the environment used to code this repo.
- run the below command to install all the needed packages.
```
pip install -r requirements.txt
```

### Pyhton and PyTorch versions

- pytorch version 1.10.0+cu111
- torchvision version 0.11.1+cu111
- python version 3.7.12
<!--
```
#pip install glob
#pip install timm
#pip install torch torchvision
#pip install numpy
#pip install pyyaml
#pip install numba
#pip install tensorboard
#pip install matplotlib
```
-->
---

### change the config forn training

- open and modify the file pretrain.yaml or train_pl_ss.yaml inside the config folder
- select the model that you want to train in
```
model : "MASWinTNet-ss"
```
- setting up where your data folders are located in
```
train_data_path : "[TRAIN FOLDER]"
valid_data_path : "[VALID FOLDER]"
```
- Define numbers of epochs, batch size will be automatically detected by Pytorch Ligthening Tuner
```
epochs : 100
```

---
In order to train  
- A semantic segmentation vanilla model run the command
```
python main_pl_ss.py
```


---
In order to train  
- to pretrain the networks with dino and perform semantic segmentation task run the command
```
python main_pl_ss_dino.py
```

the training parameters are defined in the .yaml config file inside the relative folder.

---

## Results

### vanilla results

mean value over the last 10 epochs on Validation Set

| name                | resolution | mIoU  | pixel_acc     | mse       | #params |
| :------------------ | :--------: | :---: | :-----------: | --------: | ------: |
| MASWinT-Net-Nano    | 224x224    | TODO  | TODO          | TODO      | 25.7 M  |
| MASWinT-Net-Nano-VQ | 224x224    | TODO  | TODO          | TODO      | 26.9 M  |
| MASWinT-Net-Tiny    | 224x224    | TODO  | TODO          | TODO      | 38.7 M  |
| MASWinT-Net-Tiny-VQ | 224x224    | TODO  | TODO          | TODO      | 39.9 M  |
| MaResU-Net-18       | 224x224    | TODO  | TODO          | TODO      | 16.2 M  |
| MaResU-Net-18-VQ    | 224x224    | TODO  | TODO          | TODO      | 16.7 M  |
| MaResU-Net-50       | 224x224    | TODO  | TODO          | TODO      | 112   M |
| MaResU-Net-50-VQ    | 224x224    | TODO  | TODO          | TODO      | 121   M |



### pretrain results
| name                | resolution | mIoU  | pixel_acc     | mse       | #params |
| :------------------ | :--------: | :---: | :-----------: | --------: | ------: |
| MASWinT-Net-Nano    | 224x224    | TODO  | TODO          | TODO      | 25.7 M  |
| MASWinT-Net-Nano-VQ | 224x224    | TODO  | TODO          | TODO      | 26.9 M  |
| MASWinT-Net-Tiny    | 224x224    | TODO  | TODO          | TODO      | 38.7 M  |
| MASWinT-Net-Tiny-VQ | 224x224    | TODO  | TODO          | TODO      | 39.9 M  |
| MaResU-Net-18       | 224x224    | TODO  | TODO          | TODO      | 16.2 M  |
| MaResU-Net-18-VQ    | 224x224    | TODO  | TODO          | TODO      | 16.7 M  |
| MaResU-Net-50       | 224x224    | TODO  | TODO          | TODO      | 112   M |
| MaResU-Net-50-VQ    | 224x224    | TODO  | TODO          | TODO      | 121   M |


---

###  mIoU/Jaccard Index Train

|    | Model                                    |    mIoU  |
|---:|:-----------------------------------------|---------:|
|  0 | MAResUNet_18_imagenet over last epoch    | 0.588168 |
|  1 | MAResUNet_50_EsViT over last epoch       | 0.601395 |
|  2 | MAResUNet_50_imagenet over last epoch    | 0.612408 |
|  3 | MAResUNet_vq_18_imagenet over last epoch | 0.601631 |
|  4 | MAResUNet_vq_50_EsViT over last epoch    | 0.544145 |
|  5 | MAResUNet_vq_50_imagenet over last epoch | 0.595947 |

|    | Model                                        |    mIoU  |
|---:|:---------------------------------------------|---------:|
|  0 | MAResUNet_18_imagenet over last 10 epochs    | 0.584018 |
|  1 | MAResUNet_50_EsViT over last 10 epochs       | 0.597124 |
|  2 | MAResUNet_50_imagenet over last 10 epochs    | 0.60948  |
|  3 | MAResUNet_vq_18_imagenet over last 10 epochs | 0.592476 |
|  4 | MAResUNet_vq_50_EsViT over last 10 epochs    | 0.538529 |
|  5 | MAResUNet_vq_50_imagenet over last 10 epochs | 0.59872  |

### mIoU/Jaccard Index Validation

|    | Model                                    | mIoU     |
|---:|:-----------------------------------------|---------:|
|  0 | MAResUNet_18_imagenet over last epoch    | 0.387273 |
|  1 | MAResUNet_50_EsViT over last epoch       | 0.366835 |
|  2 | MAResUNet_50_imagenet over last epoch    | 0.382851 |
|  3 | MAResUNet_vq_18_imagenet over last epoch | 0.432939 |
|  4 | MAResUNet_vq_50_EsViT over last epoch    | 0.414664 |
|  5 | MAResUNet_vq_50_imagenet over last epoch | 0.428687 |


|    | Model                                        |    mIoU  |
|---:|:---------------------------------------------|---------:|
|  0 | MAResUNet_18_imagenet over last 10 epochs    | 0.403535 |
|  1 | MAResUNet_50_EsViT over last 10 epochs       | 0.380418 |
|  2 | MAResUNet_50_imagenet over last 10 epochs    | 0.363466 |
|  3 | MAResUNet_vq_18_imagenet over last 10 epochs | 0.428589 |
|  4 | MAResUNet_vq_50_EsViT over last 10 epochs    | 0.39112  |
|  5 | MAResUNet_vq_50_imagenet over last 10 epochs | 0.419492 |

---

## accuracy Train

|    | Model                                    |   accuracy |
|---:|:-----------------------------------------|-----------:|
|  0 | MAResUNet_18_imagenet over last epoch    |   0.80961  |
|  1 | MAResUNet_50_EsViT over last epoch       |   0.829796 |
|  2 | MAResUNet_50_imagenet over last epoch    |   0.8325   |
|  3 | MAResUNet_vq_18_imagenet over last epoch |   0.819065 |
|  4 | MAResUNet_vq_50_EsViT over last epoch    |   0.793297 |
|  5 | MAResUNet_vq_50_imagenet over last epoch |   0.820648 |

|    | Model                                        |   accuracy |
|---:|:---------------------------------------------|-----------:|
|  0 | MAResUNet_18_imagenet over last 10 epochs    |   0.806369 |
|  1 | MAResUNet_50_EsViT over last 10 epochs       |   0.826565 |
|  2 | MAResUNet_50_imagenet over last 10 epochs    |   0.831056 |
|  3 | MAResUNet_vq_18_imagenet over last 10 epochs |   0.811945 |
|  4 | MAResUNet_vq_50_EsViT over last 10 epochs    |   0.788985 |
|  5 | MAResUNet_vq_50_imagenet over last 10 epochs |   0.8228   |

## accuracy Validation

|    | Model                                    |   accuracy |
|---:|:-----------------------------------------|-----------:|
|  0 | MAResUNet_18_imagenet over last epoch    |   0.62851  |
|  1 | MAResUNet_50_EsViT over last epoch       |   0.614038 |
|  2 | MAResUNet_50_imagenet over last epoch    |   0.623326 |
|  3 | MAResUNet_vq_18_imagenet over last epoch |   0.677248 |
|  4 | MAResUNet_vq_50_EsViT over last epoch    |   0.654268 |
|  5 | MAResUNet_vq_50_imagenet over last epoch |   0.681332 |

|    | Model                                        |   accuracy |
|---:|:---------------------------------------------|-----------:|
|  0 | MAResUNet_18_imagenet over last 10 epochs    |   0.640804 |
|  1 | MAResUNet_50_EsViT over last 10 epochs       |   0.634995 |
|  2 | MAResUNet_50_imagenet over last 10 epochs    |   0.621785 |
|  3 | MAResUNet_vq_18_imagenet over last 10 epochs |   0.677473 |
|  4 | MAResUNet_vq_50_EsViT over last 10 epochs    |   0.638496 |
|  5 | MAResUNet_vq_50_imagenet over last 10 epochs |   0.676439 |

---

# MASwinTNet

##  mIoU/Jaccard Index Train

|    | Model                                  |      mIoU |
|---:|:---------------------------------------|----------:|
|  0 | MASWinTNet over last epoch             | 0.238541  |
|  1 | MASWinTNet_EsViT over last epoch       | 0.254514  |
|  2 | MASWinTNet_EsViT_vq over last epoch    | 0.0414447 |
|  3 | MASWinTNet_imagenet over last epoch    | 0.0415642 |
|  4 | MASWinTNet_imagenet_vq over last epoch | 0.0415641 |
|  5 | MASWinTNet_vq over last epoch          | 0.344936  |

|    | Model                                      |      mIoU |
|---:|:-------------------------------------------|----------:|
|  0 | MASWinTNet over last 10 epochs             | 0.257441  |
|  1 | MASWinTNet_EsViT over last 10 epochs       | 0.273527  |
|  2 | MASWinTNet_EsViT_vq over last 10 epochs    | 0.0648157 |
|  3 | MASWinTNet_imagenet over last 10 epochs    | 0.0415642 |
|  4 | MASWinTNet_imagenet_vq over last 10 epochs | 0.0415641 |
|  5 | MASWinTNet_vq over last 10 epochs          | 0.331601  |

##  mIoU/Jaccard Index Validation

|    | Model                                  |     mIoU |
|---:|:---------------------------------------|---------:|
|  0 | MASWinTNet over last epoch             | 0.152228 |
|  1 | MASWinTNet_EsViT over last epoch       | 0.211923 |
|  2 | MASWinTNet_EsViT_vq over last epoch    | 0.055805 |
|  3 | MASWinTNet_imagenet over last epoch    | 0.055802 |
|  4 | MASWinTNet_imagenet_vq over last epoch | 0.055802 |
|  5 | MASWinTNet_vq over last epoch          | 0.310147 |

|    | Model                                      |      mIoU |
|---:|:-------------------------------------------|----------:|
|  0 | MASWinTNet over last 10 epochs             | 0.190263  |
|  1 | MASWinTNet_EsViT over last 10 epochs       | 0.233886  |
|  2 | MASWinTNet_EsViT_vq over last 10 epochs    | 0.0665292 |
|  3 | MASWinTNet_imagenet over last 10 epochs    | 0.055802  |
|  4 | MASWinTNet_imagenet_vq over last 10 epochs | 0.055802  |
|  5 | MASWinTNet_vq over last 10 epochs          | 0.278946  |

--- 

## accuracy Train

|    | Model                                  |   accuracy |
|---:|:---------------------------------------|-----------:|
|  0 | MASWinTNet over last epoch             |   0.515631 |
|  1 | MASWinTNet_EsViT over last epoch       |   0.547259 |
|  2 | MASWinTNet_EsViT_vq over last epoch    |   0.247352 |
|  3 | MASWinTNet_imagenet over last epoch    |   0.249385 |
|  4 | MASWinTNet_imagenet_vq over last epoch |   0.249385 |
|  5 | MASWinTNet_vq over last epoch          |   0.613597 |

|    | Model                                      |   accuracy |
|---:|:-------------------------------------------|-----------:|
|  0 | MASWinTNet over last 10 epochs             |   0.543043 |
|  1 | MASWinTNet_EsViT over last 10 epochs       |   0.562272 |
|  2 | MASWinTNet_EsViT_vq over last 10 epochs    |   0.279593 |
|  3 | MASWinTNet_imagenet over last 10 epochs    |   0.249385 |
|  4 | MASWinTNet_imagenet_vq over last 10 epochs |   0.249385 |
|  5 | MASWinTNet_vq over last 10 epochs          |   0.603402 |

## accuracy Validation

|    | Model                                  |   accuracy |
|---:|:---------------------------------------|-----------:|
|  0 | MASWinTNet over last epoch             |   0.37323  |
|  1 | MASWinTNet_EsViT over last epoch       |   0.465948 |
|  2 | MASWinTNet_EsViT_vq over last epoch    |   0.334811 |
|  3 | MASWinTNet_imagenet over last epoch    |   0.334812 |
|  4 | MASWinTNet_imagenet_vq over last epoch |   0.334812 |
|  5 | MASWinTNet_vq over last epoch          |   0.565879 |

|    | Model                                      |   accuracy |
|---:|:-------------------------------------------|-----------:|
|  0 | MASWinTNet over last 10 epochs             |   0.444921 |
|  1 | MASWinTNet_EsViT over last 10 epochs       |   0.499766 |
|  2 | MASWinTNet_EsViT_vq over last 10 epochs    |   0.342275 |
|  3 | MASWinTNet_imagenet over last 10 epochs    |   0.334812 |
|  4 | MASWinTNet_imagenet_vq over last 10 epochs |   0.334812 |
|  5 | MASWinTNet_vq over last 10 epochs          |   0.534676 |
