## Dataset
- Unzip data.zip to `./ccbda-2022-hw1-data`
    ```sh
    unzip data.zip -d ./data
    ```
- Folder structure
    ```
    .
    ├── ccbda-2022-hw1-data
    │   ├── test/
    │   ├── train/
    │   └── train_jpg/
    ├── model_weight
    ├── outputs
    ├── cnn_models.py
    ├── test.py
    ├── Readme.md
    ├── requirements.txt
    ├── myutils.py
    ├── pre_prepare_data.py
    ├── prepare_data.py
    ├── data.csv
    └── train.py
    ```

## Environment
- Python 3.9 or later version
    ```sh
    conda create --name <env> --file requirements.txt
    ```

## Preprocess training data
- Split training video into 10 frames per video.
- If the total length of a video is shorter than 10 frames, it will not be used in training data.
- Create train_jpg directory
```sh
python pre_prepare_data.py
```
- Saving the jpgs paths together with their corresponding labels.
- Create a csv file named as `data.csv`
```sh
python prepare_data.py
```


## Train
- ResNet34
- With RTX 2080ti and 128GB RAM, it may cost a day to train.
- The trained model weight will be saved in folder "model_weight", its weight name **may be** `resnet34_epoch_49.pt`. If the validation loss of this epoch is lower than the current best validation loss. It will be saved.
```sh
python train.py
```

## Make Prediction
- Please use the correct model name and its corresponding weight.
```sh
python test.py
```
The prediction file **may be** `prediction_V2_resnet34_epoch49.csv`.