#%%
'''
USAGE:
python test.py --model ../outputs/sports.pth --label-bin ../outputs/lb.pkl --input ../input/example_clips/chess.mp4 --output ../outputs/chess.mp4
'''
import torch
import numpy as np
import argparse
import joblib
import cv2
import torch.nn as nn
import torch.nn.functional as F
import time
import cnn_models
import albumentations
from torchvision.transforms import transforms   
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

def most_frequent(List):
    return max(set(List), key = List.count) 
#%%
# construct the argument parser
# ap = argparse.ArgumentParser()
# ap.add_argument('-m', '--model', required=True,
#     help="path to trained serialized model")
# ap.add_argument('-l', '--label-bin', required=True,
#     help="path to  label binarizer")
# ap.add_argument('-i', '--input', required=True,
#     help='path to our input video')
# ap.add_argument('-o', '--output', required=True, type=str,
#     help='path to our output video')
# args = vars(ap.parse_args())

#%%
# load the trained model and label binarizer from disk
# print('Loading model and label binarizer...')
# lb = joblib.load(f'C:/CCBDA/HW1/V2/lb.pkl')
model = cnn_models.CustomResNet34().cuda()
print('Model Loaded...')
model.load_state_dict(torch.load(f'C:/CCBDA/HW1/V2/model_weight/resnet34_epoch_{49}.pt'))
print('Loaded model state_dict...')
aug = albumentations.Compose([
    albumentations.Resize(224, 224),
    ])

#%%
predict_list = []
for i in range(1,10001):
    if i < 10:
        index = f'0000{i}'
    elif i < 100:
        index = f'000{i}'
    elif i < 1000:
        index = f'00{i}'
    elif i < 10000:
        index = f'0{i}'
    else:
        index = f'{i}'
    cap = cv2.VideoCapture(f"C:/CCBDA/HW1/ccbda-2022-hw1-data/test/{index}.mp4")
    if (cap.isOpened() == False):
        print('Error while trying to read video. Plese check again...')
    # get the frame width and height
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # define codec and create VideoWriter object
    #out = cv2.VideoWriter(str(args['output']), cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width,frame_height))

    # read until end of video
    preds = []
    while(cap.isOpened()):
        # capture each frame of the video
        ret, frame = cap.read()
        if ret == True:
            model.eval()
            with torch.no_grad():
                # conver to PIL RGB format before predictions
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                pil_image = aug(image=np.array(pil_image))['image']
                pil_image = np.transpose(pil_image, (2, 0, 1)).astype(np.float32)
                pil_image = torch.tensor(pil_image, dtype=torch.float).cuda()
                pil_image = pil_image.unsqueeze(0)
                
                outputs = model(pil_image)
                _, pred = torch.max(outputs.data, 1)
                preds.append(pred.item())
            # cv2.putText(frame, lb.classes_[preds], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)
            # cv2.imshow('image', frame)
            # out.write(frame)
            # # press `q` to exit
            # if cv2.waitKey(27) & 0xFF == ord('q'):
            #     break
        else: 
            break

    print(f"most frequent prediction of {index}.mp4 is {most_frequent(preds)}")
    predict_list.append(most_frequent(preds))
    # release VideoCapture()
    cap.release()
    # close all frames and video windows
    cv2.destroyAllWindows()

#%%
with open("prediction_V2_resnet34_epoch49.csv", 'w') as f:
        f.write("name,label\n")
        for i in range(10000):
            f.write(f"{i + 1:05d}.mp4,{predict_list[i]}\n")
# %%
