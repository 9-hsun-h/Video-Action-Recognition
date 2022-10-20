#%%
import pandas as pd
import joblib
import os
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer

#%%
# get all the image folder paths
all_paths = os.listdir('C:/CCBDA/HW1/ccbda-2022-hw1-data/train_jpg')
folder_paths = [x for x in all_paths if os.path.isdir('C:/CCBDA/HW1/ccbda-2022-hw1-data/train_jpg/' + x)]
folder_paths = [int(i) for i in folder_paths]
folder_paths = sorted(folder_paths)
folder_paths = [str(i) for i in folder_paths]
print(f"Folder paths: {folder_paths}")
print(f"Number of folders: {len(folder_paths)}")

#%%
# we will create the data for the following labels, 
# add more to list to use those for creating the data as well
create_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38']
# create a DataFrame
data = pd.DataFrame()

#%%
#image_formats = ['jpg', 'JPG', 'PNG', 'png'] # we only want images that are in this format
image_formats = ['jpg'] # we only want images that are in this format
labels = []
counter = 0
# for i, folder_path in tqdm(enumerate(folder_paths), total=len(folder_paths)):
#     if folder_path not in create_labels:
#         continue
#     image_paths = os.listdir('C:/CCBDA/HW1/ccbda-2022-hw1-data/train_jpg/'+folder_path)
#     label = folder_path
#     # save image paths in the DataFrame
#     for image_path in image_paths:
#         if image_path.split('.')[-1] in image_formats:
#             data.loc[counter, 'image_path'] = f"C:/CCBDA/HW1/ccbda-2022-hw1-data/train_jpg/{folder_path}/{image_path}"
#             labels.append(label)
#             counter += 1

for i, folder_path in tqdm(enumerate(folder_paths), total=len(folder_paths)):
    if folder_path not in create_labels:
        continue
    videos_paths = os.listdir('C:/CCBDA/HW1/ccbda-2022-hw1-data/train_jpg/'+folder_path)
    label = folder_path
    for video_path in videos_paths:
        image_paths  = os.listdir('C:/CCBDA/HW1/ccbda-2022-hw1-data/train_jpg/'+folder_path+'/'+video_path)
        # save image paths in the DataFrame
        for image_path in image_paths:
            if image_path.split('.')[-1] in image_formats:
                data.loc[counter, 'image_path'] = f"C:/CCBDA/HW1/ccbda-2022-hw1-data/train_jpg/{folder_path}/{video_path}/{image_path}"
                labels.append(label)
                counter += 1

#%%
# labels = np.array(labels)
# # one-hot encode the labels
# lb = LabelBinarizer()
# labels = lb.fit_transform(labels)

#%%
if len(labels[0]) == 1:
    for i in range(len(labels)):
        index = labels[i]
        data.loc[i, 'target'] = int(index)
elif len(labels[0]) > 1:
    for i in range(len(labels)):
        index = np.argmax(labels[i])
        data.loc[i, 'target'] = int(index)

#%%
# shuffle the dataset
data = data.sample(frac=1).reset_index(drop=True)
#print(f"Number of labels or classes: {len(lb.classes_)}")
print(f"Number of labels or classes: {len(create_labels)}")
#print(f"The first one hot encoded labels: {labels[0]}")
print(f"The first label: {labels[0]}")
#print(f"Mapping the first one hot encoded label to its category: {lb.classes_[0]}")
print(f"Total instances: {len(data)}")
 
# save as CSV file
data.to_csv('C:/CCBDA/HW1/V2/data.csv', index=False)
 
# pickle the binarized labels
# print('Saving the binarized labels as pickled file')
# joblib.dump(lb, 'C:/CCBDA/HW1/V2/lb.pkl')
 
print(data.head(5))
# %%
