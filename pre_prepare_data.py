#%%
import os

path2data = "C:/CCBDA/HW1/ccbda-2022-hw1-data"
sub_folder = "train"
sub_folder_jpg = "train_jpg"
path2aCatgs = os.path.join(path2data, sub_folder)

listOfCategories = os.listdir(path2aCatgs)
listOfCategories, len(listOfCategories)
#%%
for cat in listOfCategories:
    print("category:", cat)
    path2acat = os.path.join(path2aCatgs, cat)
    listOfSubs = os.listdir(path2acat)
    print("number of sub-folders:", len(listOfSubs))
    print("-"*50)
#%%
import myutils
#%%
extension = ".mp4"
n_frames = 10
for root, dirs, files in os.walk(path2aCatgs, topdown=False):
    for name in files:
        if extension not in name:
            continue
        path2vid = os.path.join(root, name)
        frames, vlen = myutils.get_frames(path2vid, n_frames= n_frames)
        if vlen <10:
            print("this video is too short")
            continue
        path2store = path2vid.replace(sub_folder, sub_folder_jpg)
        path2store = path2store.replace(extension, "")
        print(path2store)
        os.makedirs(path2store, exist_ok= True)
        myutils.store_frames(frames, path2store)
    print("-"*50)    
# %%
