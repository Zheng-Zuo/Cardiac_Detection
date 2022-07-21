# 数据来源：https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
# 数据标注是在224*224的尺寸上进行的，所以bounding box不再进行缩放

from pathlib import Path
import pydicom
import numpy as np
import cv2
import pandas as pd

labels = pd.read_csv("./rsna_heart_detection.csv")
ROOT_PATH = Path("/path/to/rsna-pneumonia-detection-challenge/stage_2_train_images/", )
SAVE_PATH = Path("Processed-Heart-Detection/")

sums = 0
sums_squared = 0
train_ids = []
val_ids = []

for counter, patient_id in enumerate(list(labels.name)):  
    dcm_path = ROOT_PATH/patient_id  
    dcm_path = dcm_path.with_suffix(".dcm")  
    
    dcm = pydicom.read_file(dcm_path)  
     
    dcm_array = dcm.pixel_array
    assert dcm_array.shape == (1024, 1024)

    dcm_array = (cv2.resize(dcm_array, (224, 224)) / 255).astype(np.float16)
            
    # 4/5 train split, 1/5 val split
    train_or_val = "train" if counter < 400 else "val" 
    
    # Add to corresponding train or validation patient index list
    if train_or_val == "train":
        train_ids.append(patient_id)
    else:
        val_ids.append(patient_id)
    
    current_save_path = SAVE_PATH/train_or_val 
    current_save_path.mkdir(parents=True, exist_ok=True)
    
    np.save(current_save_path/patient_id, dcm_array) 
    
    normalizer = dcm_array.shape[0] * dcm_array.shape[1]  # Normalize sum of image
    if train_or_val == "train":  # Only use train data to compute dataset statistics
        sums += np.sum(dcm_array) / normalizer
        sums_squared += (np.power(dcm_array, 2).sum()) / normalizer

np.save("Processed-Heart-Detection/train_subjects_det", train_ids)
np.save("Processed-Heart-Detection/val_subjects_det", val_ids)

mean = sums / len(train_ids)
std = np.sqrt(sums_squared / len(train_ids) - (mean**2), dtype=np.float64)
print(f"Mean of Dataset: {mean}, STD: {std}")