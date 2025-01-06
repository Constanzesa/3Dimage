import os
import shutil
import numpy as np

# https://github.com/eeyhsong/NICE-EEG/issues/2

things_path = './data/THINGS/Images/'
things_eeg_test_images_path = './data/Things-EEG2/Image_set/image_set/test_images/'
things_eeg_center_images_path = './data/Things-EEG2/Image_set/image_set/center_images/'

things_list = os.listdir(things_path)[6:]
print(things_list)
things_list.sort()
test_list = os.listdir(things_eeg_test_images_path)
test_list.sort()
# center_list = os.listdir(things_eeg_center_images_path)
for i in range(len(test_list)):
    print("path = ", things_path, test_list[i], test_list[i][6:], things_eeg_center_images_path, test_list[i][6:])
    shutil.copytree(things_path+test_list[i][6:], things_eeg_center_images_path+test_list[i][6:])
    os.rename(things_eeg_center_images_path+test_list[i][6:], things_eeg_center_images_path+test_list[i])
    test_img = os.listdir(things_eeg_test_images_path+test_list[i])
    os.unlink(things_eeg_center_images_path+test_list[i]+'/'+test_img[0])

print('ttt')