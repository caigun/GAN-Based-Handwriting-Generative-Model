import os
import cv2
import matplotlib.pyplot as plt


########################################################################
# NOTE:
# ONLY RUN THIS SCRIPT ONCE FOR THE DIRECTORY `_page_img_i` FOR i CORRESPONDING TO IAMGES FROM AP EXAM
########################################################################

# used to remove lines with deep color from the image
data_dir = 'data'
new_dataset_dir = os.path.join(data_dir, 'new_dataset')
dir_idx = ['4', '6']
page_img_dirs = [os.path.join(
    new_dataset_dir, '_page_img_') + i for i in dir_idx]

for page_img_dir in page_img_dirs:
    for page_name in os.listdir(page_img_dir):
        ext = page_name.rsplit('.', 1)[-1]
        if ext not in ['jpg', 'png', 'jpeg']:
            continue

        page_path = os.path.join(page_img_dir, page_name)
        img = cv2.imread(page_path)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
        blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
        result_raw = 255-blackhat
        result = cv2.cvtColor(result_raw, cv2.COLOR_BGR2GRAY)

        # save the image and remove the original one
        # os.remove(page_path)
        cv2.imwrite(page_path, result)

    # # show the last image
    # plt.imshow(result, cmap='gray')
    # plt.show()
