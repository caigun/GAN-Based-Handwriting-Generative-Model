import os
from PIL import Image
import pytesseract
import cv2
import pytesseract
from pytesseract import Output


folder_path = 'data/new_dataset/word_img'

# iterate through all files in the folder
match = 0
dismatch = 0
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
            # join the path to the file
            file_path = os.path.join(root, file)

            img = Image.open(file_path)
            # gray_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            #         # set a threshold value
            # threshold_value = 128

            # # binarize the image
            # ret, binary_img = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)

            extracted_text = pytesseract.image_to_string(
                img, lang='eng', config='--psm 6')

            # extract title from file name
            title = os.path.splitext(file)[0]

            # check if the extracted text contains the title
            if title.lower() in extracted_text.lower():
                print(f"Image '{file}' text matches title.")
                match = match+1
            else:
                print(f"Image '{file}' text does not match title.")
                # remove the file for which the text does not match the title
                os.remove(file_path)
                dismatch = dismatch+1

print(match)
print(dismatch)
