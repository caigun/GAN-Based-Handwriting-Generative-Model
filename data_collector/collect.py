import os
import cv2
from PIL import Image
from ocr import PytOCR


data_dir = 'data'
new_dataset_dir = os.path.join(data_dir, 'new_dataset')

page_img_dir = os.path.join(new_dataset_dir, 'page_img')
box_img_dir = os.path.join(new_dataset_dir, 'box_img')
word_img_dir = os.path.join(new_dataset_dir, 'word_img')

if not os.path.exists(box_img_dir):
    os.makedirs(box_img_dir)
if not os.path.exists(word_img_dir):
    os.makedirs(word_img_dir)

# https://github.com/first20hours/google-10000-english
ocr = PytOCR(all_words_pth='data/google-10000-english.txt')


for img_name in os.listdir(page_img_dir):
    writer_idx, img_ext = img_name.split(".")
    writer, idx = writer_idx.rsplit("_", 1)
    img_ext = img_ext.lower()

    if img_ext not in ["png", "jpg", "jpeg"]:
        continue
    if img_ext in ["jpg", "jpeg"]:
        # convert jpg/jpeg to png, delete the jpg/jpeg version and save the png version
        img_path = os.path.join(page_img_dir, img_name)
        img = Image.open(img_path)
        png_name = img_name.rsplit('.', 1)[0] + '.png'
        png_path = os.path.join(page_img_dir, png_name)
        img.save(png_path)
        os.remove(img_path)
        img_name = png_name

    img_path = os.path.join(page_img_dir, img_name)
    img = cv2.imread(img_path)

    # draw and save bounding box images to visualize recognition result
    dict_text_box = ocr.img2data(img)
    ocr.draw_bounding_box(
        img.copy(), dict_text_box, save_pth=os.path.join(box_img_dir, img_name))
    # split the image into multiple images based on the bounding box
    save_dir = os.path.join(word_img_dir, writer)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ocr.split_img(img, dict_text_box, save_dir, margin_w=1, margin_h=1)
