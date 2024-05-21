import os
import numpy as np
import pytesseract
from pytesseract import Output
import cv2

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

# https://tesseract-ocr.github.io/tessdoc/


class PytOCR:
    def __init__(self, all_words_pth: str = os.path.join('data', 'english_words.txt')):
        self.tess = pytesseract
        self.all_words_pth = all_words_pth
        with open(self.all_words_pth, 'r') as f:
            self.set_all_words = set(f.read().splitlines())

    def img2data(self, img: np.array = None):
        '''
        Extract text data from the image.
        Input:
            img: np.array, image to extract text data from.
        Output:
            dict_text_box: dict, dictionary containing text as keys, and a list containing four values of elements 'left', 'top', 'width', 'height' as values.
        '''
        tess_text = self.tess.image_to_data(
            img, output_type=Output.DICT, lang='eng', config='--psm 6')
        # convert the tess_text into a dictionary, with 'text' as keys, and a list containing four values of elements 'left', 'top', 'width', 'height' as values
        dict_text_box = dict()
        for i in range(len(tess_text['text'])):
            if len(tess_text['text'][i]) > 1:
                dict_text_box[tess_text['text'][i]] = [tess_text['left'][i],
                                                       tess_text['top'][i],
                                                       tess_text['width'][i],
                                                       tess_text['height'][i]
                                                       ]

        return dict_text_box

    def draw_bounding_box(self, img: np.ndarray, dict_text_box: dict = None, save_pth: str = None, margin_w: int = 1, margin_h: int = 1):
        '''
        Draw bounding box around the text in the image.
        Input:
            img: np.array, image to extract text data from.
            dict_text_box: dict, dictionary containing text as keys, and a list containing four values of elements 'left', 'top', 'width', 'height' as values.
            save_pth: str, path to save the image with bounding box.
            margin_w: int, margin width of the bounding box. Default is 1.
            margin_h: int, margin height of the bounding box. Default is 1.
        Output:
            img: np.array, image with bounding box around the text.
        '''
        if dict_text_box == None:
            dict_text_box = self.img2data(img)

        for (text, box) in dict_text_box.items():
            (x, y, w, h) = box
            cv2.rectangle(img,
                          (x-margin_w, y-margin_h),
                          (x + w + margin_w, y + h + margin_h),
                          (255, 0, 0),
                          1)
            im = Image.fromarray(img)
            draw = ImageDraw.Draw(im)

            font_pth = 'font/arial.ttf'
            font = ImageFont.truetype(
                font=font_pth, size=18, encoding='utf-8')

            draw.text(
                (x, y - 20), text, (255, 0, 0), font=font)
            img = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

        if save_pth:
            cv2.imwrite(save_pth, img)

        return img

    def split_img(self, img: np.array, dict_text_box: dict, save_dir: str, margin_w: int = 1, margin_h: int = 1):
        '''
        Split the image into multiple images based on the bounding box.
        Input:
            img: np.array, image to extract text data from.
            dict_text_box: dict, dictionary containing text as keys, and a list containing four values of elements 'left', 'top', 'width', 'height' as values.
            save_dir: str, directory to save the cropped images.
            margin_w: int, margin width of the bounding box. Default is 1.
            margin_h: int, margin height of the bounding box. Default is 1.
        Output:
            None
        '''
        img_h, img_w = img.shape[:2]

        for (text, box) in dict_text_box.items():
            # skip all the single character words and possible misrecognized words
            if len(text) > 1 and text.lower() in self.set_all_words:
                (x, y, w, h) = box

                l = max(x - margin_w, 0)
                r = min(x + margin_w + w, img_w)
                u = max(y - margin_h, 0)
                d = min(y + margin_h + h, img_h)
                cropped_img = img[u:d, l:r]

                cv2.imwrite(os.path.join(
                    save_dir, text+'.png'), cropped_img)
        return None
