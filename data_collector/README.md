This directory includes all the python files to process the data.

Necessary python files (File 2-7 should be run in order):
1. `ocr.py`: This script contains a class `PytOCR` that uses the **Tesseract OCR engine** to extract text from images, draw bounding boxes around the extracted text in the images, and split the images based on the bounding boxes. It also filters out single character words and possible misrecognized words based on a provided dictionary.
2. `docx2page.py`: Extract the images from the docx files. One demo docx file is provided in the `data/new_dataset` directory. The extracted images are saved in directories `_page_img_i` (for the i-th docs file) under the `data/new_dataset` directory.
3. `remove_line.py`: Fade out the structural horizontal lines in the images from the `data/new_dataset/page_img` directory. The code `dir_idx = ['4', '6']` is used to specipy which directories to process. The processed images are saved in directories `_page_img_i` (for the i-th docs file) under the `data/new_dataset` directory.
4. `merge_pageimgdir.py`: Merge the images from the directories `_page_img_i` (for the i-th docs file) under the `data/new_dataset` directory into a single directory `data/new_dataset/page_img`.
5. `collect.py`: Use the OCR engine to extract word-level images from the images in the `data/new_dataset/page_img` directory. The extracted images are saved in the `data/new_dataset/word_img` directory (images from each writer will be saved in a separate directory), and the OCR visualization results is saved in the `data/new_dataset/box_img` directory.
6. `img2hdf5.py`: Convert the images in the `data/new_dataset/word_img` directory to a single HDF5 file `data/new_dataset/word_img.h5`.
7. `merge_hdf5.py`: Merge the HDF5 files from IAM dataset and the new dataset.
   - `data/iam/trnvalset_words64_OrgSz.hdf5` + `data/new_dataset/trnvalset_words64_newdataset.hdf5` -> `data/merged_dataset/trnvalset_words64_merged.hdf5`
   - `data/iam/testset_words64_OrgSz.hdf5` + `data/new_dataset/testset_words64_newdataset.hdf5` -> `data/merged_dataset/testset_words64_merged.hdf5`.

Optional python files:
- `check_hdf5.py`: Check the HDF5 file to see if the images are correctly saved.
- `validate.py`: This file can be used to validate the extracted images, but it maybe not necessary.

Remarks:
- Since our newly collected dataset in the HDF5 format is provided, you can skip the steps 2-6 and direcly go to step 7 to merge the HDF5 files. Or you can even download our merged dataset ([`trnvalset_words64_merged.hdf5`](https://huggingface.co/datasets/dearsikadeer/OCRWordImages/blob/2aff07a976df4cc0097136b5f38ed7b0ad4e087f/trnvalset_words64_merged.hdf5) and [`testset_words64_merged.hdf5`](https://huggingface.co/datasets/dearsikadeer/OCRWordImages/blob/2aff07a976df4cc0097136b5f38ed7b0ad4e087f/testset_words64_merged.hdf5)) directly and skip all the steps.
