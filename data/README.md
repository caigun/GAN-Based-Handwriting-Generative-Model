This directory mainly includes the data used in the project. 
Files existed already:
- Three word dictionary files: `english_words.txt`, [`google-10000-english.txt`](https://github.com/first20hours/google-10000-english), and `iam_word_test.txt`.
- Some sample images in the `image_samples` directory.
- `example.docx`: A short docx file to demonstrate the correct format of the docx files to be processed by the `data_collector/docx2page.py` script.

Files need to be downloaded from **Hugging Face**:
- The iam dataset in HDF5 format: [trnvalset_words64_OrgSz.hdf5](https://github.com/ganji15/HiGANplus/releases/download/dataset/trnvalset_words64_OrgSz.hdf5) and [testset_words64_OrgSz.hdf5](https://github.com/ganji15/HiGANplus/releases/download/dataset/testset_words64_OrgSz.hdf5). You need to create a folder `data/iam` and put these two files in it.
- Our newly collected dataset (**OCRWordImages**) in HDF5 format: [trnvalset_words64_newdataset.hdf5](https://huggingface.co/datasets/dearsikadeer/OCRWordImages/blob/2aff07a976df4cc0097136b5f38ed7b0ad4e087f/trnvalset_words64_newdataset.hdf5) and [testset_words64_newdataset.hdf5](https://huggingface.co/datasets/dearsikadeer/OCRWordImages/blob/2aff07a976df4cc0097136b5f38ed7b0ad4e087f/testset_words64_newdataset.hdf5). You need to create a folder `data/newdataset` and put these two files in it.
  
Remark:
- We also offer all the image files in the processed `data/word_img` directory [here](https://huggingface.co/datasets/dearsikadeer/OCRWordImages/tree/2aff07a976df4cc0097136b5f38ed7b0ad4e087f).
