# GAN-Based-Handwriting-Generative-Model

This is the implementation of the course project: __Enhancing GAN-Based Handwriting Generative Model: Handwriting Feature Extraction through LSTM and Transformer__.

The implementation includes the pipeline for creating handwriting (auto-labeled) dataset supplementary to IAM dataset, and the handwriting generator model with LSTM based feature extraction module.

**Final Project** for _DDA4210 Advanced Machine Learning (2024 Spring)_ in CUHKSZ.

**Members of Group 7**:

- Lu, Li (121090272)

- Yiqu, Yang (121090711)

- Jingxuan, Wu (121090607)

- Caijun, Wang (121090532)

----

| Content                             | URL                                                          | Implementation |
| ----------------------------------- | ------------------------------------------------------------ | -------------- |
| Handwriting dataset (supplementary) | -                                                            | This repo      |
| Handwriting dataset                 | http://www.fki.inf.unibe.ch/databases/iam-handwriting-database | IAM            |
| Writer Identifier                   | -                                                            | This repo      |
| Style Encoder                       | -                                                            | This repo      |
| OCR module                          | https://dl.acm.org/doi/10.1145/3550070                       | HiGAN+         |
| GAN                                 | https://dl.acm.org/doi/10.1145/3550070                       | HiGAN+         |

### Recipe

#### Environment

We trained and tested our model on `Python 3.7` and `PyTorch 1.11.0`.

#### Our Dataset Collection

Please refer to two directories `data` and `data_collector` for more details.

#### Train

Please First edit the configure file stored in the `configs` directory. You need to specify the dataset path and pre-trained checkpoint path.

Then, run the command

```
python train.py --config ./configs/config-name.yml
```

#### Inference

Using style mode: (Handwriting style transfer)

Store the images of your own handwriting in the `data/image_samples` folder. Rename the file as `[text_of_the_handwriting].png` (not `jpg`).

Then, run the command

```
python eval_demo.py --config ./configs/gan_image.yml --ckpt ./pretrained/ckpt.pth --mode style
```

Change the newly trained model after `--ckpt` to the directory of the new `.pth` file.
