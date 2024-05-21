from collections import Counter
import cv2
import numpy as np
import itertools
import os
import h5py


def preprocess_img(img_pth: str, text_len: int, target_height: int = 64, if_scale: bool = False):
    """
    Preprocess the image, including cutting the egde of the image, and resize the image to have fixed height and the original ratio
    Input:
        img_pth: str, the path of the image
        text_len: int, the length of the text
        target_height: int, the target height of the image
        if_scale: bool, whether to scale the image (if false, then keep the original ratio)
    Output:
        img: np.array, the preprocessed image
    """

    img = cv2.imread(img_pth, cv2.IMREAD_GRAYSCALE)

    # increase contrast of the image
    factor = 3
    ctr_img = factor * (img - 127.5) + 127.5
    ctr_img = np.clip(ctr_img, 0, 255).astype(np.uint8)

    # find the index for the n//2 largest value for the adjusted image
    h, w = img.shape
    n = h * w
    index = n - n//3
    threshold = np.partition(ctr_img.flatten(), index, axis=None)[index]

    # find the index for the n//2 smallest value for the original image
    index = n - n//3
    pad_v = np.partition(img.flatten(), index, axis=None)[index]

    # cut the image according to the threshold
    col_min = ctr_img.min(axis=0)
    row_min = ctr_img.min(axis=1)
    left_index = np.argmax(col_min < threshold)
    right_index = w - np.argmax(col_min[::-1] < threshold)
    top_index = np.argmax(row_min < threshold)
    bottom_index = h - np.argmax(row_min[::-1] < threshold)

    img = img[top_index:bottom_index, left_index:right_index]

    target_width = int(text_len * target_height/2)

    if if_scale:
        img = cv2.resize(img, (target_width, target_height))
    else:  # not scale, but pad the width or height
        # resize the image first
        orig_ratio = img.shape[0] / img.shape[1]  # h/w
        new_ratio = target_height / target_width

        # old_width = img.shape[1]
        if orig_ratio > new_ratio:  # pad the width
            resized_width = round(target_height / orig_ratio)
            img = cv2.resize(img, (resized_width, target_height))

            # calculate the padding width
            width_gap = target_width - resized_width

            pad_left = width_gap // 2
            pad_right = width_gap - pad_left

            # pad the image
            img = np.pad(img, ((0, 0), (pad_left, pad_right)),
                         mode='constant', constant_values=pad_v)

        else:  # pad the height
            resized_height = round(target_width * orig_ratio)
            img = cv2.resize(img, (target_width, resized_height))

            # calculate the padding height
            height_gap = target_height - resized_height
            pad_top = height_gap // 2
            pad_bottom = height_gap - pad_top

            # pad the image
            img = np.pad(img, ((pad_top, pad_bottom), (0, 0)),
                         mode='constant', constant_values=pad_v)

        # reverse the color
        img = 255 - img
    return img


def gen_h5file(all_imgs, all_texts, all_wids, save_path):
    img_seek_idxs, img_lens = [], []
    cur_seek_idx = 0
    for img in all_imgs:
        img_seek_idxs.append(cur_seek_idx)
        img_lens.append(img.shape[-1])
        cur_seek_idx += img.shape[-1]

    lb_seek_idxs, lb_lens = [], []
    cur_seek_idx = 0
    for lb in all_texts:
        lb_seek_idxs.append(cur_seek_idx)
        lb_lens.append(len(lb))
        cur_seek_idx += len(lb)

    save_imgs = np.concatenate(all_imgs, axis=-1)
    save_texts = list(itertools.chain(*all_texts))
    save_lbs = [ord(ch) for ch in save_texts]
    h5f = h5py.File(save_path, 'w')
    h5f.create_dataset('imgs',
                       data=save_imgs,
                       compression='gzip',
                       compression_opts=4,
                       dtype=np.uint8)
    h5f.create_dataset('lbs',
                       data=save_lbs,
                       dtype=np.int32)
    h5f.create_dataset('img_seek_idxs',
                       data=img_seek_idxs,
                       dtype=np.int64)
    h5f.create_dataset('img_lens',
                       data=img_lens,
                       dtype=np.int16)
    h5f.create_dataset('lb_seek_idxs',
                       data=lb_seek_idxs,
                       dtype=np.int64)
    h5f.create_dataset('lb_lens',
                       data=lb_lens,
                       dtype=np.int16)
    h5f.create_dataset('wids',
                       data=all_wids,
                       dtype=np.int16)
    h5f.close()
    print('save->', save_path)


H = 64  # fixed height for target images
data_dir = 'data'
new_dataset_dir = os.path.join(data_dir, 'new_dataset')
word_img_dir = os.path.join(new_dataset_dir, 'word_img')
processed_img_dir = os.path.join(new_dataset_dir, 'processed_img')

all_imgs = []  # store word images
all_texts = []  # store corresponding texts
all_wids = []  # store writer indices

writer_idx = 0
for writer_name in os.listdir(word_img_dir):
    if not os.path.isdir(os.path.join(word_img_dir, writer_name)):
        continue

    writer_dir = os.path.join(word_img_dir, writer_name)
    img_save_dir = os.path.join(processed_img_dir, writer_name)
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)

    for img_name in os.listdir(writer_dir):
        if not img_name.endswith('.png'):
            continue

        img_path = os.path.join(writer_dir, img_name)

        text = img_name.split('.')[0]
        img = preprocess_img(img_path, len(text), target_height=H)
        img_save_pth = os.path.join(img_save_dir, img_name)
        cv2.imwrite(img_save_pth, img)

        all_imgs.append(img)
        all_texts.append(text)
        all_wids.append(writer_idx)

    writer_idx += 1


# if we split the dataset by writer
# split training and testing set
train_ratio = 0.75   # 75% for training

n_wids = len(set(all_wids))
n_train_wids = int(n_wids * train_ratio)
split_idx = all_wids.index(n_train_wids) + all_wids.count(n_train_wids)
print('split_idx:', split_idx)
print(all_wids[split_idx-10: split_idx])
print(all_wids[split_idx: split_idx+10])

train_imgs, train_texts, train_wids = all_imgs[:
                                               split_idx], all_texts[:split_idx], all_wids[:split_idx]
test_imgs, test_texts, test_wids = all_imgs[split_idx:
                                            ], all_texts[split_idx:], all_wids[split_idx:]
print(train_wids[-10:])
print(test_wids[:10])

# if for each writer, we split the dataset into training and testing set
# some bugs: wids for testset should start with the number of writers in the training set!!!!!!!
# # dictionary with writer index as key and count as value
# counter = Counter(all_wids)
# train_count = {writer: int(count * train_ratio)
#                for writer, count in counter.items()}
# now_train_count = {writer: 0 for writer in counter.keys()}

# train_imgs, train_texts, train_wids = [], [], []
# test_imgs, test_texts, test_wids = [], [], []

# for img, text, wid in zip(all_imgs, all_texts, all_wids):
#     if now_train_count[wid] < train_count[wid]:
#         train_imgs.append(img)
#         train_texts.append(text)
#         train_wids.append(wid)
#         now_train_count[wid] += 1
#     else:
#         test_imgs.append(img)
#         test_texts.append(text)
#         test_wids.append(wid)


train_save_path = os.path.join(
    new_dataset_dir, 'trnvalset_words'+str(H)+'_newdataset.hdf5')
test_save_path = os.path.join(
    new_dataset_dir, 'testset_words'+str(H)+'_newdataset.hdf5')

# generate hdf5 file for training and testing set for the newdataset
gen_h5file(train_imgs, train_texts, train_wids, train_save_path)
gen_h5file(test_imgs, test_texts, test_wids, test_save_path)
