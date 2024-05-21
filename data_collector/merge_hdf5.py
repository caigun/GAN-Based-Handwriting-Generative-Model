import os
import h5py
import numpy as np


def merge_hdf5(h5_pth1, h5_pth2, merged_pth, increment_wid=0):
    with h5py.File(h5_pth1, 'r') as f1, h5py.File(h5_pth2, 'r') as f2, h5py.File(merged_pth, 'w') as merged_f:
        # Merge images
        imgs1 = f1['imgs'][:]
        imgs2 = f2['imgs'][:]
        merged_imgs = np.concatenate((imgs1, imgs2), axis=1)
        merged_f.create_dataset('imgs', data=merged_imgs, dtype=np.uint8)

        # Merge labels
        lbs1 = f1['lbs'][:]
        lbs2 = f2['lbs'][:]
        merged_lbs = np.concatenate((lbs1, lbs2))
        merged_f.create_dataset('lbs', data=merged_lbs, dtype=np.int32)

        # Update indices and lengths
        for key in ['img_seek_idxs', 'img_lens', 'lb_seek_idxs', 'lb_lens']:
            data1 = f1[key][:]
            data2 = f2[key][:]
            if key == 'img_seek_idxs':
                last_idx = data1[-1] + f1['img_lens'][-1]
                data2 += last_idx
            elif key == 'lb_seek_idxs':
                last_idx = data1[-1] + f1['lb_lens'][-1]
                data2 += last_idx
            merged_data = np.concatenate((data1, data2))
            merged_f.create_dataset(key, data=merged_data, dtype=data1.dtype)

        # Merge writer indices
        wids1 = f1['wids'][:]
        print("wids1: ", wids1)
        wids2 = f2['wids'][:]
        decrease_wid = wids2[0]
        wids2 = [wid - decrease_wid for wid in wids2]
        print("wids2: ", wids2)
        last_wid = wids1[-1]
        wids2 += last_wid + 1
        print('wids1', wids1)
        print('wids2', wids2)
        
        wids1 += increment_wid
        wids2 += increment_wid
        print('wids1', wids1)
        print('wids2', wids2)

        merged_wids = np.concatenate((wids1, wids2))
        merged_f.create_dataset('wids', data=merged_wids, dtype=np.int16)

        # Print sizes of the original and merged datasets
        print("For imgs:")
        print("Size of original dataset 1:", imgs1.shape)
        print("Size of original dataset 2:", imgs2.shape)
        print("Size of merged dataset:", merged_imgs.shape)
        print("For lbls:")
        print("Size of original dataset 1:", lbs1.shape)
        print("Size of original dataset 2:", lbs2.shape)
        print("Size of merged dataset:", merged_lbs.shape)
        print("For img_seek_idxs:")
        print("Size of original dataset 1:", f1['img_seek_idxs'].shape)
        print("Size of original dataset 2:", f2['img_seek_idxs'].shape)
        print("Size of merged dataset:", merged_f['img_seek_idxs'].shape)
        print("For lb_seek_idxs:")
        print("Size of original dataset 1:", f1['lb_seek_idxs'].shape)
        print("Size of original dataset 2:", f2['lb_seek_idxs'].shape)
        print("Size of merged dataset:", merged_f['lb_seek_idxs'].shape)
        print("For wids:")
        print("Size of original dataset 1:", wids1.shape)
        print("Size of original dataset 2:", wids2.shape)
        print("Size of merged dataset:", merged_wids)
        print("For lb_lens:")
        print("Size of original dataset 1:", f1['lb_lens'].shape)
        print("Size of original dataset 2:", f2['lb_lens'].shape)
        print("Size of merged dataset:", merged_f['lb_lens'].shape)
        print("number of writers: ")
        print("Size of original dataset 1:", len(np.unique(wids1)))
        print("Size of original dataset 2:", len(np.unique(wids2)))
        print("Unique wids merged dataset:", len(np.unique(merged_wids)))
        print("index of last writer in original dataset 1: ", wids1[-1])


data_dir = "data"
iam_dir = os.path.join(data_dir, "iam")
new_dataset_dir = os.path.join(data_dir, "new_dataset")
merged_dataset_dir = os.path.join(data_dir, "merged_dataset")
if not os.path.exists(merged_dataset_dir):
    os.makedirs(merged_dataset_dir)

h5_name1 = "trnvalset_words64_OrgSz.hdf5"
h5_name2 = "trnvalset_words64_newdataset.hdf5"
merged_name = "trnvalset_words64_merged.hdf5"
h5_pth1 = os.path.join(iam_dir, h5_name1)
h5_pth2 = os.path.join(new_dataset_dir, h5_name2)
merged_h5_pth = os.path.join(merged_dataset_dir, merged_name)
merge_hdf5(h5_pth1, h5_pth2, merged_h5_pth)
print(f"Merged data from {h5_pth1} and {h5_pth2} into {merged_h5_pth}")

h5_name1 = "testset_words64_OrgSz.hdf5"
h5_name2 = "testset_words64_newdataset.hdf5"
merged_name = "testset_words64_merged.hdf5"
h5_pth1 = os.path.join(iam_dir, h5_name1)
h5_pth2 = os.path.join(new_dataset_dir, h5_name2)
merged_h5_pth = os.path.join(merged_dataset_dir, merged_name)
merge_hdf5(h5_pth1, h5_pth2, merged_h5_pth, increment_wid=289)
print(f"Merged data from {h5_pth1} and {h5_pth2} into {merged_h5_pth}")
