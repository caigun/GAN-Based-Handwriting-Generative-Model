import h5py
import matplotlib.pylab as plt
import numpy


def summary_hdf5(hdf5_pth, if_img_show=False):
    with h5py.File(hdf5_pth, 'r') as file_object:
        print('*'*50)
        print('Reading the hdf5 file: ', hdf5_pth)

        print(file_object.keys())
        img_data = file_object["imgs"]
        img_lens = file_object["img_lens"]
        wids_set = set(file_object['wids'])
        print(wids_set)
        print(min(wids_set), '~', max(wids_set))
        print('number of writers: ', len(wids_set))
        print('number of images: ', len(img_lens))

        print('img_data shape: ', img_data.shape)
        print('img_lens shape: ', img_lens.shape)
        img_pos = file_object['img_seek_idxs']
        print(min(img_pos), max(img_pos), len(img_pos))
        lb_pos = file_object['lb_seek_idxs']
        print(min(lb_pos), max(lb_pos), len(lb_pos))
        print(min(file_object['wids']), max(file_object['wids']))

        if if_img_show:

            sliced_img_data = []  # store the data of each image after slicing
            start_idx = 0
            count_number = 0
            for img_len in img_lens:
                # only loop through the first 2 images
                if count_number >= 2:
                    break
                else:
                    sliced_img_data.append(
                        img_data[:, start_idx:start_idx+img_len])
                    start_idx += img_len
                    count_number += 1

            i = 0
            for image_data in sliced_img_data:
                i += 1
                numpy.savetxt("data/iam/"+str(i)+".txt", image_data, fmt='%d')
                plt.imshow(image_data)
                plt.show()
                plt.imshow(image_data, cmap='gray')
                # write the image_data to a txt file
                plt.show()


mgd_trnval_hdf5_pth = 'data/merged_dataset/trnvalset_words64_merged.hdf5'
mgd_test_hdf5_pth = 'data/merged_dataset/testset_words64_merged.hdf5'
iam_trnval_hdf5_pth = 'data/iam/trnvalset_words64_OrgSz.hdf5'
iam_test_hdf5_pth = 'data/iam/testset_words64_OrgSz.hdf5'
new_trnval_hdf5_pth = 'data/new_dataset/trnvalset_words64_newdataset.hdf5'
new_test_hdf5_pth = 'data/new_dataset/testset_words64_newdataset.hdf5'

print('for merged dataset:', '#'*50)
summary_hdf5(mgd_trnval_hdf5_pth, if_img_show=False)
summary_hdf5(mgd_test_hdf5_pth, if_img_show=False)

print()
print('for iam dataset:', '#'*50)
summary_hdf5(iam_trnval_hdf5_pth, if_img_show=False)
summary_hdf5(iam_test_hdf5_pth, if_img_show=False)

print()
print('for new dataset:', '#'*50)
summary_hdf5(new_trnval_hdf5_pth, if_img_show=False)
summary_hdf5(new_test_hdf5_pth, if_img_show=False)
