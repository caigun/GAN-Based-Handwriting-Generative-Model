import os


data_dir = 'data'
new_dataset_dir = os.path.join(data_dir, 'new_dataset')

# create directory "page_image" in "new_dataset_dir" if not exists
if not os.path.exists(os.path.join(new_dataset_dir, 'page_img')):
    os.makedirs(os.path.join(new_dataset_dir, 'page_img'))

# combine all images in the folders with name beginning with "_page_img" into one folder named "page_img"
page_img_dir = os.path.join(new_dataset_dir, 'page_img')
last_writer_idx = 0
for folder in os.listdir(new_dataset_dir):
    max_count = 0  # max count of images in a folder
    if folder.startswith("_page_img"):
        folder_path = os.path.join(new_dataset_dir, folder)
        for file1 in os.listdir(folder_path):
            writer_idx1, after = file1.split("_")
            if int(writer_idx1) > max_count:
                max_count = int(writer_idx1) + 1
            writer_idx2 = int(writer_idx1) + last_writer_idx
            file2 = str(writer_idx2) + "_" + after
            file_path = os.path.join(folder_path, file1)
            new_file_path = os.path.join(page_img_dir, file2)
            os.rename(file_path, new_file_path)
        os.rmdir(folder_path)
        print("max_count: ", max_count)
        last_writer_idx += max_count
    print("last_writer_idx: ", last_writer_idx)
