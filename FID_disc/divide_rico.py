import os
from shutil import copyfile
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser('divide_rico')
    parser.add_argument("--rico_dir", default="../LayoutAction/datasets/rico/semantic_annotations", help="experiment name")
    parser.add_argument("--out_dir", default="./data/rico", help="/path/to/data/dir")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    train_dir = os.path.join(args.out_dir, "train")
    val_dir = os.path.join(args.out_dir, "val")
    test_dir = os.path.join(args.out_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    dirs = os.listdir(args.rico_dir)

    file_num = len(dirs)
    split = [0.85, 0.9]
    train_num = file_num*split[0]
    val_num = file_num*(split[1]-split[0])
    test_num = file_num*(1-split[1])
    json_img_num = 0

    for file in dirs:
        if file.split(".")[-1] == "json":
            src = os.path.join(args.rico_dir, file)
            if json_img_num < train_num:
                dst = os.path.join(train_dir, file)
            elif json_img_num < (train_num + val_num):
                dst = os.path.join(val_dir, file)
            else:
                dst = os.path.join(test_dir, file)
            
            copyfile(src, dst)
        json_img_num += 1

    train_file_num = len(os.listdir(train_dir))
    val_file_num = len(os.listdir(val_dir))
    test_file_num = len(os.listdir(test_dir))
    print("all num: ", file_num)
    print("train file num: ", train_file_num)
    print("val file num: ", val_file_num)
    print("test file num: ", test_file_num)
        