from glob import glob
import os
import tarfile

# from numpy import full, record
from tqdm import tqdm


# 遍历文件夹下的所有文件

# def list_tar_file(path_root):
#     file_list = list()

#     for root,ds,fs in os.walk(path_root):
#         for f in fs:
#             if f.endswith('.tar'):
#                 fullname_tar = os.path.join(root,f)
#                 fullname = fullname_tar.split('.')[0]
#                 file_list.append(fullname)

#     return file_list

#解压文件

def extar_flie(file_path):
    file = os.path.basename(file_path)
    # save_file = os.path.split(file)[0]
    save_file = file.split(".")[0]
    print("starting extart {}".format(save_file))
    tar = tarfile.open(file_path)

    names = tar.getnames()

    for name in tqdm(names):
        new_save = "/".join(os.path.dirname(file_path).split("/")[:-1])
        tar.extract(name,new_save+"/train/"+save_file)

    tar.close


if __name__ == "__main__":
    floder = "/media/yang/games/datasat/imagnet/ILSVRC2012_img_train_t3/*.tar"

    record = open("record.txt","w")

    for i in glob(floder):
        try:
            print("file name:",i)
            extar_flie(i)
        
        except:
            record.write(str(i)+"\n")

            continue








