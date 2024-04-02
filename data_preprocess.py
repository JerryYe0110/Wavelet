import os
import shutil

def Read(Dir, name):
    folders = os.listdir(Dir)
    os.mkdir(name)
    target_gt = (os.path.join(name, "target"))
    target_input = (os.path.join(name, "input"))
    os.mkdir(target_input)
    os.mkdir(target_gt)
    index = 0
    n = 1
    for folder in folders:
        print("Start", folder, n, "/", len(folders))
        folder_path = os.path.join(Dir, folder)
        img = os.listdir(folder_path)
        gt = ""
        for i in img:
            if "-C-" in i:
                gt = i
                del i
                break
        gt = os.path.join(folder_path, gt)
        for i in img:
            os.system('cp {} {}'.format(os.path.join(folder_path, i), os.path.join(target_input, str(index)+".png")))
            os.system('cp {} {}'.format(gt, os.path.join(target_gt, str(index)+".png")))
            index = index + 1
        n = n + 1


#shutil.unpack_archive('./GT-RAIN_val.zip', './GT-RAIN_val')
#shutil.unpack_archive('./GT-RAIN_train.zip', './GT-RAIN_train')
#os.remove('./GT-RAIN_train.zip')
#os.remove('./GT-RAIN_val.zip')
#
#os.mkdir("GT-RAIN")
#shutil.move("./GT-RAIN_val", "./GT-RAIN/GT-RAIN_val")
#shutil.move("./GT-RAIN_train", "./GT-RAIN/GT-RAIN_train")
Read("./GT-RAIN/GT-RAIN_val", "./GT-RAIN/val")
print("Val finish")
Read("./GT-RAIN/GT-RAIN_train", "./GT-RAIN/train")
print("Train finish")
shutil.rmtree('./GT-RAIN/GT-RAIN_train')
shutil.rmtree('./GT-RAIN/GT-RAIN_val')

