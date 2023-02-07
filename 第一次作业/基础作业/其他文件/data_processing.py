import os
import shutil
import zipfile

def make_zip(source_dir, output_filename):
    zipf = zipfile.ZipFile(output_filename, 'w')
    pre_len = len(os.path.dirname(source_dir))
    for parent, _, filenames in os.walk(source_dir):
        for filename in filenames:
            pathfile = os.path.join(parent, filename)
            arcname = pathfile[pre_len:].strip(os.path.sep)     # 相对路径
            zipf.write(pathfile, arcname)
    zipf.close()

names = ["daisy", "rose", "sunflower", "tulip", "dandelion"]     # 在训练和验证文件夹中创建5个类别的子文件夹
for name in names:
    path1 = "flower_dataset\\train\\"+name
    if not os.path.exists(path1):
        os.makedirs(path1)
    path2 = "flower_dataset\\val\\" + name
    if not os.path.exists(path2):
        os.makedirs(path2)

if not os.path.exists("flower"):
    os.mkdir("flower")           # 将原始子文件夹移入新建文件夹flower中
for name in names:
    if not os.path.exists("flower\\"+name):
        shutil.move("flower_dataset\\" + name, "flower")

dirs = []               # 保存五种花的每张图像的名字
for name in names:
    path = "flower\\" + name
    if os.path.exists(path):
        for root, dir, file in os.walk(path):
            dirs.append(file)

count = 0            # 将不同种类的花以8：2的比例划分为训练集与验证集，放入两个文件夹中
f = open("flower_dataset\\classes.txt", mode="w")
fw1 = open("flower_dataset\\train.txt", mode="w")
fw2 = open("flower_dataset\\val.txt", mode="w")
for name in names:
    path1 = "flower_dataset/train/" + name
    path2 = "flower_dataset/val/" + name
    file_path = "flower\\" + name
    file = dirs[count]
    for i in range(len(file)):
        if i <= int(0.8 * len(file)):
            shutil.copy(file_path + "\\" + file[i], path1)
            fw1.write("/data/home/scv9034/run/mmclassification/data/flower/"+path1+"/"+file[i]+" "+str(count)+"\n")          # 向txt文件中写入训练集图像名称及标签
        else:
            shutil.copy(file_path + "\\" + file[i], path2)
            fw2.write("/data/home/scv9034/run/mmclassification/data/flower/"+path2 + "/" + file[i] + " " + str(count) + "\n")         # 向txt文件中写入测试集图像名称及标签
    count += 1
    f.write(name + "\n")
f.close()
fw1.close()
fw2.close()

make_zip("flower_dataset", "flower.zip")         # 压缩为zip格式压缩包


