import os

folder = '/disk1/panzhiyu/THUPALMLAB/mask_match/'
for file in os.listdir(folder):
    # rename png to bmp
    if file.endswith('.png'):
        os.rename(os.path.join(folder, file), os.path.join(folder, file[:-3] + 'bmp'))