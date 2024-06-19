import os
import shutil

root_dir = "E:\\Random Python Scripts\\FarmLand-Segmentation-main\\New Approach\\archive\\train"

mask_dir = os.path.join(root_dir, 'mask')
image_dir = os.path.join(root_dir, 'images')

os.makedirs(mask_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

for filename in os.listdir(root_dir):
    if filename.endswith('.png'):
        src_path = os.path.join(root_dir, filename)
        dst_path = os.path.join(mask_dir, filename)
        shutil.copy2(src_path, dst_path)
    elif filename.endswith('.jpg'):
        src_path = os.path.join(root_dir, filename)
        dst_path = os.path.join(image_dir, filename)
        shutil.copy2(src_path, dst_path)
