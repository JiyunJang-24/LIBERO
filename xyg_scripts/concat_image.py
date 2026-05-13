import os
from PIL import Image
from glob import glob
import numpy as np
import matplotlib.pyplot as plt


def concat_image(image_path_list, output_path):
    if 'rotate' in output_path:
        # basename 为 rotate_{angle:.2f}.png: 提取中间的 float 数值， 并转换为 float
        title_list = [float(os.path.basename(image_path).split('_')[1][:-4]) for image_path in image_path_list]
        # 将 image_path_list 按照 title_list 大小排序
        image_path_list = [x for _, x in sorted(zip(title_list, image_path_list), key=lambda pair: pair[0])]
    else:
        # basename 为 light_{alpha:.2f}.png: 提取中间的 float 数值， 并转换为 float
        title_list = [float(os.path.basename(image_path).split('_')[1][:-4]) for image_path in image_path_list]
        # 将 image_path_list 按照 title_list 大小排序
        image_path_list = [x for _, x in sorted(zip(title_list, image_path_list), key=lambda pair: pair[0])]
    
    images = [Image.open(image_path) for image_path in image_path_list]
    
    # 使用 matplotlib 显示图片
    plt.figure(figsize=(5 * len(images), 5))
    for i, image in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(image)
        plt.axis('off')
        
        title = float(os.path.basename(image_path_list[i]).split('_')[1][:-4])
        plt.title(f"{title:.2f}")
    
    plt.savefig(output_path)
    plt.close()
    
    
def main():
    image_light_path_list = glob(os.path.join('./LIBERO/xyg_scripts/image_light_example', "light_*.png"))
    image_rotation_path_list = glob(os.path.join('./LIBERO/xyg_scripts/image_rotation_example', "rotate_*.png"))
    
    concat_image(image_light_path_list, "./LIBERO/xyg_scripts/image_light_example/all_concat.png")
    concat_image(image_rotation_path_list, "./LIBERO/xyg_scripts/image_rotation_example/all_concat.png")


if __name__ == "__main__":
    main()