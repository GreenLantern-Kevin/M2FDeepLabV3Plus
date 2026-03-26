from PIL import Image
import numpy as np

img = Image.open('BijieOnlyLandslide/VOC2012/SegmentationClass/df008.png')
img_array = np.array(img)
print(f"图像模式: {img.mode}")
print(f"包含的像素值: {np.unique(img_array)}")

# import os
# import numpy as np
# from PIL import Image
# from tqdm import tqdm  # 建议安装，可以看到处理进度


# def convert_to_color_mask(input_dir, output_dir):
#     # 1. 如果输出文件夹不存在，则创建
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#         print(f"Created directory: {output_dir}")

#     # 2. 定义调色板 (P模式需要 768 个值: 256种颜色 * RGB)
#     # 你可以根据需要增加更多类别的颜色
#     palette = [
#         0, 0, 0,  # index 0: 黑色 (背景)
#         128, 0, 0,  # index 1: 红色 (目标/滑坡)
#         0, 128, 0,  # index 2: 绿色
#         128, 128, 0,  # index 3: 黄色
#         0, 0, 128,  # index 4: 蓝色
#     ]
#     # 补齐至 768 个元素
#     palette += [0] * (768 - len(palette))

#     # 3. 遍历处理所有图片
#     files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
#     print(f"Starting conversion: {len(files)} files found.")

#     for name in tqdm(files):
#         # 读取原始图片并转为灰度
#         img_path = os.path.join(input_dir, name)
#         # 使用 PIL 读取可以更好地配合 P 模式转换
#         img = Image.open(img_path).convert('L')
#         img_array = np.array(img)

#         # 二值化处理：将 255 (或大于127的值) 转为 1，其余转为 0
#         # 如果你有多个类别，可以根据像素范围多次赋值
#         new_mask_array = np.zeros_like(img_array)
#         new_mask_array[img_array > 127] = 1

#         # 将 numpy 数组转回 PIL 图像，并设为 'P' (Palette) 模式
#         res = Image.fromarray(new_mask_array.astype(np.uint8)).convert('P')

#         # 注入调色板
#         res.putpalette(palette)

#         # 保存图片
#         res.save(os.path.join(output_dir, name))

#     print("\n所有标注文件已处理完成！")


# # --- 请在此处修改你的路径 ---
# origin_mask_path = 'Bijie-landslide-dataset/landslide/mask'  # 例如: './old_masks'
# save_mask_path = 'BijieOnlyLandslide/VOC2012/SegmentationClass'  # 例如: './voc_style_masks'

# if __name__ == '__main__':
#     convert_to_color_mask(origin_mask_path, save_mask_path)