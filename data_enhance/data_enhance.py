import os
import random
from PIL import Image, ImageFilter
from glob import glob


class ImageEnhancer:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = Image.open(image_path)

    def flip(self, direction='horizontal'):
        """翻转图像，支持水平和垂直翻转"""
        if direction == 'horizontal':
            self.image = self.image.transpose(Image.FLIP_LEFT_RIGHT)
        elif direction == 'vertical':
            self.image = self.image.transpose(Image.FLIP_TOP_BOTTOM)
        return self

    def crop(self, crop_ratio=0.8):
        """
        裁剪图像，crop_ratio 表示裁剪区域占原图的比例
        例如：crop_ratio=0.8，则保留80%的区域，取图中央区域
        """
        width, height = self.image.size
        new_width, new_height = int(width * crop_ratio), int(height * crop_ratio)
        left = (width - new_width) // 2
        upper = (height - new_height) // 2
        right = left + new_width
        lower = upper + new_height
        self.image = self.image.crop((left, upper, right, lower))
        return self

    def blur(self, radius=2):
        """模糊图像，radius 控制模糊程度"""
        self.image = self.image.filter(ImageFilter.GaussianBlur(radius))
        return self

    def save(self, save_path):
        """
        覆盖保存当前图像到指定路径
        """
        # 创建目录如果不存在，但覆盖保存时通常目录已经存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.image.save(save_path)


def augment_images(dataset_dir, subset, num_samples=100):
    """
    对 dataset_dir 下的某个子集（train, valid, test）进行图像增强
    说明：原始图像在 images 文件夹下，增强图像将直接覆盖原图
    """
    images_dir = os.path.join(dataset_dir, subset, "images")

    # 寻找常见图像格式的文件
    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        image_paths.extend(glob(os.path.join(images_dir, ext)))

    if not image_paths:
        print(f"[{subset}] 没有找到图像文件")
        return

    # 随机挑选 num_samples 张图像进行增强（如果图像数量不够，则全部处理）
    sample_paths = random.sample(image_paths, min(num_samples, len(image_paths)))
    print(f"#########################################开始进行增强处理############################################")

    for img_path in sample_paths:
        try:
            # 直接覆盖原图，因此保存路径与原图路径相同
            save_path = img_path

            # 建立增强处理流水线：先翻转（随机选择水平或垂直）、再裁剪、再模糊
            enhancer = ImageEnhancer(img_path)
            flip_direction = random.choice(['horizontal', 'vertical'])
            # 可根据需要调整各种操作的参数
            enhancer.flip(direction=flip_direction).crop(crop_ratio=0.8).blur(radius=2)
            enhancer.save(save_path)
            print(f"[{subset}] 增强并保存了：{save_path}")
        except Exception as e:
            print(f"[{subset}] 处理 {img_path} 时出错：{e}")
    print(f"#########################################图像增强处理完成############################################")


def main():
    # 数据集文件夹根目录
    dataset_dir = "../data"

    # 如果只对训练集增强，可修改 subsets 列表；如果需要其它集（如 valid、test），也可加入
    subsets = ["train"]  # 如需增强 valid 和 test, 则可改为: ["train", "valid", "test"]
    num_samples = 3000
    # 可修改
    for subset in subsets:
        augment_images(dataset_dir, subset, num_samples=num_samples)


if __name__ == "__main__":
    main()





