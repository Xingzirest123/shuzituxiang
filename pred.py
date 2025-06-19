import numpy as np
from ultralytics import YOLO
import cv2
import math

# 设置字体样式
font = cv2.FONT_HERSHEY_DUPLEX


# 在图像上添加带背景的文本
def add_text_with_background(
    image,
    text,
    position,
    font_face,
    font_scale,
    text_color,
    bg_color,
    thickness=1,
    padding=5,
):
    # 获取文本大小
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font_face, font_scale, thickness
    )

    # 计算文本框的大小（包括内边距）
    box_width = text_width + 2 * padding
    box_height = text_height + 2 * padding + baseline

    # 确保文本框不会超出图像边界
    x, y = position
    x = max(0, min(x, image.shape[1] - box_width))
    y = max(box_height, min(y, image.shape[0]))

    # 绘制背景矩形
    cv2.rectangle(image, (x, y - box_height), (x + box_width, y), bg_color, -1)

    # 绘制文本
    text_position = (x + padding, y - padding - baseline)
    cv2.putText(
        image, text, text_position, font_face, font_scale, text_color, thickness
    )


# 获取适合文本宽度的最优字体大小
def get_optimal_font_scale(text, width):
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(
            text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale / 10, thickness=2
        )
        new_width = textSize[0][0]
        if new_width <= width:
            return scale / 10
    return 1


# 加载YOLO模型
model = YOLO("./runs/detect/yolov115/weights/best.pt")

# 定义类别名称
classNames = ['Car', 'Car', 'Car', 'Car', 'Car']


# 进行预测
def pred(img, stream=False):
    orig = img.copy()

    # 使用YOLO模型进行预测
    results = model(img, stream=stream)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # 获取边界框的坐标
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # 获取置信度并进行四舍五入
            conf = math.ceil((box.conf[0] * 100)) / 100

            # 获取类别索引并转换为类别名称
            cls = box.cls[0]
            name = classNames[int(cls)]
            print(f"{name} " f"{conf}")

            # 获取适合文本的最优字体大小
            #font_scale = get_optimal_font_scale(f"{name} {conf}", w)
            thick = 1 if w < 210 else 2

            # 根据类别名称选择颜色并绘制边界框和文本
            cv2.rectangle(
                img=img,
                pt1=(x1, y1),
                pt2=(x1 + w, y1 + h),
                color=(0, 102, 255), #bgr
                thickness=2,
            )
            add_text_with_background(
                img,
                f"{name} {conf}",
                (x1, y1),
                font,
                1.1,
                (255, 255, 255),
                (0, 102, 255), #bgr
                thick,
                5,
            )
    return orig, img


# 主程序入口
if __name__ == "__main__":
    pred("./test_images/2.jpg")





