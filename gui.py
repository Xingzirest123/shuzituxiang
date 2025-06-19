import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk, ImageEnhance, ImageOps
import cv2
import numpy as np
import os

from pred import pred

# 颜色主题
BG_COLOR = "#f0f0f0"
BUTTON_COLOR = "#4a7a8c"
BUTTON_HOVER = "#3a6a7c"
TEXT_COLOR = "#333333"
FRAME_COLOR = "#e0e0e0"

# 显示设置
MAX_WIDTH = 1000
MAX_HEIGHT = 1000
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB


class StyledButton(tk.Button):
    """自定义样式按钮"""

    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.default_bg = BUTTON_COLOR
        self.configure(
            bg=self.default_bg,
            fg='white',
            activebackground=BUTTON_HOVER,
            activeforeground='white',
            relief='raised',
            borderwidth=2,
            font=('Arial', 10, 'bold'),
            cursor='hand2'
        )
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, e):
        self['bg'] = BUTTON_HOVER

    def on_leave(self, e):
        self['bg'] = self.default_bg


class ImageProcessor:
    """图像处理工具类"""

    @staticmethod
    def sharpen_image(image, factor=2.0):
        """锐化图像"""
        pil_img = Image.fromarray(image)
        enhancer = ImageEnhance.Sharpness(pil_img)
        sharpened = enhancer.enhance(factor)
        return np.array(sharpened)

    @staticmethod
    def rotate_image(image, angle=90):
        """旋转图像"""
        pil_img = Image.fromarray(image)
        rotated = pil_img.rotate(angle, expand=True)
        return np.array(rotated)

    @staticmethod
    def grayscale_image(image):
        """转换为灰度图"""
        if len(image.shape) == 3:  # 彩色图像
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return image

    @staticmethod
    def adjust_brightness(image, factor=1.5):
        """调整亮度"""
        pil_img = Image.fromarray(image)
        enhancer = ImageEnhance.Brightness(pil_img)
        brightened = enhancer.enhance(factor)
        return np.array(brightened)

    @staticmethod
    def adjust_contrast(image, factor=1.5):
        """调整对比度"""
        pil_img = Image.fromarray(image)
        enhancer = ImageEnhance.Contrast(pil_img)
        contrasted = enhancer.enhance(factor)
        return np.array(contrasted)

    @staticmethod
    def flip_image(image, direction='horizontal'):
        """翻转图像"""
        if direction == 'horizontal':
            return cv2.flip(image, 1)
        elif direction == 'vertical':
            return cv2.flip(image, 0)
        return image


class RadarCrackDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("车辆检测跟踪系统")
        self.root.geometry("1600x900")
        self.root.configure(bg=BG_COLOR)

        # 图像处理相关变量
        self.current_image = None
        self.processed_image = None
        self.rotation_angle = 0
        self.sharpen_factor = 1.0
        self.brightness_factor = 1.0
        self.contrast_factor = 1.0
        self.grayscale_enabled = False

        self.setup_ui()

        # 视频相关变量
        self.cap = None
        self.running = False
        self.last_orig = None
        self.last_processed = None

    def setup_ui(self):
        """初始化界面布局"""
        # 标题框架
        title_frame = tk.Frame(self.root, bg=BG_COLOR)
        title_frame.pack(pady=10)

        tk.Label(
            title_frame,
            text="车辆检测跟踪系统",
            font=('Microsoft YaHei', 18, 'bold'),
            bg=BG_COLOR,
            fg=TEXT_COLOR
        ).pack()

        # 控制按钮区
        btn_frame = tk.Frame(self.root, bg=BG_COLOR)
        btn_frame.pack(pady=10)

        self.load_img_btn = StyledButton(btn_frame, text="加载图片", command=self.load_image)
        self.load_video_btn = StyledButton(btn_frame, text="加载视频", command=self.load_video)
        self.start_cam_btn = StyledButton(btn_frame, text="启动摄像头", command=self.start_camera)
        self.stop_cam_btn = StyledButton(btn_frame, text="停止摄像头", command=self.stop_camera, state=tk.DISABLED)

        for btn in [self.load_img_btn, self.load_video_btn, self.start_cam_btn, self.stop_cam_btn]:
            btn.pack(side=tk.LEFT, padx=10, ipadx=15, ipady=5)

        # 图像处理控制区
        process_frame = tk.Frame(self.root, bg=BG_COLOR)
        process_frame.pack(pady=10)

        # 图像处理按钮
        self.sharpen_btn = StyledButton(process_frame, text="锐化",
                                        command=lambda: self.apply_image_processing('sharpen'))
        self.rotate_btn = StyledButton(process_frame, text="旋转", command=lambda: self.apply_image_processing('rotate'))
        self.grayscale_btn = StyledButton(process_frame, text="灰度化",
                                          command=lambda: self.apply_image_processing('grayscale'))
        self.brightness_btn = StyledButton(process_frame, text="亮度+",
                                           command=lambda: self.apply_image_processing('brightness'))
        self.contrast_btn = StyledButton(process_frame, text="对比度+",
                                         command=lambda: self.apply_image_processing('contrast'))
        self.reset_btn = StyledButton(process_frame, text="重置处理", command=self.reset_image_processing)

        for btn in [self.sharpen_btn, self.rotate_btn, self.grayscale_btn,
                    self.brightness_btn, self.contrast_btn, self.reset_btn]:
            btn.pack(side=tk.LEFT, padx=5, ipadx=10, ipady=3)
            btn['state'] = tk.DISABLED  # 初始禁用

        # 分隔线
        ttk.Separator(self.root, orient='horizontal').pack(fill='x', pady=10)

        # 图像显示区
        self.display_frame = tk.Frame(self.root, bg=BG_COLOR)
        self.display_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # 配置双列布局
        self.display_frame.columnconfigure(0, weight=1, minsize=450)
        self.display_frame.columnconfigure(1, weight=1, minsize=450)
        self.display_frame.rowconfigure(0, weight=1)

        # 原始图像面板
        self.orig_panel = self.create_image_panel("原始图像", "请加载图片或视频")
        self.orig_panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.orig_label = self.orig_panel.winfo_children()[1]  # 保存标签引用

        # 结果图像面板
        self.result_panel = self.create_image_panel("检测结果", "预测结果将显示在这里")
        self.result_panel.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.result_label = self.result_panel.winfo_children()[1]  # 保存标签引用

        # 状态栏
        self.status_bar = tk.Label(
            self.root,
            text="就绪",
            bd=1, relief=tk.SUNKEN,
            anchor=tk.W,
            bg=BG_COLOR,
            fg=TEXT_COLOR,
            font=('Microsoft YaHei', 9)
        )
        self.status_bar.pack(fill=tk.X, padx=5, pady=2)

    def create_image_panel(self, title, placeholder):
        """创建统一的图像显示面板"""
        frame = tk.Frame(
            self.display_frame,
            bg=FRAME_COLOR,
            bd=2,
            relief=tk.GROOVE
        )
        # 标题
        tk.Label(
            frame,
            text=title,
            font=('Microsoft YaHei', 12, 'bold'),
            bg=FRAME_COLOR,
            fg=TEXT_COLOR
        ).pack(pady=5)

        # 图像显示标签
        label = tk.Label(
            frame,
            text=placeholder,
            justify="center",
            anchor="center",
            bg=FRAME_COLOR,
            fg=TEXT_COLOR,
            font=('Microsoft YaHei', 10))
        label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        return frame

    def load_image(self):
        """加载图片文件"""
        self.status_bar['text'] = "正在加载图片..."
        self.root.update()

        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[
                ("JPEG 文件", "*.jpg *.jpeg"),
                ("PNG 文件", "*.png"),
                ("位图文件", "*.bmp"),
                ("所有文件", "*.*")
            ]
        )
        if file_path:
            try:
                # 检查文件大小
                if os.path.getsize(file_path) > MAX_FILE_SIZE:
                    raise ValueError("文件大小超过100MB限制")

                # 读取图片
                with open(file_path, 'rb') as f:
                    img_data = np.frombuffer(f.read(), dtype=np.uint8)
                    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                    if img is None:
                        raise ValueError("不支持的图片格式")

                # 保存原始图像
                self.current_image = img.copy()
                self.processed_image = img.copy()

                # 启用图像处理按钮
                for btn in [self.sharpen_btn, self.rotate_btn, self.grayscale_btn,
                            self.brightness_btn, self.contrast_btn, self.reset_btn]:
                    btn['state'] = tk.NORMAL

                # 重置处理状态
                self.reset_image_processing()

                # 处理并显示
                self.process_and_display(img, is_stream=False)
                self.status_bar['text'] = f"已加载图片: {os.path.basename(file_path)}"
            except Exception as e:
                self.status_bar['text'] = f"加载失败: {str(e)}"
                self.clear_display()

    def load_video(self):
        """加载视频文件"""
        self.status_bar['text'] = "正在加载视频..."
        self.root.update()

        file_path = filedialog.askopenfilename(
            title="选择视频",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov")]
        )
        if file_path:
            self.stop_camera()
            try:
                self.cap = cv2.VideoCapture(file_path)
                if not self.cap.isOpened():
                    raise ValueError("无法打开视频文件")

                self.running = True
                self.stop_cam_btn['state'] = tk.NORMAL
                self.status_bar['text'] = f"正在播放: {os.path.basename(file_path)}"
                self.update_video_frame()

                # 禁用图像处理按钮（视频流处理）
                for btn in [self.sharpen_btn, self.rotate_btn, self.grayscale_btn,
                            self.brightness_btn, self.contrast_btn, self.reset_btn]:
                    btn['state'] = tk.DISABLED
            except Exception as e:
                self.status_bar['text'] = f"视频加载失败: {str(e)}"
                self.clear_display()

    def start_camera(self):
        """启动摄像头"""
        self.status_bar['text'] = "正在启动摄像头..."
        self.root.update()

        self.stop_camera()
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise ValueError("摄像头不可用")

            self.running = True
            self.stop_cam_btn['state'] = tk.NORMAL
            self.status_bar['text'] = "摄像头已启用 - 实时检测中..."
            self.update_video_frame()

            # 禁用图像处理按钮（视频流处理）
            for btn in [self.sharpen_btn, self.rotate_btn, self.grayscale_btn,
                        self.brightness_btn, self.contrast_btn, self.reset_btn]:
                btn['state'] = tk.DISABLED
        except Exception as e:
            self.status_bar['text'] = f"摄像头启动失败: {str(e)}"

    def stop_camera(self):
        """停止视频/摄像头"""
        if self.cap:
            self.cap.release()
        self.running = False
        self.stop_cam_btn['state'] = tk.DISABLED
        self.status_bar['text'] = "已停止视频输入"

    def update_video_frame(self):
        """更新视频帧"""
        if self.running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.process_and_display(frame, is_stream=True)
                self.root.after(10, self.update_video_frame)
            else:
                self.stop_camera()
                self.status_bar['text'] = "视频播放结束"

    def process_and_display(self, frame, is_stream):
        """处理并显示图像"""
        try:
            # 如果是静态图片且已应用处理
            if not is_stream and self.processed_image is not None:
                frame = self.processed_image.copy()

            # 执行预测
            orig, processed = pred(frame, stream=is_stream)

            # 转换颜色空间
            orig = self.convert_color_space(orig)
            processed = self.convert_color_space(processed)

            # 转换为PIL图像
            orig_pil = Image.fromarray(orig)
            processed_pil = Image.fromarray(processed)

            # 调整显示尺寸
            orig_display = self.resize_for_display(orig_pil)
            processed_display = self.resize_for_display(processed_pil)

            # 更新显示
            self.update_image_display(orig_display, processed_display)

            # 保存最后显示的图像
            self.last_orig = orig_display
            self.last_processed = processed_display

            # 释放资源
            del orig, processed, orig_pil, processed_pil

        except Exception as e:
            self.status_bar['text'] = f"处理错误: {str(e)}"
            self.clear_display()

    def apply_image_processing(self, operation):
        """应用图像处理操作"""
        if self.current_image is None:
            return

        try:
            # 应用选定的处理操作
            if operation == 'sharpen':
                self.sharpen_factor = min(3.0, self.sharpen_factor + 0.5)
                self.processed_image = ImageProcessor.sharpen_image(self.current_image, self.sharpen_factor)
                self.status_bar['text'] = f"锐化应用 (强度: {self.sharpen_factor:.1f})"

            elif operation == 'rotate':
                self.rotation_angle = (self.rotation_angle + 90) % 360
                self.processed_image = ImageProcessor.rotate_image(self.current_image, self.rotation_angle)
                self.status_bar['text'] = f"旋转应用 ({self.rotation_angle}°)"

            elif operation == 'grayscale':
                self.grayscale_enabled = not self.grayscale_enabled
                if self.grayscale_enabled:
                    self.processed_image = ImageProcessor.grayscale_image(self.current_image)
                    self.status_bar['text'] = "灰度化应用"
                else:
                    self.processed_image = self.current_image.copy()
                    self.status_bar['text'] = "灰度化已取消"

            elif operation == 'brightness':
                self.brightness_factor = min(3.0, self.brightness_factor + 0.5)
                self.processed_image = ImageProcessor.adjust_brightness(self.current_image, self.brightness_factor)
                self.status_bar['text'] = f"亮度增强 (强度: {self.brightness_factor:.1f})"

            elif operation == 'contrast':
                self.contrast_factor = min(3.0, self.contrast_factor + 0.5)
                self.processed_image = ImageProcessor.adjust_contrast(self.current_image, self.contrast_factor)
                self.status_bar['text'] = f"对比度增强 (强度: {self.contrast_factor:.1f})"

            # 重新处理并显示
            self.process_and_display(self.processed_image, is_stream=False)

        except Exception as e:
            self.status_bar['text'] = f"图像处理失败: {str(e)}"

    def reset_image_processing(self):
        """重置所有图像处理"""
        if self.current_image is not None:
            self.processed_image = self.current_image.copy()
            self.rotation_angle = 0
            self.sharpen_factor = 1.0
            self.brightness_factor = 1.0
            self.contrast_factor = 1.0
            self.grayscale_enabled = False
            self.status_bar['text'] = "图像处理已重置"

            # 重新处理并显示
            self.process_and_display(self.processed_image, is_stream=False)

    def convert_color_space(self, img):
        """转换颜色空间处理"""
        if len(img.shape) == 2:  # 灰度图
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # 带透明通道
            return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def resize_for_display(self, img):
        """调整图像尺寸用于显示"""
        try:
            width, height = img.size
            if width == 0 or height == 0:
                return Image.new("RGB", (1, 1), (240, 240, 240))

            aspect = width / height
            if aspect > 1:
                new_width = min(width, MAX_WIDTH)
                new_height = int(new_width / aspect)
            else:
                new_height = min(height, MAX_HEIGHT)
                new_width = int(new_height * aspect)

            return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        except:
            return Image.new("RGB", (300, 300), (240, 240, 240))

    def update_image_display(self, orig_img, processed_img):
        """更新图像显示"""
        try:
            # 转换到Tkinter格式
            orig_tk = ImageTk.PhotoImage(orig_img)
            processed_tk = ImageTk.PhotoImage(processed_img)

            # 更新显示并保持引用
            self.orig_label.configure(image=orig_tk)
            self.orig_label.image = orig_tk

            self.result_label.configure(image=processed_tk)
            self.result_label.image = processed_tk

        except Exception as e:
            self.status_bar['text'] = f"显示错误: {str(e)}"
            self.clear_display()

    def clear_display(self):
        """清空显示内容"""
        blank = Image.new("RGB", (300, 300), (240, 240, 240))
        blank_tk = ImageTk.PhotoImage(blank)

        for panel in [self.orig_label, self.result_label]:
            panel.configure(image=blank_tk, text="内容不可用")
            panel.image = blank_tk


if __name__ == "__main__":
    root = tk.Tk()
    try:
        app = RadarCrackDetectionApp(root)
        root.mainloop()
    except Exception as e:
        print(f"应用程序错误: {str(e)}")