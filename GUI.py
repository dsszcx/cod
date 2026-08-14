import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
import json
import time
from Network_PVTv2 import Network
from tkinter import colorchooser
from threading import Thread
class CamouflageDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("伪装目标检测系统")
        self.root.geometry("1400x900")
        # 添加主标题（新增部分）
        title_label = tk.Label(root, text="伪装目标检测系统",
                               font=("Microsoft YaHei", 30, "bold"),
                               fg="#FFFFFF", bg="#2C3E50")
        title_label.place(relx=0.5, rely=0.1, anchor="center")
        # 设置背景渐变
        self.bg_canvas = tk.Canvas(root, width=1400, height=900, highlightthickness=0)
        self.bg_canvas.pack(fill="both", expand=True)
        self.create_gradient_background("#1a2a6c", "#b21f1f", "#fdbb2d")
        # 用户系统
        self.current_user = None
        self.users_file = "users.json"
        self.load_users()
        # 检测参数
        self.threshold = 0.5
        self.is_detecting = False
        self.cap = None
        self.video_writer = None
        # 界面主题
        self.bg_color = "#2c3e50"
        self.accent_color = "#3498db"
        self.text_color = "#ecf0f1"
        self.panel_bg = "#34495e"

        # 设备选择
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        # 创建界面
        self.create_login_screen()
        self.model = self.load_model()

    def create_gradient_background(self, color1, color2, color3):
        """创建渐变背景"""
        for i in range(900):
            # 计算渐变比例
            ratio = i / 900
            if ratio < 0.5:
                # 从color1渐变到color2
                r = int(int(color1[1:3], 16) * (1 - ratio * 2) + int(color2[1:3], 16) * ratio * 2)
                g = int(int(color1[3:5], 16) * (1 - ratio * 2) + int(color2[3:5], 16) * ratio * 2)
                b = int(int(color1[5:7], 16) * (1 - ratio * 2) + int(color2[5:7], 16) * ratio * 2)
            else:
                # 从color2渐变到color3
                r = int(int(color2[1:3], 16) * (2 - ratio * 2) + int(color3[1:3], 16) * (ratio * 2 - 1))
                g = int(int(color2[3:5], 16) * (2 - ratio * 2) + int(color3[3:5], 16) * (ratio * 2 - 1))
                b = int(int(color2[5:7], 16) * (2 - ratio * 2) + int(color3[5:7], 16) * (ratio * 2 - 1))

            color = f"#{r:02x}{g:02x}{b:02x}"
            self.bg_canvas.create_line(0, i, 1400, i, fill=color)

    def load_users(self):
        try:
            with open(self.users_file, 'r') as f:
                self.users = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.users = {}

    def save_users(self):
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f)

    def create_login_screen(self):
        self.clear_window()

        # 登录框
        login_frame = tk.Frame(self.bg_canvas, bg=self.bg_color, bd=2, relief="ridge")
        self.bg_canvas.create_window(700, 450, window=login_frame)

        tk.Label(login_frame, text="用户登录", font=("Microsoft YaHei", 16),
                 bg=self.bg_color, fg=self.accent_color).grid(row=0, column=0, columnspan=2, pady=10)

        tk.Label(login_frame, text="用户名:", font=("Microsoft YaHei", 12),
                 bg=self.bg_color, fg=self.text_color).grid(row=1, column=0, pady=5, sticky="e")
        self.username_entry = tk.Entry(login_frame, font=("Microsoft YaHei", 12), bg="#34495e", fg="white")
        self.username_entry.grid(row=1, column=1, pady=5, padx=5)

        tk.Label(login_frame, text="密码:", font=("Microsoft YaHei", 12),
                 bg=self.bg_color, fg=self.text_color).grid(row=2, column=0, pady=5, sticky="e")
        self.password_entry = tk.Entry(login_frame, show="*", font=("Microsoft YaHei", 12), bg="#34495e", fg="white")
        self.password_entry.grid(row=2, column=1, pady=5, padx=5)

        btn_frame = tk.Frame(login_frame, bg=self.bg_color)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=10)

        tk.Button(btn_frame, text="登录", command=self.login, font=("Microsoft YaHei", 12),
                  bg=self.accent_color, fg="white", relief="flat", padx=20).pack(side="left", padx=5)
        tk.Button(btn_frame, text="注册", command=self.show_register, font=("Microsoft YaHei", 12),
                  bg="#2ecc71", fg="white", relief="flat", padx=20).pack(side="left", padx=5)

        # 主题设置按钮
        tk.Button(self.bg_canvas, text="主题设置", command=self.show_theme_settings, font=("Microsoft YaHei", 10),
                  bg="black", fg="white", relief="flat").place(relx=0.95, rely=0.05, anchor="ne")

    def show_register(self):
        register_window = tk.Toplevel(self.root)
        register_window.title("用户注册")
        register_window.geometry("400x300")
        register_window.configure(bg=self.bg_color)

        tk.Label(register_window, text="用户注册", font=("Microsoft YaHei", 16),
                 bg=self.bg_color, fg=self.accent_color).pack(pady=10)

        tk.Label(register_window, text="用户名:", font=("Microsoft YaHei", 12),
                 bg=self.bg_color, fg=self.text_color).pack()
        reg_username = tk.Entry(register_window, font=("Microsoft YaHei", 12), bg="#34495e", fg="white")
        reg_username.pack(pady=5)

        tk.Label(register_window, text="密码:", font=("Microsoft YaHei", 12),
                 bg=self.bg_color, fg=self.text_color).pack()
        reg_password = tk.Entry(register_window, show="*", font=("Microsoft YaHei", 12), bg="#34495e", fg="white")
        reg_password.pack(pady=5)

        tk.Label(register_window, text="确认密码:", font=("Microsoft YaHei", 12),
                 bg=self.bg_color, fg=self.text_color).pack()
        reg_confirm = tk.Entry(register_window, show="*", font=("Microsoft YaHei", 12), bg="#34495e", fg="white")
        reg_confirm.pack(pady=5)

        def register():
            username = reg_username.get()
            password = reg_password.get()
            confirm = reg_confirm.get()

            if not username or not password:
                messagebox.showerror("错误", "用户名和密码不能为空")
                return

            if password != confirm:
                messagebox.showerror("错误", "两次输入的密码不一致")
                return

            if username in self.users:
                messagebox.showerror("错误", "用户名已存在")
                return

            self.users[username] = password
            self.save_users()
            messagebox.showinfo("成功", "注册成功")
            register_window.destroy()

        tk.Button(register_window, text="注册", command=register, font=("Microsoft YaHei", 12),
                  bg=self.accent_color, fg="white", relief="flat", padx=20).pack(pady=10)

    def login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()

        if not username or not password:
            messagebox.showerror("错误", "请输入用户名和密码")
            return

        if username in self.users and self.users[username] == password:
            self.current_user = username
            self.create_main_interface()
        else:
            messagebox.showerror("错误", "用户名或密码错误")

    def show_theme_settings(self):
        theme_window = tk.Toplevel(self.root)
        theme_window.title("主题设置")
        theme_window.geometry("400x300")
        theme_window.configure(bg="black")

        tk.Label(theme_window, text="主题设置", font=("Microsoft YaHei", 16),
                 bg="black", fg=self.accent_color).pack(pady=10)

        def choose_bg_color():
            color = colorchooser.askcolor(title="选择背景颜色")[1]
            if color:
                self.bg_color = color
                self.update_theme()

        def choose_accent_color():
            color = colorchooser.askcolor(title="选择主题颜色")[1]
            if color:
                self.accent_color = color
                self.update_theme()

        def choose_text_color():
            color = colorchooser.askcolor(title="选择文字颜色")[1]
            if color:
                self.text_color = color
                self.update_theme()

        tk.Button(theme_window, text="背景颜色", command=choose_bg_color, font=("Microsoft YaHei", 12),
                  bg=self.bg_color, fg=self.text_color, relief="flat").pack(pady=5, fill="x", padx=50)

        tk.Button(theme_window, text="主题颜色", command=choose_accent_color, font=("Microsoft YaHei", 12),
                  bg=self.bg_color, fg=self.text_color, relief="flat").pack(pady=5, fill="x", padx=50)

        tk.Button(theme_window, text="文字颜色", command=choose_text_color, font=("Microsoft YaHei", 12),
                  bg=self.bg_color, fg=self.text_color, relief="flat").pack(pady=5, fill="x", padx=50)

        def apply_theme():
            self.update_theme()
            theme_window.destroy()

        tk.Button(theme_window, text="应用", command=apply_theme, font=("Microsoft YaHei", 12),
                  bg=self.accent_color, fg="white", relief="flat").pack(pady=10, ipadx=20)

    def update_theme(self):
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Frame) or isinstance(widget, tk.Label):
                widget.configure(bg=self.bg_color, fg=self.text_color)

        # 更新按钮样式
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Button):
                if widget["text"] in ["打开图片", "执行检测", "开始摄像头", "停止检测", "打开视频", "保存结果"]:
                    widget.configure(bg=self.accent_color, fg="white")
                else:
                    widget.configure(bg=self.bg_color, fg=self.text_color)

    def create_main_interface(self):
        self.clear_window()

        # 顶部工具栏
        toolbar = tk.Frame(self.bg_canvas, bg=self.bg_color, height=50, bd=0)
        self.bg_canvas.create_window(700, 30, window=toolbar, width=1380)

        # 用户信息
        user_frame = tk.Frame(toolbar, bg=self.bg_color)
        user_frame.pack(side="left", padx=10)
        tk.Label(user_frame, text=f"用户: {self.current_user}", font=("Microsoft YaHei", 12),
                 bg=self.bg_color, fg=self.text_color).pack(side="left")

        # 登出按钮
        tk.Button(toolbar, text="登出", command=self.logout, font=("Microsoft YaHei", 12),
                  bg="black", fg="white", relief="flat").pack(side="right", padx=10)

        # 主题设置按钮
        tk.Button(toolbar, text="主题设置", command=self.show_theme_settings, font=("Microsoft YaHei", 12),
                  bg="black", fg="white", relief="flat").pack(side="right", padx=10)

        # 主内容区 - 左右布局
        main_frame = tk.Frame(self.bg_canvas, bg=self.bg_color, bd=0)
        self.bg_canvas.create_window(700, 450, window=main_frame, width=1380, height=800)

        # 左侧控制面板
        control_frame = tk.Frame(main_frame, bg=self.bg_color, width=300, relief="ridge", bd=2)
        control_frame.pack(side="left", fill="y", padx=10, pady=10)

        # 阈值调节
        threshold_frame = tk.Frame(control_frame, bg=self.bg_color)
        threshold_frame.pack(fill="x", pady=10, padx=10)
        tk.Label(threshold_frame, text="检测阈值", font=("Microsoft YaHei", 12),
                 bg=self.bg_color, fg=self.text_color).pack(anchor="w")

        self.threshold_slider = tk.Scale(threshold_frame, from_=0.1, to=0.9, resolution=0.05,
                                         orient="horizontal", command=self.update_threshold,
                                         bg=self.bg_color, fg=self.text_color,
                                         highlightbackground=self.bg_color)
        self.threshold_slider.set(self.threshold)
        self.threshold_slider.pack(fill="x")

        # 操作按钮
        btn_style = {
            "font": ("Microsoft YaHei", 12),
            "bg": "black",
            "fg": "white",
            "relief": "flat",
            "padx": 10,
            "pady": 5,
            "width": 15
        }

        tk.Button(control_frame, text="打开图片", command=self.open_image, **btn_style).pack(pady=5, fill="x")
        tk.Button(control_frame, text="执行检测", command=self.detect_camouflage, **btn_style).pack(pady=5, fill="x")
        tk.Button(control_frame, text="开始摄像头", command=self.start_camera, **btn_style).pack(pady=5, fill="x")
        tk.Button(control_frame, text="停止检测", command=self.stop_detection, **btn_style).pack(pady=5, fill="x")
        tk.Button(control_frame, text="打开视频", command=self.open_video, **btn_style).pack(pady=5, fill="x")
        tk.Button(control_frame, text="保存结果", command=self.save_result, **btn_style).pack(pady=5, fill="x")

        # 右侧显示区 - 左右分屏
        display_frame = tk.Frame(main_frame, bg=self.bg_color, bd=0)
        display_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # 左侧原始图像显示
        orig_frame = tk.Frame(display_frame, bg=self.bg_color, bd=2, relief="ridge")
        orig_frame.pack(side="left", fill="both", expand=True, padx=5)

        tk.Label(orig_frame, text="原始图像", font=("Microsoft YaHei", 12),
                 bg=self.bg_color, fg=self.text_color).pack(anchor="w", pady=5)

        self.original_panel = tk.Label(orig_frame, bg=self.panel_bg, bd=0)
        self.original_panel.pack(fill="both", expand=True, padx=5, pady=5)

        # 右侧结果图像显示
        result_frame = tk.Frame(display_frame, bg=self.bg_color, bd=2, relief="ridge")
        result_frame.pack(side="right", fill="both", expand=True, padx=5)

        tk.Label(result_frame, text="检测结果", font=("Microsoft YaHei", 12),
                 bg=self.bg_color, fg=self.text_color).pack(anchor="w", pady=5)

        self.result_panel = tk.Label(result_frame, bg=self.panel_bg, bd=0)
        self.result_panel.pack(fill="both", expand=True, padx=5, pady=5)

        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        status_bar = tk.Label(self.bg_canvas, textvariable=self.status_var, font=("Microsoft YaHei", 10),
                              bg="#2C3E50", fg="white", anchor="w")
        self.bg_canvas.create_window(700, 870, window=status_bar, width=1380)

    def clear_window(self):
        for widget in self.bg_canvas.winfo_children():
            widget.destroy()
        self.create_gradient_background("#1a2a6c", "#b21f1f", "#fdbb2d")

    def logout(self):
        self.current_user = None
        self.create_login_screen()

    def update_threshold(self, val):
        self.threshold = float(val)

    def load_model(self):
        model = Network(channel=64)
        try:
            state_dict = torch.load(
                r"C:\Users\Lenovo\Desktop\ultralytics-main\KKK-Net-main\KKK_best_48-LOSS-backbone.pth",
                map_location=self.device
            )

            if 'model' in state_dict:
                state_dict = state_dict['model']

            # 清理module前缀并匹配形状
            new_dict = {}
            for k, v in state_dict.items():
                key = k.replace('module.', '')
                if key in model.state_dict() and model.state_dict()[key].shape == v.shape:
                    new_dict[key] = v
                else:
                    print(f"[跳过] 权重不匹配: {key}")

            model.load_state_dict(new_dict, strict=False)
            model.to(self.device).eval()
            return model
        except Exception as e:
            messagebox.showerror("错误", f"模型加载失败: {str(e)}")
            self.root.destroy()

    def open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("图片文件", "*.jpg;*.jpeg;*.png;*.bmp")]
        )
        if path:
            try:
                self.original_image = Image.open(path).convert('RGB')
                self.display_image(self.original_image, self.original_panel)
                self.status_var.set(f"已加载: {os.path.basename(path)}")
            except Exception as e:
                messagebox.showerror("错误", f"图片加载失败: {str(e)}")

    def detect_camouflage(self):
        if not hasattr(self, 'original_image'):
            messagebox.showwarning("警告", "请先打开一张图片")
            return

        try:
            input_tensor = self.preprocess_image(self.original_image)

            with torch.no_grad():
                outputs = self.model(input_tensor.unsqueeze(0).to(self.device))
                pred = torch.sigmoid(outputs[0]).squeeze().cpu().numpy()

                flipped_tensor = torch.flip(input_tensor, [2]).to(self.device)
                flip_outputs = self.model(flipped_tensor.unsqueeze(0))
                flip_pred = torch.sigmoid(flip_outputs[0]).squeeze().cpu().numpy()

                flip_pred = np.fliplr(flip_pred)
                final_pred = 0.6 * pred + 0.4 * flip_pred

            result = self.postprocess(final_pred)
            self.result_image = Image.fromarray(result)
            self.display_image(self.result_image, self.result_panel)
            self.status_var.set("检测完成")

        except Exception as e:
            messagebox.showerror("错误", f"检测失败: {str(e)}")
            self.status_var.set("检测错误")

    def start_camera(self):
        if self.is_detecting:
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("错误", "无法打开摄像头")
            return

        self.is_detecting = True
        self.status_var.set("摄像头检测中...")
        Thread(target=self.camera_detection_loop, daemon=True).start()

    def camera_detection_loop(self):
        while self.is_detecting:
            ret, frame = self.cap.read()
            if not ret:
                break

            # 转换颜色空间
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            # 显示原始图像
            self.original_image = pil_img
            self.display_image(pil_img, self.original_panel)

            # 执行检测
            input_tensor = self.preprocess_image(pil_img)

            with torch.no_grad():
                outputs = self.model(input_tensor.unsqueeze(0).to(self.device))
                pred = torch.sigmoid(outputs[0]).squeeze().cpu().numpy()

                flipped_tensor = torch.flip(input_tensor, [2]).to(self.device)
                flip_outputs = self.model(flipped_tensor.unsqueeze(0))
                flip_pred = torch.sigmoid(flip_outputs[0]).squeeze().cpu().numpy()

                flip_pred = np.fliplr(flip_pred)
                final_pred = 0.6 * pred + 0.4 * flip_pred

            # 后处理
            pred_8bit = (final_pred * 255).astype(np.uint8)
            _, binary = cv2.threshold(pred_8bit, int(self.threshold * 255), 255, cv2.THRESH_BINARY)

            # 转换为彩色掩码
            mask = cv2.resize(binary, (frame.shape[1], frame.shape[0]))
            mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

            # 叠加显示
            overlay = cv2.addWeighted(frame, 0.7, mask_colored, 0.3, 0)
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

            # 显示结果
            self.result_image = Image.fromarray(overlay_rgb)
            self.display_image(self.result_image, self.result_panel)

            # 写入视频
            if self.video_writer:
                self.video_writer.write(overlay)

            # 控制帧率
            time.sleep(0.03)

        self.cap.release()
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        self.status_var.set("摄像头检测已停止")

    def stop_detection(self):
        self.is_detecting = False
        self.status_var.set("正在停止检测...")

    def open_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("视频文件", "*.mp4;*.avi;*.mov")]
        )
        if path:
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                messagebox.showerror("错误", "无法打开视频文件")
                return

            # 获取视频信息
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 创建视频保存路径
            save_path = os.path.splitext(path)[0] + "_detected.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

            self.is_detecting = True
            self.status_var.set(f"视频检测中: {os.path.basename(path)}")
            Thread(target=self.camera_detection_loop, daemon=True).start()

    def save_result(self):
        if not hasattr(self, 'result_image'):
            messagebox.showwarning("警告", "没有可保存的检测结果")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG 图片", "*.png"), ("JPEG 图片", "*.jpg"), ("所有文件", "*.*")]
        )
        if path:
            try:
                self.result_image.save(path)
                self.status_var.set(f"结果已保存到: {os.path.basename(path)}")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {str(e)}")

    def preprocess_image(self, image):
        transform = transforms.Compose([
            transforms.Resize((352, 352)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform(image)

    def postprocess(self, pred):
        pred_8bit = (pred * 255).astype(np.uint8)
        _, binary = cv2.threshold(pred_8bit, int(self.threshold * 255), 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        orig_size = self.original_image.size
        mask = cv2.resize(cleaned, orig_size)

        orig_np = np.array(self.original_image)
        overlay = orig_np.copy()
        overlay[mask > 128] = (0, 0, 255)

        alpha = 0.4
        blended = cv2.addWeighted(overlay, alpha, orig_np, 1 - alpha, 0)
        return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

    def display_image(self, image, panel):
        max_size = 500
        ratio = min(max_size / image.width, max_size / image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))

        resized = image.resize(new_size, Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(resized)

        panel.config(image=tk_img)
        panel.image = tk_img


if __name__ == "__main__":
    root = tk.Tk()
    app = CamouflageDetectionApp(root)
    root.mainloop()