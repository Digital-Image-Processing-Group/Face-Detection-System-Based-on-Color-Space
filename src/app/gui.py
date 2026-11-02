"""Responsive Tkinter viewer; workers never call Tk APIs."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from .service import detect, export_result, load_pipeline, overlay_image


class Application:
    def __init__(self, root, model_path, config_path):
        self.root = root
        self.model_path, self.config_path = model_path, config_path
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.payload = None
        self.photos = []
        self.busy = False
        self.closed = False
        root.title("YCbCr 颜色空间 · 人脸区域检测")
        root.geometry("1120x850")
        root.minsize(780, 650)
        root.protocol("WM_DELETE_WINDOW", self.close)
        bar = ttk.Frame(root, padding=12)
        bar.pack(fill="x")
        self.open_button = ttk.Button(bar, text="选择照片", command=self.choose)
        self.open_button.pack(side="left")
        self.save_button = ttk.Button(bar, text="保存检测结果", command=self.save, state="disabled")
        self.save_button.pack(side="left", padx=12)
        self.status = tk.StringVar(value="选择 JPG / PNG 彩色照片开始检测")
        ttk.Label(bar, textvariable=self.status).pack(side="left", padx=8)
        self.grid = ttk.Frame(root, padding=(12, 0))
        self.grid.pack(fill="both", expand=True)
        self.labels = []
        for i, title in enumerate(("原图", "肤色掩膜", "近似人脸掩膜", "候选轮廓与边界框")):
            frame = ttk.LabelFrame(self.grid, text=title, padding=5)
            frame.grid(row=i//2, column=i%2, sticky="nsew", padx=4, pady=4)
            label = ttk.Label(frame, anchor="center", text="等待检测")
            label.pack(expand=True, fill="both")
            self.labels.append(label)
        for i in (0, 1):
            self.grid.rowconfigure(i, weight=1)
            self.grid.columnconfigure(i, weight=1)
        self.detail = tk.StringVar(value="输出为传统颜色与几何方法估计的区域；手、脖子与相似背景可能误检。")
        self.detail_label = ttk.Label(root, textvariable=self.detail, padding=12, wraplength=1050)
        # Reserve the footer before the expanding image grid, also in small windows.
        self.detail_label.pack(side="bottom", fill="x", before=self.grid)
        self.resize_job = None
        root.bind("<Configure>", self.on_resize)

    def set_busy(self, busy, text):
        self.busy = busy
        self.status.set(text)
        self.open_button.configure(state="disabled" if busy else "normal")
        self.save_button.configure(state="normal" if self.payload and not busy else "disabled")

    def choose(self):
        path = filedialog.askopenfilename(title="选择彩色照片", filetypes=[("彩色照片", "*.jpg *.jpeg *.png"), ("所有文件", "*.*")])
        if path:
            self.open_path(Path(path))

    def open_path(self, path):
        self.payload = None
        self.detail.set("正在读取照片并计算两层掩膜…")
        self.photos.clear()
        for label in self.labels:
            label.configure(image="", text="正在检测…")
        self.set_busy(True, "正在读取并检测…")
        def work():
            model, config = load_pipeline(self.model_path, self.config_path)
            return detect(path, model, config)
        self.watch(self.executor.submit(work), self.show_result)

    def watch(self, future, callback):
        if self.closed:
            return
        if not future.done():
            self.root.after(40, lambda: self.watch(future, callback))
            return
        try:
            callback(future.result())
        except Exception as error:
            self.set_busy(False, "操作失败")
            self.detail.set(str(error))
            if self.payload is None:
                for label in self.labels:
                    label.configure(image="", text="未能读取或处理此图片")
            messagebox.showerror("无法完成操作", str(error))

    def show_result(self, payload):
        self.payload = payload
        bgr, result, metadata = payload
        self.images = [Image.fromarray(bgr[:, :, ::-1]), Image.fromarray(result["skin_mask"]*255),
                       Image.fromarray(result["face_mask"]*255), Image.fromarray(overlay_image(bgr, result)[:, :, ::-1])]
        count = len(result["candidates"])
        self.set_busy(False, f"候选 {count} 个 · 算法 {result['timing_ms']:.1f} ms")
        self.detail.set(f"{metadata['image_name']}  ·  {metadata['width']} × {metadata['height']}  ·  " +
                        ("未检测到合格的人脸候选区域。" if count == 0 else "绿色为近似人脸区域，编号框表示候选位置。"))
        self.render()

    def on_resize(self, event):
        if event.widget is self.root and self.payload:
            self.detail_label.configure(wraplength=max(250, event.width-30))
            if self.resize_job:
                self.root.after_cancel(self.resize_job)
            self.resize_job = self.root.after(150, self.render)

    def render(self):
        if not self.payload or self.closed:
            return
        self.resize_job = None
        self.photos = []
        for original, label in zip(self.images, self.labels):
            preview = original.copy()
            preview.thumbnail((max(100, self.grid.winfo_width()//2-35), max(100, self.grid.winfo_height()//2-45)))
            photo = ImageTk.PhotoImage(preview)
            self.photos.append(photo)
            label.configure(image=photo, text="")

    def save(self):
        parent = filedialog.askdirectory(title="选择结果保存位置")
        if parent:
            stem = Path(self.payload[2]["image_name"]).stem
            target = Path(parent)/(stem+"_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
            self.set_busy(True, "正在保存原尺寸结果…")
            future = self.executor.submit(export_result, target, *self.payload)
            self.watch(future, lambda folder: self.set_busy(False, "已保存："+str(folder)))

    def close(self):
        self.closed = True
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def launch(model_path, config_path):
    root = tk.Tk()
    Application(root, model_path, config_path)
    root.mainloop()
