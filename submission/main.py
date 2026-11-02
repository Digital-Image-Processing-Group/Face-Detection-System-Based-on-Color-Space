"""第三次project：YCbCr颜色空间人脸分割。"""
import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import sys
import time

BASE = Path(__file__).resolve().parent
if (BASE / '_packages').exists():
    sys.path.insert(0, str(BASE / '_packages'))
try:
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont, ImageOps
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--target', str(BASE / '_packages'),
                           '-r', str(BASE / 'requirements.txt')])
    os.execv(sys.executable, [sys.executable, __file__, *sys.argv[1:]])

# 训练所得参数，特征顺序为Cb、Cr。
MEAN = np.array([113.86623333333333, 143.3184875])
COVARIANCE = np.array([[13.11251944771992, -7.345890096625399],
                       [-7.345890096625399, 16.315821194932038]])
REGULARIZATION = 1.0
THRESHOLD = 4.0
POSTPROCESS = {'open_kernel': 5, 'close_kernel': 9, 'min_area_ratio': .001,
               'max_area_ratio': .85, 'min_aspect_ratio': .3, 'max_aspect_ratio': 2.,
               'min_fill_ratio': .3, 'min_ellipse_similarity': .3,
               'hole_fill_max_area_ratio': .3, 'kernel_reference_side': 1000}
DEMO_IDS = ('hfd_12_SEAFA6', 'hfd_8_SEAMA6', 'hfd_24_SEAFA8', 'hfd_5_AFA8',
            'wf_19_Couple_Couple_19_90', 'wf_1_Handshaking_Handshaking_1_453',
            'wf_27_Spa_Spa_27_293', 'bsds_105027')


def read_image(path):
    if Path(path).suffix.lower() not in ('.jpg', '.jpeg', '.png'):
        raise ValueError('请选择JPG、JPEG或PNG彩色图片')
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode not in ('RGB', 'RGBA', 'P', 'CMYK'):
            raise ValueError('输入必须是彩色图片')
        if image.mode == 'RGBA' or 'transparency' in image.info:
            if image.convert('RGBA').getchannel('A').getextrema()[0] < 255:
                raise ValueError('不支持带透明区域的图片')
        rgb = np.asarray(image.convert('RGB'), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _fill_holes_limited(region_mask, max_hole_area):
    h, w = region_mask.shape
    inv = (1 - region_mask).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    border = set(labels[0, :].tolist()) | set(labels[-1, :].tolist())
    border |= set(labels[:, 0].tolist()) | set(labels[:, -1].tolist())
    out = region_mask.copy()
    for lab in range(1, num):
        if lab in border:
            continue
        if int(stats[lab, cv2.CC_STAT_AREA]) <= max_hole_area:
            out[labels == lab] = 1
    return out


def postprocess(skin_mask, pp):
    h, w = skin_mask.shape
    scale = min(h, w) / pp.get("kernel_reference_side", min(h, w))
    def size(key):
        return min(101, max(1, round(pp[key] * scale)) | 1)
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size("open_kernel"), size("open_kernel")))
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size("close_kernel"), size("close_kernel")))
    m = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, ko)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kc)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    face = np.zeros((h, w), dtype=np.uint8)
    candidates = []
    img_area = float(h * w)

    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if area == 0:
            continue
        rel = area / img_area
        if rel < pp["min_area_ratio"] or rel > pp["max_area_ratio"]:
            continue
        ar = bw / bh if bh > 0 else 0.0
        if ar < pp["min_aspect_ratio"] or ar > pp["max_aspect_ratio"]:
            continue
        fill = area / (bw * bh) if bw * bh > 0 else 0.0
        if fill < pp["min_fill_ratio"]:
            continue

        comp = (labels[y:y + bh, x:x + bw] == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        if len(cnt) < 5:
            continue
        _, (ma, mb), _ = cv2.fitEllipse(cnt)
        ell_area = np.pi * (ma / 2.0) * (mb / 2.0)
        if ell_area <= 0:
            continue
        sim = min(area, ell_area) / max(area, ell_area)
        if sim < pp["min_ellipse_similarity"]:
            continue

        # 按当前候选面积限制填洞。
        max_hole = int(pp["hole_fill_max_area_ratio"] * area)
        roi_filled = _fill_holes_limited(comp, max_hole)
        face[y:y + bh, x:x + bw] = np.maximum(face[y:y + bh, x:x + bw], roi_filled)
        candidates.append({
            "bbox": {"x": int(x), "y": int(y), "width": int(bw), "height": int(bh)},
            "area": int(roi_filled.sum()),
        })

    return face, candidates


def segment_image(image, diagonal=False):
    start = time.perf_counter()
    color = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    cb, cr = color[:, :, 2].astype(np.float64), color[:, :, 1].astype(np.float64)
    covariance = np.diag(np.diag(COVARIANCE)) if diagonal else COVARIANCE
    inverse = np.linalg.inv(covariance + REGULARIZATION * np.eye(2))
    difference = np.stack([cb - MEAN[0], cr - MEAN[1]], axis=-1)
    distance = np.einsum('...i,ij,...j->...', difference, inverse, difference)
    skin = (distance <= THRESHOLD).astype(np.uint8)
    face, candidates = postprocess(skin, POSTPROCESS)
    return {'skin_mask': skin, 'face_mask': face, 'candidates': candidates,
            'timing_ms': (time.perf_counter() - start) * 1000}


def overlay_image(image, result):
    output = image.copy()
    mask = result['face_mask'].astype(bool)
    output[mask] = (output[mask] * .65 + np.array([70, 200, 70]) * .35).astype(np.uint8)
    contours, _ = cv2.findContours(result['face_mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (0, 220, 255), 2)
    for index, candidate in enumerate(result['candidates'], 1):
        box = candidate['bbox']
        x, y, w, h = (box[k] for k in ('x', 'y', 'width', 'height'))
        cv2.rectangle(output, (x, y), (x + w - 1, y + h - 1), (255, 160, 20), 2)
        cv2.putText(output, str(index), (x, max(15, y)), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 160, 20), 2)
    return output


def make_preview(image, result, reference=None):
    arrays = [image[:, :, ::-1], result['skin_mask'] * 255,
              result['face_mask'] * 255, overlay_image(image, result)[:, :, ::-1]]
    labels = ['Original', 'Skin mask', 'Face mask', 'Overlay']
    if reference is not None:
        arrays.extend([reference[0] * 255, reference[1] * 255])
        labels.extend(['Skin reference', 'Face reference'])
    columns = 3 if reference is not None else 2
    canvas = Image.new('RGB', (400 * columns, 660), '#f2f4f6')
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 22)
    except OSError:
        font = ImageFont.load_default()
    for i, (array, label) in enumerate(zip(arrays, labels)):
        tile = Image.fromarray(array).convert('RGB')
        tile.thumbnail((390, 280))
        x, y = (i % columns) * 400, (i // columns) * 330
        draw.text((x + 10, y + 5), label, font=font, fill='black')
        canvas.paste(tile, (x + (400 - tile.width) // 2, y + 40 + (280 - tile.height) // 2))
    return canvas


def save_result(folder, image, result, reference=None, numeric=False):
    folder.mkdir(parents=True, exist_ok=False)
    for kind in ('skin', 'face'):
        mask = result[kind + '_mask']
        Image.fromarray(mask * 255).save(folder / (kind + '_mask.png'))
        if numeric:
            np.save(folder / (kind + '_mask.npy'), mask, allow_pickle=False)
    Image.fromarray(overlay_image(image, result)[:, :, ::-1]).save(folder / 'overlay.png')
    if not numeric:
        make_preview(image, result, reference).save(folder / 'preview.jpg')


def counts(prediction, truth):
    if prediction.shape != truth.shape or not np.isin(prediction, (0, 1)).all() or not np.isin(truth, (0, 1)).all():
        raise ValueError("指标输入必须是同尺寸 0/1 掩膜")
    pred, ref = prediction.astype(bool), truth.astype(bool)
    return [int(np.count_nonzero(pred & ref)), int(np.count_nonzero(pred & ~ref)), int(np.count_nonzero(~pred & ref))]


def metrics(value):
    tp, fp, fn = map(int, value)
    def ratio(n, d):
        return n/d if d else None
    return dict(tp=tp, fp=fp, fn=fn, precision=ratio(tp, tp+fp), recall=ratio(tp, tp+fn),
                iou=ratio(tp, tp+fp+fn), dice=ratio(2*tp, 2*tp+fp+fn))


def summarize(rows):
    summary = {"images": len(rows)}
    for kind in ("skin", "face"):
        summary[kind] = metrics([sum(r[kind][k] for r in rows) for k in ("tp", "fp", "fn")])
        positive = [r[kind]["iou"] for r in rows if r[kind]["tp"]+r[kind]["fn"] > 0]
        summary[kind]["positive_images"] = len(positive)
        summary[kind]["mean_positive_image_iou"] = float(np.mean(positive)) if positive else None
    absent = [r for r in rows if r["face"]["tp"]+r["face"]["fn"] == 0]
    false = sum(r["face"]["fp"] > 0 for r in absent)
    summary["no_face"] = {"images": len(absent), "nonempty_predictions": false,
                          "image_false_positive_rate": false/len(absent) if absent else None,
                          "false_positive_pixels": sum(r["face"]["fp"] for r in absent)}
    times = [r["timing_ms"] for r in rows]
    summary["timing"] = {"samples": len(times), "median_ms": float(np.median(times)) if times else None,
                         "p95_ms": float(np.percentile(times, 95)) if times else None,
                         "max_ms": max(times) if times else None,
                         "scope": "segmentation and postprocess"}
    return summary


def collection(image_id):
    if image_id.startswith('hfd_'):
        return 'test'
    if image_id.startswith('bsds_'):
        return 'background'
    if image_id.endswith('_hand'):
        return 'hands'
    return 'scenes'


def read_reference(archive, image_id, shape):
    if image_id not in archive:
        return None
    packed = archive[image_id]
    if packed.shape != shape or not np.isin(packed, (0, 1, 2, 3)).all():
        raise ValueError('参考掩膜尺寸或取值错误：' + image_id)
    return tuple(((packed & bit) != 0).astype(np.uint8) for bit in (1, 2))


def run_batch(paths, output, compare=False):
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    with np.load(BASE / 'masks.npz', allow_pickle=False) as archive:
        for mode in (('full', 'diagonal') if compare else ('full',)):
            for index, path in enumerate(paths, 1):
                image = read_image(path)
                reference = read_reference(archive, path.stem, image.shape[:2])
                result = segment_image(image, diagonal=(mode == 'diagonal'))
                row = {'image_id': path.stem, 'collection': collection(path.stem), 'mode': mode,
                       'timing_ms': result['timing_ms'], 'candidate_count': len(result['candidates'])}
                if reference is not None:
                    for kind, truth in zip(('skin', 'face'), reference):
                        row[kind] = metrics(counts(result[kind + '_mask'], truth))
                rows.append(row)
                if mode == 'full':
                    save_result(output / path.stem, image, result, reference)
                print(f'[{mode}] {index}/{len(paths)} {path.stem}  候选{len(result["candidates"])}个', flush=True)
    fields = ['mode', 'image_id', 'collection', 'candidate_count', 'timing_ms']
    metric_fields = [kind + '_' + key for kind in ('skin', 'face')
                     for key in ('tp', 'fp', 'fn', 'precision', 'recall', 'iou', 'dice')]
    with (output / 'per_image.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + metric_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{key: row[key] for key in fields},
                             **{kind + '_' + key: value for kind in ('skin', 'face')
                                for key, value in row.get(kind, {}).items()}})
    groups = []
    labeled = [row for row in rows if 'face' in row]
    for mode, group in sorted({(row['mode'], row['collection']) for row in labeled}):
        summary = summarize([r for r in labeled if r['mode'] == mode and r['collection'] == group])
        groups.append({'mode': mode, 'collection': group, 'images': summary['images'],
                       **{kind + '_' + key: value for kind in ('skin', 'face')
                          for key, value in summary[kind].items()},
                       **{'no_face_' + key: value for key, value in summary['no_face'].items()}})
    if groups:
        with (output / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(groups[0]))
            writer.writeheader()
            writer.writerows(groups)
    overview = Image.new('RGB', (1200, 330 * ((min(8, len(paths)) + 1) // 2)), 'white')
    for i, path in enumerate(paths[:8]):
        with Image.open(output / path.stem / 'preview.jpg') as image:
            tile = image.convert('RGB')
        tile.thumbnail((600, 330))
        overview.paste(tile, ((i % 2) * 600, (i // 2) * 330))
    overview.save(output / 'overview.jpg')
    return output / 'overview.jpg'


def launch_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from concurrent.futures import ThreadPoolExecutor
    from PIL import ImageTk

    class Window:
        def __init__(self, root):
            self.root = root
            self.pool = ThreadPoolExecutor(max_workers=1)
            self.payload = None
            self.photos = []
            root.title('YCbCr颜色空间人脸分割')
            root.geometry('1100x820')
            root.minsize(800, 650)
            root.protocol('WM_DELETE_WINDOW', self.close)
            bar = ttk.Frame(root, padding=10)
            bar.pack(fill='x')
            self.open_button = ttk.Button(bar, text='选择照片', command=self.choose)
            self.open_button.pack(side='left')
            self.save_button = ttk.Button(bar, text='保存结果', command=self.save, state='disabled')
            self.save_button.pack(side='left', padx=10)
            self.status = tk.StringVar(value='选择彩色照片开始分割')
            ttk.Label(bar, textvariable=self.status).pack(side='left')
            self.detail = tk.StringVar(value='原图  肤色掩膜  人脸掩膜  候选叠加')
            self.footer = ttk.Label(root, textvariable=self.detail, padding=10, wraplength=1000)
            self.footer.pack(side='bottom', fill='x')
            self.grid = ttk.Frame(root, padding=8)
            self.grid.pack(fill='both', expand=True)
            self.labels = []
            for i, title in enumerate(('原图', '肤色掩膜', '人脸掩膜', '候选叠加')):
                frame = ttk.LabelFrame(self.grid, text=title, padding=4)
                frame.grid(row=i // 2, column=i % 2, sticky='nsew', padx=4, pady=4)
                label = ttk.Label(frame, anchor='center', text='等待图片')
                label.pack(fill='both', expand=True)
                self.labels.append(label)
            for i in (0, 1):
                self.grid.rowconfigure(i, weight=1)
                self.grid.columnconfigure(i, weight=1)
            root.bind('<Configure>', self.resize)
            self.resize_job = None

        def choose(self):
            path = filedialog.askopenfilename(filetypes=[('彩色照片', '*.jpg *.jpeg *.png')])
            if path:
                self.open_button.configure(state='disabled')
                self.save_button.configure(state='disabled')
                self.status.set('正在处理…')
                self.poll(self.pool.submit(self.process, Path(path)), self.show)

        def process(self, path):
            image = read_image(path)
            return path, image, segment_image(image)

        def poll(self, future, callback):
            if not future.done():
                self.root.after(40, lambda: self.poll(future, callback))
                return
            try:
                callback(future.result())
            except Exception as error:
                self.status.set('操作失败')
                messagebox.showerror('操作失败', str(error))
            self.open_button.configure(state='normal')
            self.save_button.configure(state='normal' if self.payload else 'disabled')

        def show(self, payload):
            self.payload = payload
            path, image, result = payload
            self.views = [Image.fromarray(image[:, :, ::-1]), Image.fromarray(result['skin_mask'] * 255),
                          Image.fromarray(result['face_mask'] * 255),
                          Image.fromarray(overlay_image(image, result)[:, :, ::-1])]
            self.status.set(f'候选{len(result["candidates"])}个  耗时{result["timing_ms"]:.1f} ms')
            self.detail.set(f'{path.name}   {image.shape[1]} × {image.shape[0]}')
            self.render()

        def resize(self, event):
            if event.widget is self.root and self.payload:
                self.footer.configure(wraplength=max(250, event.width - 30))
                if self.resize_job:
                    self.root.after_cancel(self.resize_job)
                self.resize_job = self.root.after(120, self.render)

        def render(self):
            self.resize_job = None
            self.photos = []
            for source, label in zip(self.views, self.labels):
                image = source.copy()
                image.thumbnail((max(100, self.grid.winfo_width() // 2 - 30),
                                 max(100, self.grid.winfo_height() // 2 - 45)))
                photo = ImageTk.PhotoImage(image)
                self.photos.append(photo)
                label.configure(image=photo, text='')

        def save(self):
            parent = filedialog.askdirectory(title='选择结果保存位置')
            if parent:
                path, image, result = self.payload
                folder = Path(parent) / (path.stem + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
                save_result(folder, image, result, numeric=True)
                self.status.set('已保存结果')

        def close(self):
            self.pool.shutdown(wait=False, cancel_futures=True)
            self.root.destroy()

    root = tk.Tk()
    Window(root)
    root.mainloop()


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='YCbCr颜色空间人脸分割')
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument('--all', action='store_true', help='验证全部79个输入并比较两种模型')
    choice.add_argument('--gui', action='store_true', help='打开选图界面')
    choice.add_argument('--image', type=Path, help='处理指定图片')
    parser.add_argument('--output', type=Path, help='指定新的结果目录')
    parser.add_argument('--no-open', action='store_true', help='完成后不打开总览')
    args = parser.parse_args()
    if args.gui:
        launch_gui()
        return 0
    if args.image:
        paths = [args.image.resolve()]
    else:
        images = {p.stem: p for p in (BASE / 'images').iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')}
        paths = sorted(images.values()) if args.all else [images[key] for key in DEMO_IDS]
    if not paths:
        raise ValueError('没有找到输入图片')
    output = args.output or BASE / 'results' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    overview = run_batch(paths, output, compare=args.all)
    print('结果已保存：' + str(output), flush=True)
    if not args.no_open:
        os.startfile(str(overview))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print('运行失败：' + str(error), file=sys.stderr)
        raise SystemExit(1)
