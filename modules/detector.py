
"""
AstraFilter - 条纹检测模块
整合传统图像处理和 AI 神经网络两种检测方法
"""

import os
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy import ndimage


class StripeDetector:
    """条纹检测器：传统方法 + AI 方法"""

    def __init__(self, method='traditional', model_path=None):
        """
        参数:
            method: 'traditional' / 'unet' / 'both'
            model_path: U-Net 模型路径（method 含 unet 时必填）
        """
        self.method = method
        self.model_path = model_path
        self.model = None
        
        if method in ('unet', 'both') and model_path:
            self._load_unet()
    
    def _load_unet(self):
        """加载 U-Net 模型"""
        import torch
        import torch.nn as nn
        from torch import nn as t_nn
        
        class UNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.enc1 = nn.Sequential(
                    nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                    nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                    nn.MaxPool2d(2))
                self.enc2 = nn.Sequential(
                    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                    nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                    nn.MaxPool2d(2))
                self.enc3 = nn.Sequential(
                    nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                    nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                    nn.MaxPool2d(2))
                self.bottleneck = nn.Sequential(
                    nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                    nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU())
                self.dec3 = nn.Sequential(
                    nn.ConvTranspose2d(256, 128, 2, stride=2), nn.ReLU(),
                    nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
                self.dec2 = nn.Sequential(
                    nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU(),
                    nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
                self.dec1 = nn.Sequential(
                    nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(),
                    nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
                self.out = nn.Conv2d(32, 1, 1)
            def forward(self, x):
                e1 = self.enc1(x); e2 = self.enc2(e1); e3 = self.enc3(e2)
                b = self.bottleneck(e3)
                d3 = self.dec3(b); d2 = self.dec2(d3); d1 = self.dec1(d2)
                return self.out(d1)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UNet().to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        print(f"U-Net 加载成功，设备: {self.device}")

    def detect_traditional(self, image, n_sigma=5, min_length=10, min_aspect=3.0, min_linearity=2.5):
        """
        传统图像处理检测条纹
        
        参数:
            image: 2D numpy 数组
            n_sigma: 亮度阈值（相对背景的 sigma 倍数）
            min_length: 最短条纹长度（像素）
            min_aspect: 最小长宽比
            min_linearity: 最小 PCA 线性度
        
        返回:
            候选体列表
        """
        med = np.nanmedian(image)
        std = np.nanstd(image)
        img = np.nan_to_num(image, nan=0.0)
        
        binary = img > (med + n_sigma * std)
        labeled, num = ndimage.label(binary)
        
        candidates = []
        for i in range(1, num + 1):
            ys, xs = np.where(labeled == i)
            n_pix = len(ys)
            if n_pix < 8 or n_pix > 500:
                continue
            
            h = ys.max() - ys.min() + 1
            w = xs.max() - xs.min() + 1
            length = max(h, w)
            width = min(h, w)
            
            if length < min_length or width < 2:
                continue
            
            aspect = length / (width + 1e-6)
            if aspect < min_aspect:
                continue
            
            # PCA 线性度
            pts = np.column_stack([xs, ys]).astype(float)
            pts_centered = pts - pts.mean(axis=0)
            cov = np.cov(pts_centered.T)
            if cov.size == 4:
                eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
                if eigvals[1] > 1e-6:
                    linearity = np.sqrt(eigvals[0] / eigvals[1])
                else:
                    linearity = 999.0
            else:
                linearity = 999.0
            
            if linearity < min_linearity:
                continue
            
            candidates.append({
                'x': int(xs.mean()),
                'y': int(ys.mean()),
                'length': int(length),
                'width': int(width),
                'aspect': float(aspect),
                'linearity': float(linearity),
                'n_pixels': int(n_pix),
                'method': 'traditional',
            })
        
        # 去重
        candidates = sorted(candidates, key=lambda c: -c['linearity'])
        kept = []
        for c in candidates:
            if all(np.sqrt((c['x']-k['x'])**2 + (c['y']-k['y'])**2) > 50 for k in kept):
                kept.append(c)
        return kept

    def detect_unet(self, image, patch_size=128, stride=64, min_pixels=50):
        """U-Net 检测"""
        if self.model is None:
            raise ValueError("U-Net 模型未加载")
        
        import torch
        img = np.nan_to_num(image, nan=0.0)
        H, W = img.shape
        
        patches, coords = [], []
        for y in range(0, H - patch_size, stride):
            for x in range(0, W - patch_size, stride):
                p = img[y:y+patch_size, x:x+patch_size].copy()
                p = (p - np.mean(p)) / (np.std(p) + 1e-6)
                p = np.clip(p, -10, 10)
                patches.append(p)
                coords.append((x, y))
        
        candidates = []
        for i in range(0, len(patches), 32):
            batch = np.array(patches[i:i+32]).reshape(-1, 1, patch_size, patch_size).astype(np.float32)
            with torch.no_grad():
                out = torch.sigmoid(self.model(torch.tensor(batch).to(self.device))).cpu().numpy()
            for j, prob in enumerate(out):
                n_pix = int((prob[0] > 0.5).sum())
                if n_pix >= min_pixels:
                    ys, xs = np.where(prob[0] > 0.5)
                    candidates.append({
                        'x': int(xs.mean()) + coords[i+j][0],
                        'y': int(ys.mean()) + coords[i+j][1],
                        'n_pixels': n_pix,
                        'method': 'unet',
                    })
        
        # NMS
        candidates = sorted(candidates, key=lambda c: -c['n_pixels'])
        kept = []
        for c in candidates:
            if all(np.sqrt((c['x']-k['x'])**2 + (c['y']-k['y'])**2) > 50 for k in kept):
                kept.append(c)
        return kept

    def detect_image(self, filepath, date=None, qid=None, filtercode=None):
        """检测单张图像"""
        with fits.open(filepath) as hdul:
            for hdu in hdul:
                if hdu.data is not None and len(hdu.data.shape) == 2:
                    data = np.nan_to_num(hdu.data.copy(), nan=0.0)
                    break
        
        results = []
        
        if self.method in ('traditional', 'both'):
            trad = self.detect_traditional(data)
            for c in trad:
                c['filename'] = os.path.basename(filepath)
                c['date'] = date
                c['qid'] = qid
                c['filter'] = filtercode
                results.append(c)
        
        if self.method in ('unet', 'both') and self.model is not None:
            unet = self.detect_unet(data)
            for c in unet:
                c['filename'] = os.path.basename(filepath)
                c['date'] = date
                c['qid'] = qid
                c['filter'] = filtercode
                results.append(c)
        
        return results

    def detect_batch(self, metadata_df, output_csv=None):
        """批量检测"""
        all_results = []
        n = len(metadata_df)
        for i, row in metadata_df.iterrows():
            print(f"[{i+1}/{n}] {row['date']} q{row['qid']} z{row['filtercode']}", end=" ")
            results = self.detect_image(
                row['filepath'],
                date=row['date'],
                qid=row['qid'],
                filtercode=row['filtercode']
            )
            print(f"→ {len(results)} 个候选体")
            all_results.extend(results)
        
        df = pd.DataFrame(all_results)
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"\n保存到 {output_csv}")
        return df
