
"""
AstraFilter - 候选体验证模块
多帧关联 + MPChecker 查询 + 轨迹拟合
"""

import numpy as np
import pandas as pd
import requests
from astropy.time import Time
from astropy.io import fits
from astropy.wcs import WCS


def clean_date(date_str):
    """统一日期格式，去掉连字符。'2023-10-29' -> '20231029'"""
    return str(date_str).replace('-', '').replace('/', '')


class CandidateVerifier:
    """候选体验证器"""
    
    def __init__(self, cluster_radius=100, r2_threshold=0.85,
                 min_velocity=3.0, min_motion=20.0):
        self.cluster_radius = cluster_radius
        self.r2_threshold = r2_threshold
        self.min_velocity = min_velocity
        self.min_motion = min_motion
    
    def cluster(self, df):
        if len(df) == 0:
            return []
        df = df.sort_values('linearity', ascending=False).reset_index(drop=True)
        used = [False] * len(df)
        clusters = []
        for i, row in df.iterrows():
            if used[i]:
                continue
            cluster = [row.to_dict()]
            used[i] = True
            for j, row2 in df.iterrows():
                if used[j] or j == i:
                    continue
                dist = np.sqrt((row['x']-row2['x'])**2 + (row['y']-row2['y'])**2)
                if dist < self.cluster_radius:
                    cluster.append(row2.to_dict())
                    used[j] = True
            clusters.append(cluster)
        return clusters
    
    def fit_track(self, cluster):
        best_per_date = {}
        for c in cluster:
            date = clean_date(c.get('date'))
            if date not in best_per_date or c['linearity'] > best_per_date[date]['linearity']:
                best_per_date[date] = c
        
        points = sorted(best_per_date.values(), key=lambda x: clean_date(x['date']))
        n_dates = len(points)
        if n_dates < 3:
            return None
        
        days = np.array([Time(f"{clean_date(p['date'])[:4]}-{clean_date(p['date'])[4:6]}-{clean_date(p['date'])[6:8]}").mjd for p in points])
        xs = np.array([p['x'] for p in points])
        ys = np.array([p['y'] for p in points])
        days_rel = days - days.min()
        
        try:
            coef_x = np.polyfit(days_rel, xs, 1)
            coef_y = np.polyfit(days_rel, ys, 1)
        except Exception:
            return None
        
        pred_x = np.polyval(coef_x, days_rel)
        pred_y = np.polyval(coef_y, days_rel)
        r2_x = 1 - np.sum((xs-pred_x)**2) / (np.sum((xs-xs.mean())**2) + 1e-6)
        r2_y = 1 - np.sum((ys-pred_y)**2) / (np.sum((ys-ys.mean())**2) + 1e-6)
        
        v = np.sqrt(coef_x[0]**2 + coef_y[0]**2)
        total_motion = np.sqrt((xs[-1]-xs[0])**2 + (ys[-1]-ys[0])**2)
        
        return {
            'n_dates': n_dates,
            'r2_x': float(r2_x),
            'r2_y': float(r2_y),
            'velocity': float(v),
            'total_motion': float(total_motion),
            'points': points,
            'vx': float(coef_x[0]),
            'vy': float(coef_y[0]),
        }
    
    def filter_tracks(self, tracks):
        good = []
        for t in tracks:
            if (t['r2_x'] > self.r2_threshold and
                t['r2_y'] > self.r2_threshold and
                t['velocity'] > self.min_velocity and
                t['total_motion'] > self.min_motion):
                good.append(t)
        return good
    
    def check_mpc(self, ra, dec, radius_arcmin=5, date=None):
        url = "https://www.minorplanetcenter.net/cgi-bin/checkmp.cgi"
        if date is None:
            date = "20231019"
        date = clean_date(date)
        
        params = {
            'year': date[:4],
            'month': date[4:6],
            'day': date[6:8],
            'which': 'pos',
            'ra': ra,
            'dec': dec,
            'r': str(radius_arcmin),
            'limit': '24.0',
            'oc': '500',
            'sort': 'd',
            'mot': 'h',
            'tmot': 's',
            'pdes': 'u',
            'needed': 'f',
            'ps': 'n',
            'type': 'p',
        }
        
        try:
            r = requests.get(url, params=params, timeout=30)
            text = r.text
            if "No known minor planets" in text:
                return {'status': 'clear', 'n_objects': 0, 'message': '无已知天体', 'raw': text[:500]}
            else:
                import re
                rows = re.findall(r'<tr[^>]*>(.*?)</tr>', text, re.DOTALL)
                n = len([row for row in rows if 'href' in row and 'mp' in row.lower()])
                return {'status': 'found', 'n_objects': n, 'message': f'找到 {n} 颗已知天体', 'raw': text[:500]}
        except Exception as e:
            return {'status': 'error', 'n_objects': -1, 'message': str(e), 'raw': ''}
    
    def pixel_to_radec(self, filepath, x, y):
        with fits.open(filepath) as hdul:
            for hdu in hdul:
                if hdu.data is not None and len(hdu.data.shape) == 2:
                    header = hdu.header
                    break
        wcs = WCS(header)
        ra, dec = wcs.all_pix2world(x, y, 0)
        return float(ra), float(dec)
    
    def verify(self, candidates_df, data_dir):
        if len(candidates_df) == 0:
            return [], {'n_clusters': 0, 'n_tracks': 0, 'n_good': 0}
        
        clusters = self.cluster(candidates_df)
        tracks = []
        for c in clusters:
            t = self.fit_track(c)
            if t is not None:
                tracks.append(t)
        
        good_tracks = self.filter_tracks(tracks)
        
        for t in good_tracks:
            p = t['points'][0]
            fpath = f"{data_dir}/{p['filename']}"
            try:
                ra, dec = self.pixel_to_radec(fpath, p['x'], p['y'])
                t['ra'] = ra
                t['dec'] = dec
                t['mpc'] = self.check_mpc(ra, dec, date=p['date'])
            except Exception as e:
                t['ra'] = None
                t['dec'] = None
                t['mpc'] = {'status': 'error', 'message': str(e)}
        
        summary = {
            'n_candidates': len(candidates_df),
            'n_clusters': len(clusters),
            'n_tracks': len(tracks),
            'n_good': len(good_tracks),
        }
        return good_tracks, summary
