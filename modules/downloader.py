
"""
AstraFilter - 数据下载模块
从 IRSA 的 ZTF 存档中批量下载差分图像
"""

import os
import requests
import pandas as pd
import astropy.time as time
from ztfquery import query
from getpass import getpass


class ZTFDownloader:
    """ZTF 数据下载器"""
    
    def __init__(self, email=None, password=None, data_dir='./data'):
        self.email = email or input("IRSA 邮箱: ")
        self.password = password or getpass("IRSA 密码: ")
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def query(self, ra, dec, start_date, end_date, radius=0.5):
        zquery = query.ZTFQuery()
        start_jd = time.Time(start_date).jd
        end_jd = time.Time(end_date).jd
        zquery.load_metadata(
            radec=[ra, dec],
            size=radius,
            sql_query=f"obsjd BETWEEN {start_jd} AND {end_jd}"
        )
        meta = zquery.metatable.copy()
        if len(meta) == 0:
            return pd.DataFrame()
        meta = meta.sort_values('obsjd').reset_index(drop=True)
        return meta
    
    def pick_best_field(self, meta, min_observations=10):
        from collections import Counter
        combos = Counter(zip(meta['field'], meta['ccdid']))
        for (field, ccdid), count in combos.most_common():
            if count >= min_observations:
                mask = (meta['field'] == field) & (meta['ccdid'] == ccdid)
                subset = meta[mask].sort_values('obsjd').reset_index(drop=True)
                return field, ccdid, subset
        return None, None, None
    
    def download_single(self, row):
        filefracday = str(int(row['filefracday']))
        year = filefracday[:4]
        monthday = filefracday[4:8]
        fracday = filefracday[8:]
        field = f"{int(row['field']):06d}"
        filtercode = row['filtercode']
        ccdid = f"c{int(row['ccdid']):02d}"
        qid = f"q{int(row['qid'])}"
        fname = f"ztf_{filefracday}_{field}_{filtercode}_{ccdid}_o_{qid}_scimrefdiffimg.fits.fz"
        fpath = os.path.join(self.data_dir, fname)
        if os.path.exists(fpath):
            return True, fpath
        url = f"https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/{year}/{monthday}/{fracday}/{fname}"
        try:
            r = requests.get(url, auth=(self.email, self.password), timeout=60)
            if r.status_code == 200:
                with open(fpath, 'wb') as f:
                    f.write(r.content)
                return True, fpath
            else:
                return False, f"HTTP {r.status_code}"
        except Exception as e:
            return False, str(e)
    
    def download_batch(self, subset, max_count=30):
        n = min(max_count, len(subset))
        downloaded = 0
        records = []
        for i in range(n):
            row = subset.iloc[i]
            success, result = self.download_single(row)
            if success:
                downloaded += 1
                records.append({
                    'filename': os.path.basename(result),
                    'filepath': result,
                    'obsjd': row['obsjd'],
                    'date': time.Time(row['obsjd'], format='jd').iso[:10],
                    'field': int(row['field']),
                    'ccdid': int(row['ccdid']),
                    'filtercode': row['filtercode'],
                    'qid': int(row['qid']),
                })
                print(f"[{i+1}/{n}] OK {os.path.basename(result)[:55]}")
            else:
                print(f"[{i+1}/{n}] FAIL {result}")
        meta_df = pd.DataFrame(records)
        meta_path = os.path.join(self.data_dir, 'metadata.csv')
        meta_df.to_csv(meta_path, index=False)
        return downloaded, meta_df
