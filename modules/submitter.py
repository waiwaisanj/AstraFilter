
"""
AstraFilter - MPC 提交模块
把验证通过的候选体整理成 MPC 的 80 列格式
"""

import os
import numpy as np
from datetime import datetime
from astropy.time import Time


def clean_date(date_str):
    """去掉连字符"""
    return str(date_str).replace('-', '').replace('/', '')


class MPCSubmitter:
    """MPC 提交文件生成器"""
    
    def __init__(self, observatory_code='I41'):
        self.observatory_code = observatory_code
    
    def to_mpc_date(self, date_str, time_utc=None):
        date_str = clean_date(date_str)
        year = date_str[:4]
        month = date_str[4:6]
        day = date_str[6:8]
        
        if time_utc is None:
            time_utc = 0.5
        
        iso = f"{year}-{month}-{day}T00:00:00"
        t = Time(iso, format='isot', scale='utc')
        t = t + time_utc / 24.0
        
        return t.iso[:4], t.iso[5:7], t.iso[8:10], t.iso[11:19]
    
    def format_observation(self, ra, dec, date_str, mag=None, time_utc=None):
        year, month, day, frac = self.to_mpc_date(date_str, time_utc)
        date_field = f"{year} {month} {day}"
        
        ra_h = ra / 15.0
        ra_hh = int(ra_h)
        ra_mm = int((ra_h - ra_hh) * 60)
        ra_ss = ((ra_h - ra_hh) * 60 - ra_mm) * 60
        ra_field = f"{ra_hh:02d} {ra_mm:02d} {ra_ss:05.2f}"
        
        dec_sign = '+' if dec >= 0 else '-'
        dec = abs(dec)
        dec_dd = int(dec)
        dec_mm = int((dec - dec_dd) * 60)
        dec_ss = ((dec - dec_dd) * 60 - dec_mm) * 60
        dec_field = f"{dec_sign}{dec_dd:02d} {dec_mm:02d} {dec_ss:04.1f}"
        
        mag_field = "    " if mag is None else f"{mag:4.1f}"
        obs_field = self.observatory_code.ljust(3)
        
        line = (
            "     "
            f"{date_field:<17}"
            f"{ra_field:<13}"
            f"{dec_field:<13}"
            "         "
            "         "
            f"{mag_field:<6}"
            "      "
            f"{obs_field}"
        )
        return line
    
    def generate_mpc_file(self, tracks, output_path, temporary_designation='ASTRA'):
        lines = []
        lines.append("# AstraFilter - MPC Submission File")
        lines.append(f"# Generated: {datetime.now().isoformat()}")
        lines.append(f"# Observatory: {self.observatory_code}")
        lines.append(f"# Number of tracks: {len(tracks)}")
        lines.append("")
        
        for i, track in enumerate(tracks):
            lines.append(f"# --- Track {i+1} ---")
            lines.append(f"# R2_x={track.get('r2_x', 'N/A')}, R2_y={track.get('r2_y', 'N/A')}")
            lines.append(f"# Velocity: {track.get('velocity', 0):.2f} px/day")
            lines.append(f"# N dates: {track.get('n_dates', 0)}")
            
            for p in track['points']:
                if p.get('ra') is None:
                    continue
                line = self.format_observation(ra=p['ra'], dec=p['dec'], date_str=p['date'])
                lines.append(line)
            lines.append("")
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        return output_path
    
    def generate_csv_report(self, tracks, output_path):
        import pandas as pd
        rows = []
        for i, track in enumerate(tracks):
            for p in track['points']:
                rows.append({
                    'track_id': i + 1,
                    'date': p.get('date'),
                    'x': p.get('x'),
                    'y': p.get('y'),
                    'length': p.get('length'),
                    'linearity': p.get('linearity'),
                    'ra': p.get('ra'),
                    'dec': p.get('dec'),
                    'r2_x': track.get('r2_x'),
                    'r2_y': track.get('r2_y'),
                    'velocity_px_per_day': track.get('velocity'),
                })
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        return output_path
