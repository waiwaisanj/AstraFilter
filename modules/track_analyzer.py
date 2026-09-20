
"""
AstraFilter - 轨迹分析模块
从多帧位置计算运动速率、方向、预测未来位置
"""

import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u


class TrackAnalyzer:
    """轨迹分析器：从多个观测点计算运动参数"""
    
    def __init__(self):
        pass
    
    def angular_distance(self, ra1, dec1, ra2, dec2):
        """计算两个天球坐标之间的角距离（度）"""
        c1 = SkyCoord(ra=ra1*u.degree, dec=dec1*u.degree)
        c2 = SkyCoord(ra=ra2*u.degree, dec=dec2*u.degree)
        return c1.separation(c2).degree
    
    def position_angle(self, ra1, dec1, ra2, dec2):
        """计算从点 1 到点 2 的位置角（度，从北向东）"""
        c1 = SkyCoord(ra=ra1*u.degree, dec=dec1*u.degree)
        c2 = SkyCoord(ra=ra2*u.degree, dec=dec2*u.degree)
        return c1.position_angle(c2).degree
    
    def analyze(self, observations):
        """
        分析一组观测点
        
        参数:
            observations: list of dict, 每个含 date (YYYY-MM-DD), ra, dec
        
        返回:
            dict 包含运动速率、方向、拟合质量
        """
        if len(observations) < 2:
            return {"error": "需要至少 2 个观测点"}
        
        # 按日期排序
        obs = sorted(observations, key=lambda x: x["date"])
        
        # 转成 MJD
        times = []
        ras = []
        decs = []
        for o in obs:
            t = Time(o["date"], format="iso")
            times.append(t.mjd)
            ras.append(o["ra"])
            decs.append(o["dec"])
        
        times = np.array(times)
        ras = np.array(ras)
        decs = np.array(decs)
        
        days_rel = times - times.min()
        
        # ===== 方法 1：整体平均速度 =====
        if len(obs) >= 2:
            first = obs[0]
            last = obs[-1]
            total_days = times[-1] - times[0]
            total_dist = self.angular_distance(
                first["ra"], first["dec"], last["ra"], last["dec"])
            avg_velocity = total_dist / total_days if total_days > 0 else 0
            avg_pa = self.position_angle(
                first["ra"], first["dec"], last["ra"], last["dec"])
        else:
            avg_velocity = 0
            avg_pa = 0
        
        # ===== 方法 2：线性拟合 =====
        if len(obs) >= 3:
            # RA 在赤道附近可以用 cos(dec) 修正
            cos_dec = np.cos(np.radians(np.mean(decs)))
            
            # 用线性回归拟合 ra*cos(dec) 和 dec 随时间变化
            coef_ra = np.polyfit(days_rel, ras * cos_dec, 1)
            coef_dec = np.polyfit(days_rel, decs, 1)
            
            # 预测值
            pred_ra = np.polyval(coef_ra, days_rel) / cos_dec
            pred_dec = np.polyval(coef_dec, days_rel)
            
            # R²
            ss_res_ra = np.sum((ras - pred_ra)**2)
            ss_tot_ra = np.sum((ras - ras.mean())**2)
            r2_ra = 1 - ss_res_ra / (ss_tot_ra + 1e-10)
            
            ss_res_dec = np.sum((decs - pred_dec)**2)
            ss_tot_dec = np.sum((decs - decs.mean())**2)
            r2_dec = 1 - ss_res_dec / (ss_tot_dec + 1e-10)
            
            # 线性速度（度/天）
            v_ra = coef_ra[0] / cos_dec   # deg/day
            v_dec = coef_dec[0]            # deg/day
            linear_velocity = np.sqrt(v_ra**2 + v_dec**2)
        else:
            r2_ra = None
            r2_dec = None
            linear_velocity = None
            v_ra = None
            v_dec = None
        
        # ===== 预测未来位置（用线性拟合外推） =====
        if len(obs) >= 3:
            # 预测 7 天后的位置
            future_day = days_rel[-1] + 7
            future_ra = np.polyval(coef_ra, future_day) / cos_dec
            future_dec = np.polyval(coef_dec, future_day)
        else:
            future_ra = None
            future_dec = None
        
        result = {
            "n_observations": len(obs),
            "total_days": float(times[-1] - times[0]),
            "total_angular_distance": float(total_dist),
            "average_velocity_deg_per_day": float(avg_velocity),
            "average_position_angle_deg": float(avg_pa),
            "linear_velocity_deg_per_day": float(linear_velocity) if linear_velocity else None,
            "v_ra_deg_per_day": float(v_ra) if v_ra else None,
            "v_dec_deg_per_day": float(v_dec) if v_dec else None,
            "r2_ra": float(r2_ra) if r2_ra else None,
            "r2_dec": float(r2_dec) if r2_dec else None,
            "future_7d_ra": float(future_ra) if future_ra else None,
            "future_7d_dec": float(future_dec) if future_dec else None,
            "observations": obs,
        }
        
        # ===== 分类：可能的天体类型 =====
        v = result["average_velocity_deg_per_day"]
        if v < 0.01:
            result["classification"] = "疑似固定目标（速度太慢）"
        elif v < 0.1:
            result["classification"] = "疑似主带小行星"
        elif v < 1.0:
            result["classification"] = "疑似近地小行星或大型主带天体"
        elif v < 10.0:
            result["classification"] = "疑似快速移动小行星 (FMO)"
        else:
            result["classification"] = "疑似人造卫星或空间碎片"
        
        return result
    
    def format_report(self, analysis):
        """把分析结果格式化成可读的文本报告"""
        lines = []
        lines.append("=" * 60)
        lines.append("轨迹分析报告")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"观测点数: {analysis["n_observations"]}")
        lines.append(f"时间跨度: {analysis["total_days"]:.2f} 天")
        lines.append(f"总角距离: {analysis["total_angular_distance"]:.4f} 度")
        lines.append("")
        lines.append(f"平均速度: {analysis["average_velocity_deg_per_day"]:.4f} 度/天")
        lines.append(f"平均位置角: {analysis["average_position_angle_deg"]:.2f} 度")
        
        if analysis.get("linear_velocity_deg_per_day"):
            lines.append(f"线性拟合速度: {analysis["linear_velocity_deg_per_day"]:.4f} 度/天")
            lines.append(f"  v_RA: {analysis["v_ra_deg_per_day"]:.4f} 度/天")
            lines.append(f"  v_Dec: {analysis["v_dec_deg_per_day"]:.4f} 度/天")
            lines.append(f"  R² (RA): {analysis["r2_ra"]:.4f}")
            lines.append(f"  R² (Dec): {analysis["r2_dec"]:.4f}")
        
        lines.append("")
        lines.append(f"分类: {analysis["classification"]}")
        
        if analysis.get("future_7d_ra"):
            lines.append("")
            lines.append("预测 7 天后位置:")
            lines.append(f"  RA:  {analysis["future_7d_ra"]:.5f} 度")
            lines.append(f"  Dec: {analysis["future_7d_dec"]:.5f} 度")
        
        lines.append("")
        lines.append("观测记录:")
        for o in analysis["observations"]:
            lines.append(f"  {o["date"]}: RA={o["ra"]:.5f}, Dec={o["dec"]:.5f}")
        
        return chr(10).join(lines)


# ===== 测试 =====
if __name__ == "__main__":
    analyzer = TrackAnalyzer()
    
    # 用模拟的 FMO 观测数据测试
    test_obs = [
        {"date": "2023-09-22", "ra": 359.3400, "dec": 0.3000},
        {"date": "2023-09-25", "ra": 359.3466, "dec": 0.3133},
        {"date": "2023-09-27", "ra": 359.3500, "dec": 0.3210},
        {"date": "2023-10-03", "ra": 359.3650, "dec": 0.3480},
    ]
    
    result = analyzer.analyze(test_obs)
    print(analyzer.format_report(result))
