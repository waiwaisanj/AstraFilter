
"""
AstraFilter - 主流程 v2
整合六个模块：下载 → 检测 → 验证 → 多源验证 → 轨迹分析 → MPC 提交
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime


class AstraFilterPipeline:
    """AstraFilter 主流程"""

    def __init__(self, email, password, work_dir):
        self.email = email
        self.password = password
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)
        os.makedirs(f"{work_dir}/data", exist_ok=True)
        os.makedirs(f"{work_dir}/output", exist_ok=True)

    def run(self, ra, dec, start_date, end_date,
            max_download=30, method="traditional",
            model_path=None, validate_multi_source=True,
            analyze_tracks=True):
        """
        运行完整流程

        参数:
            ra, dec: 天区中心
            start_date, end_date: 日期范围
            max_download: 最多下载图像数
            method: 检测方法
            model_path: U-Net 模型路径
            validate_multi_source: 是否使用多源验证
            analyze_tracks: 是否做轨迹分析
        """
        from downloader import ZTFDownloader
        from detector import StripeDetector
        from verifier import CandidateVerifier
        from submitter import MPCSubmitter

        print("=" * 60)
        print("  AstraFilter v2 - 六模块自动化流程")
        print("=" * 60)

        # ===== [1/6] 下载 =====
        print("\n[1/6] 数据下载")
        print("-" * 60)

        downloader = ZTFDownloader(
            email=self.email, password=self.password,
            data_dir=f"{self.work_dir}/data"
        )

        meta = downloader.query(ra, dec, start_date, end_date)
        print(f"查到 {len(meta)} 条观测记录")

        if len(meta) == 0:
            return {"status": "error", "message": "没有观测数据"}

        field, ccdid, subset = downloader.pick_best_field(meta, min_observations=5)
        if field is None:
            return {"status": "error", "message": "没有合适的 field+ccdid 组合"}

        print(f"选定 field={field}, ccdid={ccdid}, 共 {len(subset)} 条")
        n_downloaded, meta_df = downloader.download_batch(subset, max_count=max_download)
        print(f"下载完成: {n_downloaded} 张")

        if len(meta_df) == 0:
            return {"status": "error", "message": "下载失败"}

        # ===== [2/6] 检测 =====
        print("\n[2/6] 条纹检测")
        print("-" * 60)

        detector = StripeDetector(method=method, model_path=model_path)
        candidates_csv = f"{self.work_dir}/output/candidates.csv"
        candidates_df = detector.detect_batch(meta_df, output_csv=candidates_csv)
        print(f"检测完成: {len(candidates_df)} 个候选体")

        if len(candidates_df) == 0:
            return {
                "status": "no_candidates",
                "message": "没有检测到候选体",
                "n_images": n_downloaded,
            }

        # ===== [3/6] 多帧验证 =====
        print("\n[3/6] 多帧验证")
        print("-" * 60)

        verifier = CandidateVerifier()
        good_tracks, summary = verifier.verify(candidates_df, f"{self.work_dir}/data")

        print(f"聚类数: {summary["n_clusters"]}")
        print(f"可拟合轨迹数: {summary["n_tracks"]}")
        print(f"符合筛选条件的轨迹: {summary["n_good"]}")

        if len(good_tracks) == 0:
            return {
                "status": "no_tracks",
                "message": "没有找到符合运动规律的轨迹",
                "n_images": n_downloaded,
                "n_candidates": len(candidates_df),
                "summary": summary,
                "outputs": {"candidates_csv": candidates_csv},
            }

        # ===== [4/6] 多源验证 =====
        print("\n[4/6] 多源验证 (MPChecker + SkyBoT + JPL Horizons)")
        print("-" * 60)

        if validate_multi_source:
            from multi_validator import MultiValidator
            multi_validator = MultiValidator()

            for i, t in enumerate(good_tracks):
                ra_i = t.get("ra")
                dec_i = t.get("dec")
                date_i = t["points"][0]["date"] if t["points"] else None

                if ra_i is None or dec_i is None or date_i is None:
                    t["multi_validation"] = {"verdict": "skip", "message": "坐标缺失"}
                    continue

                print(f"\n轨迹 {i+1}: RA={ra_i:.4f}, Dec={dec_i:.4f}, date={date_i}")
                result = multi_validator.cross_validate(ra_i, dec_i, date_i)
                print(f"  判定: {result["verdict"]}")
                print(f"  说明: {result["message"]}")
                for source, r in result["results"].items():
                    print(f"    {source}: {r["status"]} (数量: {r["n"]})")

                t["multi_validation"] = result

        # ===== [5/6] 轨迹分析 =====
        print("\n[5/6] 轨迹分析")
        print("-" * 60)

        if analyze_tracks:
            from track_analyzer import TrackAnalyzer
            analyzer = TrackAnalyzer()

            for i, t in enumerate(good_tracks):
                # 把像素位置转成 RA/Dec
                observations = []
                for p in t["points"]:
                    try:
                        fpath = os.path.join(self.work_dir, "data", p["filename"])
                        ra_p, dec_p = verifier.pixel_to_radec(fpath, p["x"], p["y"])
                        observations.append({
                            "date": p["date"],
                            "ra": ra_p,
                            "dec": dec_p,
                        })
                    except Exception:
                        continue

                if len(observations) >= 2:
                    analysis = analyzer.analyze(observations)
                    t["analysis"] = analysis
                    print(f"\n轨迹 {i+1}:")
                    print(f"  平均速度: {analysis["average_velocity_deg_per_day"]:.4f} 度/天")
                    print(f"  分类: {analysis["classification"]}")

        # ===== [6/6] MPC 提交 =====
        print("\n[6/6] 生成 MPC 提交文件")
        print("-" * 60)

        submitter = MPCSubmitter()
        mpc_path = f"{self.work_dir}/output/mpc_submission.txt"
        submitter.generate_mpc_file(good_tracks, mpc_path)
        print(f"MPC 文件: {mpc_path}")

        csv_path = f"{self.work_dir}/output/tracks_report.csv"
        submitter.generate_csv_report(good_tracks, csv_path)
        print(f"CSV 报告: {csv_path}")

        # ===== 完成 =====
        print("\n" + "=" * 60)
        print("  流程完成")
        print("=" * 60)
        print(f"下载图像: {n_downloaded} 张")
        print(f"候选体总数: {len(candidates_df)}")
        print(f"符合轨迹的候选体: {len(good_tracks)}")
        print(f"输出目录: {self.work_dir}/output")

        return {
            "status": "success",
            "n_images": n_downloaded,
            "n_candidates": len(candidates_df),
            "n_good_tracks": len(good_tracks),
            "tracks": good_tracks,
            "summary": summary,
            "outputs": {
                "candidates_csv": candidates_csv,
                "mpc_file": mpc_path,
                "tracks_report": csv_path,
            },
        }
