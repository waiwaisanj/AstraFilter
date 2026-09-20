
"""
AstraFilter - 多验证源查询模块 (v3)
MPChecker + SkyBoT + JPL Horizons (修正版)
"""

import requests
import numpy as np
import re
from astropy.time import Time


class MultiValidator:
    def __init__(self):
        self.sources = ["MPChecker", "SkyBoT", "JPL Horizons"]

    def check_mpchecker(self, ra, dec, date_str, radius_arcmin=5):
        date_str = str(date_str).replace("-", "")
        url = "https://www.minorplanetcenter.net/cgi-bin/checkmp.cgi"
        params = {
            "year": date_str[:4], "month": date_str[4:6], "day": date_str[6:8],
            "which": "pos", "ra": ra, "dec": dec,
            "r": str(radius_arcmin), "limit": "24.0", "oc": "500",
            "sort": "d", "mot": "h", "tmot": "s", "pdes": "u",
            "needed": "f", "ps": "n", "type": "p",
        }
        try:
            r = requests.get(url, params=params, timeout=30)
            text = r.text
            if "No known minor planets" in text or "no known" in text.lower():
                return {"status": "clear", "n": 0, "source": "MPChecker"}
            object_pattern = re.findall(r"\(\d{4,5}\)|\d{4}\s+[A-Z]{2}\d+", text)
            if len(object_pattern) > 0:
                return {"status": "found", "n": len(object_pattern), "source": "MPChecker"}
            return {"status": "clear", "n": 0, "source": "MPChecker"}
        except Exception as e:
            return {"status": "error", "n": -1, "message": str(e), "source": "MPChecker"}

    def check_skybot(self, ra, dec, date_str, radius_arcmin=5):
        date_str = str(date_str).replace("-", "")
        iso = date_str[:4] + "-" + date_str[4:6] + "-" + date_str[6:8] + "T12:00:00"
        epoch = Time(iso, format="isot", scale="utc").jd
        url = "https://ssp.imcce.fr/webservices/skybot/api/conesearch.php"
        params = {
            "-c": str(ra) + "," + str(dec),
            "-ep": str(epoch),
            "-mime": "votable",
            "-rs": str(radius_arcmin / 60.0),
        }
        try:
            r = requests.get(url, params=params, timeout=30)
            text = r.text
            rows = re.findall(r"<TR>(.*?)</TR>", text, re.DOTALL)
            real_rows = [row for row in rows if row.strip() and "<TD>" in row]
            if len(real_rows) == 0:
                return {"status": "clear", "n": 0, "source": "SkyBoT"}
            return {"status": "found", "n": len(real_rows), "source": "SkyBoT"}
        except Exception as e:
            return {"status": "error", "n": -1, "message": str(e), "source": "SkyBoT"}

    def check_horizons(self, ra, dec, date_str, radius_arcmin=5):
        date_str = str(date_str).replace("-", "")
        # 修正：start 和 stop 必须跨至少 1 天
        iso_start = date_str[:4] + "-" + date_str[4:6] + "-" + date_str[6:8]
        # 结束日期加 1 天
        t = Time(iso_start, format="isot")
        iso_stop = (t + 1).iso[:10]

        asteroids = [";1", ";2", ";3", ";4", ";7", ";15", ";324"]
        found = []
        try:
            from astroquery.jplhorizons import Horizons
        except ImportError:
            return {"status": "error", "n": -1,
                    "message": "astroquery 未安装", "source": "JPL Horizons"}

        for ast in asteroids:
            try:
                obj = Horizons(id=ast, location="500@399",
                               epochs={"start": iso_start, "stop": iso_stop, "step": "1d"})
                eph = obj.ephemerides(quantities="1")
                if len(eph) > 0:
                    ast_ra = float(eph["RA"][0])
                    ast_dec = float(eph["DEC"][0])
                    cos_dist = (np.sin(np.radians(dec)) * np.sin(np.radians(ast_dec)) +
                                np.cos(np.radians(dec)) * np.cos(np.radians(ast_dec)) *
                                np.cos(np.radians(ra - ast_ra)))
                    cos_dist = np.clip(cos_dist, -1, 1)
                    dist_arcmin = np.degrees(np.arccos(cos_dist)) * 60
                    if dist_arcmin < radius_arcmin:
                        found.append({"id": ast, "ra": ast_ra, "dec": ast_dec,
                                      "dist_arcmin": float(dist_arcmin)})
            except Exception:
                continue

        if len(found) == 0:
            return {"status": "clear", "n": 0, "source": "JPL Horizons"}
        return {"status": "found", "n": len(found), "objects": found, "source": "JPL Horizons"}

    def cross_validate(self, ra, dec, date_str, radius_arcmin=5):
        results = {}
        results["MPChecker"] = self.check_mpchecker(ra, dec, date_str, radius_arcmin)
        results["SkyBoT"] = self.check_skybot(ra, dec, date_str, radius_arcmin)
        results["JPL Horizons"] = self.check_horizons(ra, dec, date_str, radius_arcmin)

        found_sources = [k for k, v in results.items() if v["status"] == "found"]
        clear_sources = [k for k, v in results.items() if v["status"] == "clear"]
        error_sources = [k for k, v in results.items() if v["status"] == "error"]

        if len(found_sources) > 0:
            verdict = "known"
            message = "在 " + str(len(found_sources)) + " 个源中找到已知天体: " + str(found_sources)
        elif len(clear_sources) == 3:
            verdict = "candidate"
            message = "三个源都显示没有已知天体，是真候选体"
        elif len(clear_sources) >= 2:
            verdict = "candidate"
            message = str(len(clear_sources)) + " 个源清空，视为候选体"
        else:
            verdict = "partial"
            message = str(len(clear_sources)) + " 个源清空，" + str(len(error_sources)) + " 个源出错"

        return {
            "verdict": verdict,
            "message": message,
            "results": results,
            "found_sources": found_sources,
            "clear_sources": clear_sources,
            "error_sources": error_sources,
        }
