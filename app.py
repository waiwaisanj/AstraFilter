
import streamlit as st
import sys
import os
from datetime import datetime

st.set_page_config(page_title="AstraFilter", page_icon="🔭", layout="wide")

st.title("🔭 AstraFilter")
st.subheader("快速移动天体自动检测与验证系统")

st.markdown("从 ZTF 公开数据自动搜索快速移动天体，包含数据下载、AI 检测、多帧验证、多源查询、轨迹分析和 MPC 提交文件生成。")

with st.sidebar:
    st.header("搜索参数")
    ra = st.number_input("赤经 RA (度)", min_value=0.0, max_value=360.0, value=180.0, step=0.1)
    dec = st.number_input("赤纬 Dec (度)", min_value=-90.0, max_value=90.0, value=5.0, step=0.1)

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("开始日期", value=datetime(2023, 10, 1))
    with col2:
        end_date = st.date_input("结束日期", value=datetime(2023, 11, 15))

    max_download = st.slider("最多下载图像数", 5, 50, 20)
    method = st.selectbox("检测方法", ["traditional", "unet", "both"])
    validate_multi = st.checkbox("启用多源验证", value=True)
    analyze_trk = st.checkbox("启用轨迹分析", value=True)

    st.markdown("---")
    st.header("IRSA 凭证")
    email = st.text_input("IRSA 邮箱", value="")
    password = st.text_input("IRSA 密码", value="", type="password")

    run_button = st.button("开始搜索", type="primary")

if run_button:
    if not email or not password:
        st.error("请填写 IRSA 邮箱和密码")
    else:
        st.info("这是在线演示版本。实际计算在 Google Colab 中执行，这里展示的是预置的真实结果。")
        st.markdown("---")
        st.header("搜索结果（预置演示）")

        st.success("找到 1 条符合运动规律的轨迹")

        col1, col2, col3 = st.columns(3)
        col1.metric("下载图像", "20")
        col2.metric("候选体", "16")
        col3.metric("符合轨迹", "1")

        with st.expander("轨迹 1 - 4 个日期", expanded=True):
            st.markdown("**线性拟合**")
            st.write("R²_x = 0.9812")
            st.write("R²_y = 0.9667")
            st.write("像素速度 = 2.8 px/day")

            st.markdown("**多源验证**")
            st.success("MPChecker: clear (0 个已知天体)")
            st.success("SkyBoT: clear (0 个已知天体)")
            st.success("JPL Horizons: clear (0 个已知天体)")

            st.markdown("**轨迹分析**")
            st.write("平均速度: 0.0049 度/天")
            st.write("分类: 疑似固定目标")
            st.write("预测 7 天后: RA=359.3805°, Dec=0.3783°")

            import pandas as pd
            demo_data = pd.DataFrame({
                "date": ["2023-10-29", "2023-10-30", "2023-10-31", "2023-11-08"],
                "x": [1178, 1174, 1186, 1193],
                "y": [554, 562, 558, 568],
                "length": [15, 21, 13, 20],
                "linearity": [8.9, 5.8, 8.3, 8.1],
            })
            st.dataframe(demo_data)

        st.markdown("---")
        st.subheader("已发现的两个未编目候选体")

        st.markdown("**候选体 1（M31 天区）**")
        st.write("坐标: RA=10.645666°, Dec=40.703490°")
        st.write("检测日期: 2023-09-07")
        st.write("条纹长度: 约 83 像素")
        st.write("MPChecker: 检查 1,455,742 个已知天体，无匹配")
        st.write("多源验证: MPChecker + SkyBoT + JPL Horizons 三方均无已知天体")

        st.markdown("**候选体 2（黄道面天区）**")
        st.write("坐标: RA=359.3466°, Dec=0.3133°")
        st.write("检测日期: 2023-09-25")
        st.write("置信度: 0.955")
        st.write("多源验证: 三方均无已知天体")

else:
    st.markdown("---")
    st.subheader("AstraFilter 六模块架构")
    st.markdown("1. 数据下载 — 从 IRSA 下载 ZTF 差分图像")
    st.markdown("2. 条纹检测 — 传统图像处理 + U-Net 神经网络")
    st.markdown("3. 多帧验证 — 跨日期聚类 + 线性轨迹拟合")
    st.markdown("4. 多源验证 — MPChecker + SkyBoT + JPL Horizons 三方交叉查询")
    st.markdown("5. 轨迹分析 — 计算运动速度、方向、分类、预测未来位置")
    st.markdown("6. MPC 提交 — 自动生成 80 列格式提交文件")
    st.markdown("")

    st.subheader("使用说明")
    st.markdown("1. 输入天区坐标（例如 RA=180°, Dec=5°）")
    st.markdown("2. 选择日期范围（建议 2-4 周）")
    st.markdown("3. 选择最多下载图像数（建议 20-30）")
    st.markdown("4. 填入 IRSA 账号（在 irsa.ipac.caltech.edu 免费注册）")
    st.markdown("5. 点击开始搜索")

    st.markdown("---")
    st.subheader("已发现的两个未编目候选体")
    st.markdown("**候选体 1（M31 天区）**: RA=10.645666°, Dec=40.703490°")
    st.markdown("**候选体 2（黄道面天区）**: RA=359.3466°, Dec=0.3133°")

    st.markdown("---")
    st.subheader("技术栈")
    st.markdown("- 数据来源: NASA/IPAC IRSA (ZTF)")
    st.markdown("- 检测方法: 传统图像处理 + PyTorch U-Net")
    st.markdown("- 验证源: MPChecker + SkyBoT + JPL Horizons")
    st.markdown("- 前端: Streamlit")
