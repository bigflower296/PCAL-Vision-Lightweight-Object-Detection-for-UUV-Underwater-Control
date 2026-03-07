import sys
import os
import torch

# 挂载当前目录
sys.path.insert(0, os.getcwd())

from ultralytics import YOLO


def get_model_stats(model_config):
    """加载模型配置并返回统计数据"""
    print(f"DEBUG: 正在加载 {model_config} ...")

    # 1. 直接加载，不加 try-except
    model = YOLO(model_config)

    # 2. 获取 info
    # 注意：这里我们开启 verbose=True，确保能看到底层的报错
    results = model.info(verbose=True)

    if not results:
        raise ValueError(f"❌ 致命错误：{model_config} model.info() 返回为空！请检查 YAML 文件格式。")

    # results 格式: (layers, params, gradients, flops)
    return results[3], results[1]


def save_report(content, filename="model_comparison_report.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n✅ 对比报告已保存至: {filename}")


def main():
    print("=" * 60)
    print("🚀 PC-YOLO vs. YOLOv8s: 性能对比与训练启动脚本")
    print("=" * 60)

    baseline_cfg = 'yolov8s.yaml'
    ours_cfg = 'yolov8_pc.yaml'

    # --- 步骤 1: 获取原版数据 ---
    print(f"\n[1/3] 分析基准模型 ({baseline_cfg})...")
    # 如果这一步报错，说明根目录下缺少 yolov8s.yaml
    flops_base, params_base = get_model_stats(baseline_cfg)

    # --- 步骤 2: 获取改进版数据 ---
    print(f"\n[2/3] 分析改进模型 ({ours_cfg})...")
    # 如果这一步报错，说明 yolov8_pc.yaml 写错了
    flops_ours, params_ours = get_model_stats(ours_cfg)

    # --- 步骤 3: 计算与打印 ---
    flops_drop = (flops_base - flops_ours) / flops_base * 100
    params_drop = (params_base - params_ours) / params_base * 100

    report = f"""
============================================================
               轻量化性能对比报告 (Model Comparison)
============================================================
模型 (Model)        | 计算量 (GFLOPs) ↓ | 参数量 (Params) ↓
------------------------------------------------------------
YOLOv8s (基准)      | {flops_base:<15.2f} | {params_base / 1e6:<15.2f}M
PC-YOLO (本文)      | {flops_ours:<15.2f} | {params_ours / 1e6:<15.2f}M
------------------------------------------------------------
优化幅度 (Reduction)| ⬇️ {flops_drop:.2f}%          | ⬇️ {params_drop:.2f}%
============================================================
"""
    print(report)
    save_report(report)

    # --- 步骤 4: 训练 ---
    print(f"\n[3/3] 开始训练改进模型 ({ours_cfg})...")

    device = 0 if torch.cuda.is_available() else 'cpu'
    model = YOLO(ours_cfg)
    model.train(
        data='E:/DeskTop/dataset/rov_yolo/data.yaml',
        epochs=100,
        imgsz=640,
        device=device,
        batch=16,
        workers=8,
        name='pc_yolo_training',
        exist_ok=True,
        patience=50,
        optimizer='auto',
        verbose=True
    )


if __name__ == '__main__':
    main()