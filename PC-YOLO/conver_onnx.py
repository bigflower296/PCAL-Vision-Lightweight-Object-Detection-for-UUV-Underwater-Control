import sys
import subprocess
import os


def force_install_requirements():
    """强制安装所需的依赖"""
    packages = ['onnx', 'onnxruntime', 'onnxslim']

    for package in packages:
        try:
            print(f"正在安装/更新 {package}...")
            # 卸载然后重新安装
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", package])
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except Exception as e:
            print(f"处理 {package} 时出错: {e}")


def check_environment():
    """检查当前环境"""
    print(f"Python 路径: {sys.executable}")
    print(f"Python 版本: {sys.version}")

    # 检查关键包
    try:
        import onnx
        print(f"ONNX 版本: {onnx.__version__}")
    except ImportError:
        print("ONNX 未正确安装")
        return False

    try:
        import onnxruntime
        print(f"ONNX Runtime 版本: {onnxruntime.__version__}")
    except ImportError:
        print("ONNX Runtime 未正确安装")
        return False

    return True


def convert_with_environment_check(pt_model_path, img_size=640):
    """带环境检查的转换函数"""

    # 首先检查环境
    if not check_environment():
        print("环境检查失败，尝试修复...")
        force_install_requirements()

        # 再次检查
        if not check_environment():
            print("环境修复失败，请手动安装依赖")
            return None

    try:
        from ultralytics import YOLO

        print("正在加载YOLO模型...")
        model = YOLO(pt_model_path)

        onnx_save_path = pt_model_path.replace('.pt', f'_jetson_{img_size}.onnx')

        print(f"正在转换为ONNX格式 (img_size={img_size})...")

        # 使用更简单的导出设置
        model.export(
            format='onnx',
            imgsz=img_size,
            simplify=True,
            opset=12,
            dynamic=False  # 先尝试静态batch
        )

        # 重命名文件
        default_onnx = pt_model_path.replace('.pt', '.onnx')
        if os.path.exists(default_onnx):
            os.rename(default_onnx, onnx_save_path)
            print(f"转换成功! ONNX模型已保存至: {onnx_save_path}")
            return onnx_save_path
        else:
            print("错误: 未找到生成的ONNX文件")
            return None

    except Exception as e:
        print(f"转换失败: {e}")
        return None


if __name__ == "__main__":
    pt_path = "E:/DeskTop/light.pt"
    convert_with_environment_check(pt_path, img_size=640)