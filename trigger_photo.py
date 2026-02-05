"""
手动触发拍照的Python脚本

用法:
    python trigger_photo.py [layer_height_in_microns]

示例:
    python trigger_photo.py 10200    # Z=10.2mm
    python trigger_photo.py 0        # Z=0mm
"""

import requests
import sys
import json

# 配置
CAPTURE_API = "http://10.168.1.118:5000/capture"

def trigger_photo(layer_um):
    """触发拍照

    Args:
        layer_um: 层高度（微米）
    """
    payload = {
        "layer": layer_um,
        "filename": "manual_trigger"
    }

    print(f"触发拍照:")
    print(f"  URL: {CAPTURE_API}")
    print(f"  层高: {layer_um} um ({layer_um/1000:.3f} mm)")
    print()

    try:
        response = requests.post(
            CAPTURE_API,
            json=payload,
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ 成功!")
            print(f"  响应: {json.dumps(result, indent=2, ensure_ascii=False)}")

            if result.get('success'):
                print(f"\n数据已保存:")
                print(f"  图像: {result.get('data', {}).get('image_path', 'N/A')}")
                print(f"  点数: {result.get('data', {}).get('stats', {}).get('n_points', 'N/A')}")
                print(f"  RMS: {result.get('data', {}).get('stats', {}).get('rms_um', 'N/A')} um")
            else:
                warning = result.get('warning', result.get('error', 'Unknown'))
                print(f"⚠️ 警告: {warning}")
        else:
            print(f"❌ HTTP错误: {response.status_code}")
            print(f"  {response.text}")

    except requests.exceptions.ConnectionError:
        print(f"❌ 连接失败: 无法连接到 {CAPTURE_API}")
        print(f"  请确保Flask服务正在运行")
        print(f"  启动命令: python experiments/auto_data_collector_existing.py ...")
    except Exception as e:
        print(f"❌ 错误: {e}")

    print()
    print(f"查看日志: type data\\collection.log")

if __name__ == '__main__':
    # 解析命令行参数
    if len(sys.argv) > 1:
        try:
            layer_um = int(sys.argv[1])
        except ValueError:
            print(f"错误: 层高必须是整数（微米）")
            print(f"用法: python trigger_photo.py <layer_um>")
            sys.exit(1)
    else:
        # 默认Z=0
        layer_um = 0
        print(f"未指定层高，使用默认值: {layer_um} um")

    trigger_photo(layer_um)
