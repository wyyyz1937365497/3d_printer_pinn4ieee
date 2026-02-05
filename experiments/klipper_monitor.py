"""
Klipper状态监控器 - 自动触发拍照

功能：定期查询Klipper状态，检测Z高度变化，自动触发拍照

用法:
    python experiments/klipper_monitor.py [--interval 2] [--threshold 0.2]

参数:
    --interval: 检查间隔（秒），默认2秒
    --threshold: 层高变化阈值（mm），默认0.2mm
"""

import time
import requests
import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class KlipperMonitor:
    """Klipper状态监控器"""

    def __init__(self,
                 klipper_api="http://10.168.1.123:19255",
                 capture_api="http://10.168.1.118:5000/capture",
                 interval=2.0,
                 layer_threshold=0.2):
        """
        Args:
            klipper_api: Klipper Moonraker API地址
            capture_api: Flask捕获服务地址
            interval: 检查间隔（秒）
            layer_threshold: 层高变化阈值（mm）
        """
        self.klipper_api = klipper_api.rstrip('/')
        self.capture_api = capture_api
        self.interval = interval
        self.layer_threshold = layer_threshold

        self.last_z = 0.0
        self.layer_count = 0
        self.running = True

    def get_toolhead_position(self):
        """获取当前工具头位置"""
        try:
            url = f"{self.klipper_api}/printer/objects/query?toolhead"
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                data = response.json()
                position = data['result']['status']['toolhead']['position']
                # position = [x, y, z, e]
                return position[2]  # Z坐标
            else:
                print(f"  获取位置失败: HTTP {response.status_code}")
                return None

        except Exception as e:
            print(f"  获取位置错误: {e}")
            return None

    def trigger_capture(self, z_pos):
        """触发拍照"""
        try:
            payload = {
                "layer": int(z_pos * 1000),  # 转换为微米
                "filename": "auto_monitor"
            }

            response = requests.post(
                self.capture_api,
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    stats = result.get('data', {}).get('stats', {})
                    print(f"  ✅ 拍照成功")
                    print(f"     点数: {stats.get('n_points', 'N/A')}")
                    print(f"     RMS: {stats.get('rms_um', 'N/A')} um")
                    self.layer_count += 1
                else:
                    warning = result.get('warning', result.get('error', 'Unknown'))
                    print(f"  ⚠️ {warning}")
            else:
                print(f"  ❌ HTTP错误: {response.status_code}")

        except Exception as e:
            print(f"  ❌ 触发错误: {e}")

    def check_and_capture(self):
        """检查Z高度变化并触发拍照"""
        z_pos = self.get_toolhead_position()

        if z_pos is None:
            return

        # 检测新的层
        z_diff = abs(z_pos - self.last_z)

        if z_diff >= self.layer_threshold and z_pos > 0.01:
            print(f"\n🎯 检测到新层!")
            print(f"  Z高度: {z_pos:.3f} mm")
            print(f"  变化: {z_diff:.3f} mm")

            # 触发拍照
            self.trigger_capture(z_pos)

            self.last_z = z_pos

            print(f"  已收集: {self.layer_count} 层\n")

        # 打印进度（每10次检查一次）
        elif int(z_pos * 100) % 10 == 0 and z_pos > 0:
            print(f"  当前Z: {z_pos:.3f} mm (监控中...)", end='\r')

    def run(self):
        """运行监控循环"""
        print("=" * 60)
        print("Klipper自动拍照监控器")
        print("=" * 60)
        print(f"  Klipper API: {self.klipper_api}")
        print(f"  捕获API: {self.capture_api}")
        print(f"  检查间隔: {self.interval} 秒")
        print(f"  层高阈值: {self.layer_threshold} mm")
        print("=" * 60)
        print("\n监控中... (按Ctrl+C停止)\n")

        try:
            while self.running:
                self.check_and_capture()
                time.sleep(self.interval)

        except KeyboardInterrupt:
            print("\n\n监控已停止")
            print(f"总共收集: {self.layer_count} 层")

    def stop(self):
        """停止监控"""
        self.running = False


def main():
    parser = argparse.ArgumentParser(description='Klipper自动拍照监控器')
    parser.add_argument('--klipper-api', type=str,
                       default='http://10.168.1.123:19255',
                       help='Klipper Moonraker API地址')
    parser.add_argument('--capture-api', type=str,
                       default='http://10.168.1.118:5000/capture',
                       help='Flask捕获服务地址')
    parser.add_argument('--interval', type=float, default=2.0,
                       help='检查间隔（秒）')
    parser.add_argument('--threshold', type=float, default=0.2,
                       help='层高变化阈值（mm）')

    args = parser.parse_args()

    # 创建监控器
    monitor = KlipperMonitor(
        klipper_api=args.klipper_api,
        capture_api=args.capture_api,
        interval=args.interval,
        layer_threshold=args.threshold
    )

    # 运行监控
    monitor.run()


if __name__ == '__main__':
    main()
