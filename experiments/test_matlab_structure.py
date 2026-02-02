"""
测试脚本 - 检查MATLAB保存的.mat文件结构
"""
import matlab.engine
import numpy as np

def test_matlab_structure():
    """检查MATLAB返回的数据结构"""
    print("="*70)
    print("测试MATLAB数据结构")
    print("="*70)

    # 启动MATLAB
    print("\n启动MATLAB...")
    matlab = matlab.engine.start_matlab()

    # 加载.mat文件
    mat_file = 'results/realtime_correction/realtime_correction_layer_25.mat'
    print(f"\n加载文件: {mat_file}")

    mat_data = matlab.load(mat_file)

    print("\n" + "="*70)
    print("MATLAB数据结构分析")
    print("="*70)

    def print_structure(obj, indent=0):
        """递归打印MATLAB对象结构"""
        prefix = "  " * indent

        if isinstance(obj, dict):
            for key, value in obj.items():
                print(f"{prefix}{key}: {type(value).__name__}")
                print_structure(value, indent + 1)
        elif hasattr(obj, '_data'):
            print(f"{prefix}[MATLAB array] shape: {np.array(obj._data).shape}")
        elif hasattr(obj, '__dict__'):
            # 尝试获取所有属性
            try:
                if hasattr(obj, '_nargout'):
                    print(f"{prefix}[Function or Method]")
                    return

                # 获取属性名
                attr_list = dir(obj)
                field_names = [attr for attr in attr_list
                              if not attr.startswith('_') and not callable(getattr(obj, attr, None))]

                for field in field_names[:10]:  # 只显示前10个字段
                    try:
                        field_obj = getattr(obj, field)
                        print(f"{prefix}{field}: {type(field_obj).__name__}")
                        if indent < 2:  # 限制递归深度
                            print_structure(field_obj, indent + 1)
                    except Exception as e:
                        print(f"{prefix}{field}: [Error: {e}]")
            except Exception as e:
                print(f"{prefix}[Error accessing attributes: {e}]")
        else:
            print(f"{prefix}{type(obj).__name__}: {str(obj)[:100]}")

    # 分析结构
    print_structure(mat_data)

    # 详细检查results
    print("\n" + "="*70)
    print("详细检查 'results' 对象")
    print("="*70)

    if 'results' in mat_data:
        results = mat_data['results']
        print(f"\nresults类型: {type(results)}")
        print(f"results属性: {dir(results)[:20]}")

        # 尝试访问time
        if hasattr(results, 'time'):
            time_data = results.time
            print(f"\ntime属性类型: {type(time_data)}")
            if hasattr(time_data, '_data'):
                print(f"time._data: {np.array(time_data._data)[:5]}")
            print(f"time dir: {dir(time_data)}")

    # 关闭MATLAB
    matlab.quit()
    print("\n" + "="*70)
    print("测试完成")
    print("="*70)

if __name__ == '__main__':
    test_matlab_structure()
