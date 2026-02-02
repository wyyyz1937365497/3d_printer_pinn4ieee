"""
简单测试脚本 - 检查MATLAB .mat文件结构
"""
import matlab.engine as matlab_engine
import numpy as np

def main():
    print("Starting MATLAB...")
    eng = matlab_engine.start_matlab()

    mat_file = 'results/realtime_correction/realtime_correction_layer_25.mat'
    print(f"Loading file: {mat_file}")

    mat_data = eng.load(mat_file)

    print("\n=== Top level keys ===")
    print(f"Type: {type(mat_data)}")
    print(f"Keys: {list(mat_data.keys()) if hasattr(mat_data, 'keys') else 'Not a dict'}")

    print("\n=== Checking 'results' ===")
    if 'results' in mat_data:
        results = mat_data['results']
        print(f"Type of results: {type(results)}")

        if isinstance(results, dict):
            print(f"\nResults keys: {list(results.keys())[:20]}")

            # Check each key
            for key in list(results.keys())[:10]:
                value = results[key]
                print(f"\n{key}:")
                print(f"  Type: {type(value)}")

                if isinstance(value, dict):
                    print(f"  Sub-keys: {list(value.keys())[:10]}")
                    for subkey in list(value.keys())[:3]:
                        subvalue = value[subkey]
                        print(f"    {subkey}: type={type(subvalue).__name__}")
                        if hasattr(subvalue, 'shape'):
                            print(f"      shape={subvalue.shape}")
                elif hasattr(value, '_data'):
                    data = np.array(value._data)
                    print(f"  shape: {data.shape}")
                    print(f"  first 5: {data[:5]}")

    eng.quit()
    print("\nDone!")

if __name__ == '__main__':
    main()
