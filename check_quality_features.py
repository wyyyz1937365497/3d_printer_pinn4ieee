import h5py
import glob

# 查找最新的文件
files = glob.glob('data_simulation_*/*.mat')
if not files:
    print('没有找到数据文件')
    exit(1)

# 使用第一个文件检查
filepath = files[0]
print(f'检查文件: {filepath}')
print()

with h5py.File(filepath, 'r') as f:
    if 'simulation_data' in f:
        sim_data = f['simulation_data']
        field_names = list(sim_data.keys())

        print(f'文件中的字段数量: {len(field_names)}')
        print()

        # 检查质量特征
        quality_features = ['adhesion_ratio', 'internal_stress', 'porosity',
                           'dimensional_accuracy', 'quality_score']
        print('质量特征检查:')
        all_present = True
        for feat in quality_features:
            if feat in field_names:
                sample_data = sim_data[feat]
                if hasattr(sample_data, 'shape'):
                    print(f'  ✓ {feat}: 存在 (shape={sample_data.shape})')
                else:
                    print(f'  ✓ {feat}: 存在')
            else:
                print(f'  ✗ {feat}: 缺失')
                all_present = False

        print()
        if all_present:
            print('✅ 所有5个质量特征都存在！')
            print()
            print('下一步：使用项目原始训练流程')
            print('  python experiments/quick_train_simulation.py \\')
            print('      --data_dir data_simulation_* --epochs 100 --batch_size 64')
        else:
            print('❌ 缺少质量特征')
            print()
            print('可用的输入特征:')
            input_features = ['x_ref', 'y_ref', 'z_ref', 'vx_ref', 'vy_ref', 'vz_ref',
                             'T_nozzle', 'T_interface', 'F_inertia_x', 'F_inertia_y',
                             'cooling_rate', 'layer_num']
            for feat in input_features:
                if feat in field_names:
                    print(f'  ✓ {feat}')
                else:
                    print(f'  ✗ {feat}')
