@echo off
REM 并行数据收集启动脚本
REM 同时启动3个MATLAB实例处理不同的gcode文件

echo ============================================================
echo 并行数据收集 - 3个MATLAB实例
echo ============================================================
echo.

echo [1/3] 启动 collect_bearing5_remaining.m (30层)
start "MATLAB_Bearing5" matlab -batch "collect_bearing5_remaining"
timeout /t 5 /nobreak >nul

echo [2/3] 启动 collect_nautilus_all.m (56层)
start "MATLAB_Nautilus" matlab -batch "collect_nautilus_all"
timeout /t 5 /nobreak >nul

echo [3/3] 启动 collect_boat_all.m (74层)
start "MATLAB_Boat" matlab -batch "collect_boat_all"
timeout /t 5 /nobreak >nul
echo ============================================================
echo 所有任务已启动！
echo ============================================================
echo.
echo 监控进度:
echo   查看各MATLAB窗口的输出
echo   或检查输出目录:
echo     dir data_simulation_bearing5_PLA_2h27m_sampled_75layers
echo     dir data_simulation_Nautilus_Gears_Plate_PLA_3h36m_sampled_56layers
echo     dir data_simulation_simple_boat5_PLA_4h4m_sampled_74layers
echo.
pause
