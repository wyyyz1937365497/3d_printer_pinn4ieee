@echo off
REM 手动触发拍照的批处理脚本
REM
REM 用法:
REM     trigger_photo.bat [layer_height_in_microns]
REM
REM 示例:
REM     trigger_photo.bat 10200    # Z=10.2mm
REM     trigger_photo.bat 0        # Z=0mm

setlocal

REM 配置
set CAPTURE_API=http://10.168.1.118:5000/capture

REM 解析参数
set LAYER_UM=0
if not "%1"=="" set LAYER_UM=%1

echo 触发拍照:
echo   URL: %CAPTURE_API%
echo   层高: %LAYER_UM% um
echo.

REM 发送HTTP请求
curl -X POST %CAPTURE_API% ^
  -H "Content-Type: application/json" ^
  -d "{\"layer\": %LAYER_UM%, \"filename\": \"manual_trigger\"}" ^
  -w "\nHTTP状态码: %%{http_code}\n"

echo.
echo 查看日志: type data\collection.log
echo.

pause
