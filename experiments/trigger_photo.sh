#!/bin/bash
# 数据收集触发脚本
# 用法: trigger_photo.sh "layer|filename"
# 示例: trigger_photo.sh "100|test.gcode"

# 解析参数（用 | 分隔）
IFS='|' read -r LAYER FILENAME <<< "$1"

# 调试日志
echo "DEBUG: '$1' -> LAYER='${LAYER}' FILENAME='${FILENAME}'" >> /tmp/trigger_photo_debug.log

curl -s -X POST http://10.168.1.118:5000/capture \
  -H "Content-Type: application/json" \
  -d "{\"layer\": ${LAYER}, \"filename\": \"${FILENAME}\"}"
