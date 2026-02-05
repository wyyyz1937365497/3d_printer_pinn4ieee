#!/bin/bash
# 数据收集触发脚本
# 用法: trigger_photo.sh <layer> <filename>

LAYER=$1
FILENAME=$2

curl -s -X POST http://10.168.1.118:5000/capture \
  -H "Content-Type: application/json" \
  -d "{\"layer\": ${LAYER}, \"filename\": \"${FILENAME}\"}"
