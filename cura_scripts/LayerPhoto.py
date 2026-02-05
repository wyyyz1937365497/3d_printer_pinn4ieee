import re
from ..Script import Script

class LayerPhoto(Script):
    def __init__(self):
        super().__init__()

    def getSettingData(self):
        return {
            "name": "Layer Photo Trigger",
            "key": "LayerPhoto",
            "metadata": {},
            "settings": {
                "trigger_macro": {
                    "label": "Trigger Macro Name",
                    "description": "The G-code macro to call on each layer",
                    "type": "str",
                    "default_value": "LAYER_COMPLETE"
                },
                "skip_first_layers": {
                    "label": "Skip First N Layers",
                    "description": "Number of layers to skip before triggering photos",
                    "type": "int",
                    "default_value": 0,
                    "minimum_value": 0
                },
                "photo_interval": {
                    "label": "Photo Every N Layers",
                    "description": "Trigger photo every N layers (1 = every layer)",
                    "type": "int",
                    "default_value": 1,
                    "minimum_value": 1
                }
            }
        }

    def execute(self, data):
        trigger_macro = self.getSettingValueByKey("trigger_macro")
        skip_first = self.getSettingValueByKey("skip_first_layers")
        interval = self.getSettingValueByKey("photo_interval")

        # 匹配 ;LAYER:数字 格式
        layer_pattern = re.compile(r'^;LAYER:(\d+)')

        for layer_number, layer in enumerate(data):
            # 跳过设置层
            if layer_number == 0:
                data[layer_number] = layer
                continue

            lines = layer.split("\n")
            modified_lines = []

            for line in lines:
                modified_lines.append(line)

                # 检测 ;LAYER:X 注释
                match = layer_pattern.match(line.strip())
                if match:
                    layer_num = int(match.group(1))

                    # 跳过前N层
                    if layer_num < skip_first:
                        continue

                    # 按间隔拍照
                    if layer_num % interval == 0:
                        modified_lines.append(f"; LayerPhoto: Triggering {trigger_macro}")
                        modified_lines.append(trigger_macro)

            data[layer_number] = "\n".join(modified_lines)

        return data
