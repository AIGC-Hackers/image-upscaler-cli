# 文件路径配置
INPUT_DIR = "input"
OUTPUT_DIR = "output"
MODELS_PATH = "models"

# 图像处理参数
DELIMITER = "4"  # 用于决定初始放大因子的分界线

# 模型名称
FIGURE_MODEL = "figure_model"
FIGURE_PRO_MODEL = "figure_pro_model"
FIGURE_BACKGROUND_MODEL = "figure_background_model"

# 日志配置
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# 其他配置项
DEFAULT_MIME_TYPE = "image/png"
SUPPORTED_SCALES = [2, 3, 4, 8, 16]  # 支持的放大比例

# 可以根据需要添加更多配置项
# 例如：
# MAX_IMAGE_SIZE = 10000000  # 最大处理图像大小（字节）
# TIMEOUT = 300  # 处理超时时间（秒）

COMMANDS_PATH = "bin/upscaling-realesrgan"