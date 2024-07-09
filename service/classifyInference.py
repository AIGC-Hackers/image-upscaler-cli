
import os
from functools import lru_cache
from PIL import Image
import numpy as np
from transformers import pipeline

labels_text = 'human,anime,landscope'

upscale_dic = {
       'general': 'RealESRGAN_General_x4_v3',
       'anime': 'realesrgan-x4plus-anime',
       'landscope': 'remacri',
       'human': 'GFPGANv1.4.pth'
}

# 禁止并行拉取tokenizer库，避免内核态死锁现象
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@lru_cache(maxsize=1)
def get_pipeline():
    return pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")

def classify_inference(image_path, upscale_label=''):
    # 使用上下文管理器来确保图像文件被正确关闭
    with Image.open(image_path) as img_pil:
        # 直接转换为RGB，避免额外的转换步骤
        img_pil = img_pil.convert('RGB')

    # 使用缓存的pipeline
    pipe = get_pipeline()

    # 假设 labels_text 和 upscale_dic 是全局变量或从某处导入
    labels = labels_text.split(",")

    res = pipe(images=img_pil,  # 直接使用numpy数组
               candidate_labels=labels,
               hypothesis_template="This is a photo of a {}")

    # 使用列表推导式和 next() 函数来简化标签选择
    upscale_label = next((dic["label"] for dic in res if round(dic["score"] * 100) >= 80), "general")

    return upscale_dic[upscale_label]

    # 如果需要日志记录，可以取消下面的注释
    # logger.info({dic["label"]: dic["score"] for dic in res})