import cv2
import gc
import torch
from pathlib import Path
from typing import Tuple, Optional, Union
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan.utils import GFPGANer
from realesrgan.utils import RealESRGANer
import logging

logger = logging.getLogger(__name__)

def set_realesrgan(model_name: str, model_path: Path) -> RealESRGANer:
    """Set up RealESRGAN model."""
    half = torch.cuda.is_available()
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    return RealESRGANer(
        scale=2,
        model_path=str(model_path / model_name),
        model=model,
        tile=400,
        tile_pad=10,
        pre_pad=0,
        half=half
    )

# 解释：
# 1. 这个函数设置并返回一个 RealESRGAN 模型实例。
# 2. 使用 torch.cuda.is_available() 检查是否可以使用 GPU，如果可用，half 为 True，这可以提高性能。
# 3. 使用 Path 对象处理文件路径，提高了跨平台兼容性。
# 4. 返回类型注解 (-> RealESRGANer) 清晰地表明了函数的返回类型。

def set_figure_face_enhancer(upsampler: RealESRGANer, model: str, model_path: Path) -> GFPGANer:
    """Set up GFPGAN face enhancer."""
    full_model_path = model_path / model
    arch = 'RestoreFormer' if model == 'RestoreFormer.pth' else 'clean'
    return GFPGANer(model_path=str(full_model_path), upscale=2, arch=arch, channel_multiplier=2, bg_upsampler=upsampler)

# 解释：
# 1. 这个函数设置并返回一个 GFPGAN 人脸增强器实例。
# 2. 使用三元运算符简化了 arch 的选择逻辑。
# 3. upsampler 作为参数传入，允许更灵活的使用。

def load_image(input_img: Path) -> Tuple[Optional[np.ndarray], Optional[str], Optional[str]]:
    """Load and preprocess the input image."""
    img = cv2.imread(str(input_img), cv2.IMREAD_UNCHANGED)
    if img is None:
        logger.error(f"Failed to read image: {input_img}")
        return None, None, "Failed to read image"

    img_mode = 'RGBA' if len(img.shape) == 3 and img.shape[2] == 4 else None
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    h, w = img.shape[:2]
    if max(h, w) > 5000:
        logger.error(f'Origin image {input_img} is too large: {w}x{h}')
        return None, None, "Image is too large"

    return img, img_mode, None

# 解释：
# 1. 这个函数负责加载和预处理图像。
# 2. 使用 Optional 类型提示表示可能返回 None。
# 3. 进行了图像格式和尺寸的检查，增加了代码的健壮性。
# 4. 使用 logger 记录错误，而不是简单的 print 语句，便于调试和监控。

def process_image(img: np.ndarray, face_enhancer: GFPGANer, scale: float) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Process the image using face enhancer and apply scaling."""
    try:
        _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True, weight=0.5)
    except RuntimeError as error:
        logger.error(f'Error during face enhancement: {error}')
        return None, str(error)

    if scale != 2:
        h, w = img.shape[:2]
        interpolation = cv2.INTER_AREA if scale < 2 else cv2.INTER_LANCZOS4
        output = cv2.resize(output, (int(w * scale), int(h * scale)), interpolation=interpolation)

    return output, None

# 解释：
# 1. 这个函数处理图像增强和缩放。
# 2. 使用 try-except 块捕获可能的 RuntimeError，提高了错误处理能力。
# 3. 根据 scale 值选择不同的插值方法，优化了图像质量。

def save_image(output: np.ndarray, outfile: Path, img_mode: Optional[str], save_image_as: str) -> Tuple[Optional[Path], Optional[str]]:
    """Save the processed image."""
    if img_mode == 'RGBA' and save_image_as.lower() != 'png':
        save_image_as = 'png'
        outfile = outfile.with_suffix('.png')
        logger.info(f"Changed output file to: {outfile}")

    try:
        cv2.imwrite(str(outfile), output)
        return outfile, None
    except Exception as e:
        logger.error(f"Failed to save image: {e}")
        return None, str(e)

# 解释：
# 1. 这个函数负责保存处理后的图像。
# 2. 使用 with_suffix 方法更改文件扩展名，这是一个更安全和 Pythonic 的方式。
# 3. 使用 try-except 块捕获可能的异常，增强了错误处理能力。

def figure_inference(
    input_img: Union[str, Path],
    outfile: Union[str, Path],
    models_path: Union[str, Path],
    general_model: str,
    figure_model: str,
    scale: float,
    gpuid: int,
    save_image_as: str
) -> Tuple[Optional[Path], Optional[str]]:
    """Perform image enhancement and upscaling."""
    input_img = Path(input_img)
    outfile = Path(outfile)
    models_path = Path(models_path)

    try:
        img, img_mode, error = load_image(input_img)
        if error:
            return None, error

        bg_upsampler = set_realesrgan(general_model, models_path)
        face_enhancer = set_figure_face_enhancer(bg_upsampler, figure_model, models_path)

        output, error = process_image(img, face_enhancer, scale)
        if error:
            return None, error

        return save_image(output, outfile, img_mode, save_image_as)

    except Exception as error:
        logger.exception(f"An unexpected error occurred during image processing: {error}")
        return None, str(error)

    finally:
        torch.cuda.empty_cache()
        gc.collect()

# 解释：
# 1. 这是主函数，orchestrates 整个图像处理流程。
# 2. 使用 Union 类型提示允许输入 str 或 Path 对象，增加了灵活性。
# 3. 将复杂的处理逻辑分解为多个步骤，每个步骤都有专门的函数处理，提高了可读性和可维护性。
# 4. 使用 try-except-finally 结构确保即使发生错误也能清理资源。
# 5. 在 finally 块中调用 torch.cuda.empty_cache() 和 gc.collect() 确保及时释放 GPU 和系统内存。

if __name__ == "__main__":
    # 这里可以添加一些测试代码
    logging.basicConfig(level=logging.INFO)
    result = figure_inference(
        input_img="path/to/input/image.jpg",
        outfile="path/to/output/image.png",
        models_path="path/to/models",
        general_model="RealESRGAN_x4plus.pth",
        figure_model="GFPGANv1.3.pth",
        scale=2.0,
        gpuid=0,
        save_image_as="png"
    )
    print(result)

# 解释：
# 1. 这个部分允许脚本作为独立程序运行，方便测试和调试。
# 2. 设置了基本的日志配置，有助于在运行时捕获有用的信息。
# 3. 提供了一个使用 figure_inference 函数的示例，帮助理解如何使用这个模块。