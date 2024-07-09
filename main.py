import argparse
import logging
import os
import time
import uuid
from urllib.parse import urlparse

from PIL import Image

from config import INPUT_DIR, OUTPUT_DIR, MODELS_PATH, DELIMITER
from config import FIGURE_MODEL, FIGURE_PRO_MODEL, FIGURE_BACKGROUND_MODEL
from service.classifyInference import classify_inference
from service.figureInference import figure_inference
from service.imageupscaler import upscale_image
from service.downloadimage import download_image

logger = logging.getLogger(__name__)

def is_valid_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def get_mime_type(url: str) -> str:
    path = urlparse(url).path
    ext = os.path.splitext(path)[1].lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp'
    }
    return mime_types.get(ext, 'application/octet-stream')

def process_image(input_path: str, output_path: str, upscale_model: str, scale: int, mime_ext: str):
    if upscale_model in [FIGURE_MODEL, FIGURE_PRO_MODEL]:
        figure_inference(input_path, output_path, MODELS_PATH, FIGURE_BACKGROUND_MODEL,
                         upscale_model, scale, '', mime_ext)
    else:
        upscale_image(INPUT_DIR, os.path.basename(input_path), output_path,
                      MODELS_PATH, upscale_model, DELIMITER, '', mime_ext)
        
    if scale != int(DELIMITER):
        adjust_image_scale(input_path, output_path, scale)

def adjust_image_scale(input_path: str, output_path: str, scale: int):
    with Image.open(input_path) as origin_im, Image.open(output_path) as upscale_im:
        new_size = (origin_im.size[0] * scale, origin_im.size[1] * scale)
        new_upscale_im = upscale_im.resize(new_size)
        new_upscale_im.save(output_path)

def main(args: argparse.Namespace):
    start_time = time.time()

    if not is_valid_url(args.imageUrl):
        logger.error("Invalid URL format")
        return

    mime_type = get_mime_type(args.imageUrl)
    if mime_type == 'application/octet-stream':
        logger.warning("Could not determine MIME type from URL. Defaulting to image/png")
        mime_type = 'image/png'

    image_id = args.id or uuid.uuid4().hex
    mime_ext = mime_type.split('/')[-1]
    input_image_path = os.path.join(INPUT_DIR, f"{image_id}.{mime_ext}")
    output_image_path = os.path.join(OUTPUT_DIR, f"{image_id}_upscale.{mime_ext}")

    try:
        if not download_image(args.imageUrl, input_image_path):
            logger.error("Failed to download image")
            return

        upscale_model = classify_inference(input_image_path)
        logger.info(f'Using model for upscaling image: {upscale_model}')

        process_image(input_image_path, output_image_path, upscale_model, args.scale, mime_ext)

        logger.info(f'Processing completed. Time taken: {time.time() - start_time:.2f} seconds')
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image upscaling script")
    parser.add_argument("-imageUrl", required=True, help="URL of the image to upscale")
    parser.add_argument("-scale", type=int, choices=[2, 3, 4, 8, 16], default=2, help="Upscaling factor")
    parser.add_argument("-id", help="Optional ID for the image")

    args = parser.parse_args()
    main(args)