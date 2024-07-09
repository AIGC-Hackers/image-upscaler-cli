import subprocess
from pathlib import Path
from typing import List, Dict, Union
import logging
from config import COMMANDS_PATH

logger = logging.getLogger(__name__)

COMMANDS = Path(COMMANDS_PATH)

def construct_upscaling_command(arguments: List[str]) -> List[str]:
    """Construct the full command for upscaling."""
    return [str(COMMANDS)] + arguments


def get_single_image_arguments(
    input_dir: Path,
    filename: str,
    output_file: Path,
    models_path: Path,
    model: str,
    scale: str,
    gpu_id: str,
    save_format: str
) -> List[str]:
    """Construct arguments for single image upscaling."""
    return [
        "-i", str(input_dir / filename),
        "-o", str(output_file),
        "-s", scale,
        "-m", str(models_path),
        "-n", model,
        "-g", gpu_id,
        "-f", save_format,
    ]


def upscale_image(
    input_dir: Union[str, Path],
    filename: str,
    output_file: Union[str, Path],
    models_path: Union[str, Path],
    model: str,
    scale: str,
    gpu_id: str = "",
    save_format: str = "png"
) -> Dict[str, Union[bool, str]]:
    """
    Upscale a single image using the specified model and parameters.

    Returns a dictionary with 'success' status and 'message' or 'error'.
    """
    input_dir = Path(input_dir)
    output_file = Path(output_file)
    models_path = Path(models_path)

    arguments = get_single_image_arguments(
        input_dir, filename, output_file, models_path, model, scale, gpu_id, save_format
    )

    command = construct_upscaling_command(arguments)
    logger.info(f"Upscaling command: {' '.join(map(str, command))}")

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info("Image upscaling completed successfully.")
        return {"success": True, "message": "Image upscaled successfully"}
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during image upscaling: {e.stderr}")
        return {"success": False, "error": f"Upscale Image CalledProcessError: {e.stderr}"}
    except Exception as e:
        logger.exception("Unexpected error during image upscaling")
        return {"success": False, "error": f"Unexpected error: {str(e)}"}


# Example usage (not part of the module):
if __name__ == "__main__":
    result = upscale_image(
        input_dir="input",
        filename="test.png",
        output_file="output/test_upscaled.png",
        models_path="models",
        model="RealESRGAN_x4plus",
        scale="4",
        gpu_id="0",
        save_format="png"
    )
    print(result)
