import os
import requests
from requests.exceptions import RequestException
import logging
from config import LOG_FORMAT

# 设置日志
logging.basicConfig(level=logging.INFO,
                    format=LOG_FORMAT)
logger = logging.getLogger(__name__)

def download_image(url, filename):
    """
    从给定的URL下载图片并保存到指定文件名。

    Args:
        url (str): 图片的URL
        filename (str): 保存图片的文件名（包括路径）

    Returns:
        bool: 下载成功返回True，否则返回False
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # 如果请求不成功则抛出异常
    except RequestException as e:
        logger.error(f'下载图片时出错，地址: {url}. 错误: {str(e)}')
        return False

    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as file:
            file.write(response.content)
        logger.info(f'图片已成功保存为 {filename}')
        return True
    except IOError as e:
        logger.error(f'保存图片时出错，文件名: {filename}. 错误: {str(e)}')
        return False