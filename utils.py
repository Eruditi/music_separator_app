# -*- coding: utf-8 -*-
"""
工具模块 - 通用工具函数和日志配置
"""

import os
import sys
import logging
import datetime
from pathlib import Path
from config import AppConfig

# 全局变量，确保日志只初始化一次
_logging_initialized = False


def setup_detailed_logging():
    """设置详细的日志记录系统"""
    global _logging_initialized
    
    if _logging_initialized:
        return None
    
    # 清除现有的处理器
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # 创建logs目录
    logs_dir = AppConfig.LOGS_PATH
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)
    
    # 创建日志文件
    log_file_name = f"music_separator_{datetime.datetime.now().strftime('%Y%m%d')}.log"
    log_file_path = os.path.join(logs_dir, log_file_name)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # 设置控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # 添加处理器
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.DEBUG)
    
    _logging_initialized = True
    return log_file_path


def log_system_info():
    """记录系统信息"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info(f"🚀 {AppConfig.APP_NAME} v{AppConfig.VERSION} 启动")
    logger.info(f"🐍 Python版本: {sys.version}")
    logger.info(f"💻 操作系统: {os.name} ({sys.platform})")
    logger.info(f"📁 工作目录: {os.getcwd()}")
    logger.info(f"📦 应用版本: {AppConfig.VERSION} ({AppConfig.BUILD_DATE})")
    logger.info("=" * 80)


def check_dependencies():
    """检查依赖文件"""
    logger = logging.getLogger(__name__)
    logger.info("🔍 检查依赖文件...")
    
    # 检查FFmpeg
    ffmpeg_dir = AppConfig.FFMPEG_PATH
    if os.path.exists(ffmpeg_dir):
        logger.info(f"✅ FFmpeg目录存在: {ffmpeg_dir}")
        ffmpeg_files = os.listdir(ffmpeg_dir)
        logger.debug(f"📁 FFmpeg文件: {ffmpeg_files}")
    else:
        logger.warning(f"⚠️ FFmpeg目录不存在: {ffmpeg_dir}")
        logger.warning("⚠️ 请确保FFmpeg已正确安装，否则音频处理可能失败")
    
    # 检查模型目录
    models_dir = AppConfig.MODELS_PATH
    if os.path.exists(models_dir):
        logger.info(f"✅ 模型目录存在: {models_dir}")
        for subdir in os.listdir(models_dir):
            subdir_path = os.path.join(models_dir, subdir)
            if os.path.isdir(subdir_path):
                file_count = len(os.listdir(subdir_path))
                logger.debug(f"📁 {subdir}模型文件: {file_count}个文件")
    else:
        logger.warning(f"⚠️ 模型目录不存在: {models_dir}")
        logger.info("💡 模型将在首次运行时自动下载")

def format_file_size(size_bytes):
    """格式化文件大小"""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.2f}{size_names[i]}"

def get_audio_file_info(file_path):
    """获取音频文件信息"""
    if not os.path.exists(file_path):
        return None
    
    stat = os.stat(file_path)
    return {
        'size': stat.st_size,
        'size_formatted': format_file_size(stat.st_size),
        'modified': datetime.datetime.fromtimestamp(stat.st_mtime),
        'extension': os.path.splitext(file_path)[1].upper()
    }


