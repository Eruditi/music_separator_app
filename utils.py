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


def validate_audio_file(file_path):
    """
    验证音频文件是否有效
    
    Args:
        file_path: 音频文件路径
        
    Returns:
        tuple: (是否有效, 错误信息, 文件信息字典)
    """
    logger = logging.getLogger(__name__)
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        return False, "文件不存在", None
    
    # 检查是否为文件
    if not os.path.isfile(file_path):
        return False, "路径不是文件", None
    
    # 检查文件扩展名
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in AppConfig.SUPPORTED_FORMATS:
        supported = ', '.join(AppConfig.SUPPORTED_FORMATS)
        return False, f"不支持的文件格式: {ext}\n支持的格式: {supported}", None
    
    # 检查文件大小
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        return False, "文件为空", None
    
    max_size = AppConfig.MAX_FILE_SIZE_MB * 1024 * 1024
    if file_size > max_size:
        return False, f"文件过大: {format_file_size(file_size)}\n最大支持: {AppConfig.MAX_FILE_SIZE_MB}MB", None
    
    # 尝试读取音频文件头部信息
    file_info = {
        'path': file_path,
        'size': file_size,
        'size_formatted': format_file_size(file_size),
        'extension': ext,
        'format_valid': False,
        'duration': None,
        'sample_rate': None,
        'channels': None
    }
    
    # 使用librosa验证音频文件
    try:
        import librosa
        info = librosa.info(file_path)
        file_info['duration'] = info.duration
        file_info['sample_rate'] = info.sample_rate
        file_info['channels'] = info.channels
        file_info['format_valid'] = True
        
        logger.info(f"✅ 音频文件验证通过: {os.path.basename(file_path)}")
        logger.info(f"   时长: {info.duration:.2f}秒, 采样率: {info.sample_rate}Hz, 声道: {info.channels}")
        
        return True, None, file_info
        
    except ImportError:
        logger.warning("librosa未安装，跳过音频格式验证")
        file_info['format_valid'] = True
        return True, None, file_info
        
    except Exception as e:
        logger.error(f"❌ 音频文件验证失败: {str(e)}")
        return False, f"音频文件格式错误或已损坏: {str(e)}", file_info


def get_system_info():
    """
    获取系统信息
    
    Returns:
        dict: 系统信息字典
    """
    info = {
        'python_version': sys.version,
        'platform': sys.platform,
        'os_name': os.name,
        'cpu_count': os.cpu_count(),
    }
    
    # 获取内存信息
    try:
        import psutil
        mem = psutil.virtual_memory()
        info['total_memory'] = format_file_size(mem.total)
        info['available_memory'] = format_file_size(mem.available)
        info['memory_percent'] = mem.percent
    except ImportError:
        info['total_memory'] = "未知"
        info['available_memory'] = "未知"
    
    # 获取GPU信息
    try:
        import torch
        info['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['gpu_count'] = torch.cuda.device_count()
            info['gpu_name'] = torch.cuda.get_device_name(0)
        else:
            info['cuda_version'] = None
            info['gpu_count'] = 0
            info['gpu_name'] = None
    except ImportError:
        info['cuda_available'] = False
        info['cuda_version'] = None
        info['gpu_count'] = 0
        info['gpu_name'] = None
    
    return info


def cleanup_temp_files(temp_dir=None):
    """
    清理临时文件
    
    Args:
        temp_dir: 临时文件目录，默认为系统临时目录
        
    Returns:
        int: 清理的文件数量
    """
    import tempfile
    import glob
    
    logger = logging.getLogger(__name__)
    
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()
    
    cleaned_count = 0
    
    # 清理常见的临时文件模式
    patterns = ['*.tmp', '*.temp', '*_separated_*']
    
    for pattern in patterns:
        try:
            files = glob.glob(os.path.join(temp_dir, pattern))
            for file_path in files:
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        cleaned_count += 1
                except Exception:
                    pass
        except Exception:
            pass
    
    if cleaned_count > 0:
        logger.info(f"🧹 清理了 {cleaned_count} 个临时文件")
    
    return cleaned_count


