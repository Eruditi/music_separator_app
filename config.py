# -*- coding: utf-8 -*-
"""
配置文件 - 应用程序常量和设置
包含颜色、字体、模型配置等全局设置
"""

import os


class AppColors:
    """应用程序颜色常量"""
    # 主色调
    PRIMARY = "#4a90e2"
    PRIMARY_DARK = "#357abd"
    PRIMARY_LIGHT = "#5ba0f2"
    
    # 背景色
    BACKGROUND_DARK = "#1a1a1a"
    BACKGROUND_MEDIUM = "#2d2d2d"
    BACKGROUND_LIGHT = "#3a3a3a"
    WIDGET_BACKGROUND = "#333333"
    
    # 边框色
    BORDER_NORMAL = "#404040"
    BORDER_HOVER = "#5a5a5a"
    BORDER_FOCUS = "#4a90e2"
    
    # 文字色
    TEXT_PRIMARY = "#ffffff"
    TEXT_SECONDARY = "#e0e0e0"
    TEXT_DISABLED = "#808080"
    
    # 状态色
    SUCCESS = "#28a745"
    SUCCESS_DARK = "#1e7e34"
    SUCCESS_LIGHT = "#34ce57"
    ERROR = "#dc3545"
    ERROR_DARK = "#c82333"
    WARNING = "#ffc107"
    INFO = "#4a90e2"

class AppConfig:
    """应用程序配置"""
    APP_NAME = "本地音乐多轨分离工具"
    VERSION = "1.3.0"
    BUILD_DATE = "2025-04-21"
    
    # 路径配置
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FFMPEG_PATH = os.path.join(BASE_DIR, 'ffmpeg', 'bin')
    MODELS_PATH = os.path.join(BASE_DIR, 'pretrained_models')
    LOGS_PATH = os.path.join(BASE_DIR, 'logs')
    
    # 音频配置
    DEFAULT_SAMPLE_RATE = 44100
    DEFAULT_BIT_DEPTH = 16
    SUPPORTED_FORMATS = ['.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac', '.wma']
    
    # 性能配置
    DEFAULT_THREADS = 4
    DEFAULT_CACHE_SIZE = "1 GB"
    ENABLE_GPU = True
    MAX_FILE_SIZE_MB = 500
    
    # 内存优化配置
    MAX_MEMORY_USAGE_PERCENT = 80  # 最大内存使用百分比
    CHUNK_SIZE_MB = 50  # 音频处理分块大小(MB)
    ENABLE_MEMORY_MONITOR = True  # 启用内存监控
    
    # 音频处理优化
    DEFAULT_DOWNSAMPLE_RATE = 22050  # 默认降采样率
    WAVEFORM_DOWNSAMPLE_FACTOR = 1000  # 波形显示降采样因子
    MAX_WAVEFORM_POINTS = 1500  # 波形显示最大点数

class ModelConfig:
    """模型配置"""
    AVAILABLE_MODELS = {
        # Demucs 模型
        "demucs:2stems": "Demucs 2轨 (人声 + 伴奏)",
        "demucs:4stems": "Demucs 4轨 (人声 + 鼓 + 贝斯 + 其他)",
        "demucs:6stems": "Demucs 6轨 (人声 + 鼓 + 贝斯 + 钢琴 + 吉他 + 其他)"
    }
    
    # 模型下载链接
    DOWNLOAD_URLS = {
        "demucs:2stems": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/2stems/htdemucs_ft.th",
        "demucs:4stems": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/4stems/htdemucs.th",
        "demucs:6stems": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/6stems/htdemucs_6s.th"
    }

class UIStyles:
    """UI样式配置"""
    
    MAIN_WINDOW_STYLE = f"""
        QMainWindow {{
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 {AppColors.BACKGROUND_DARK}, stop:1 {AppColors.BACKGROUND_MEDIUM});
            color: {AppColors.TEXT_PRIMARY};
        }}
    """
    
    BUTTON_STYLE = f"""
        QPushButton {{
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 {AppColors.PRIMARY}, stop:1 {AppColors.PRIMARY_DARK});
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            font-weight: bold;
            font-size: 14px;
        }}
        QPushButton:hover {{
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 {AppColors.PRIMARY_LIGHT}, stop:1 {AppColors.PRIMARY});
        }}
        QPushButton:pressed {{
            background: {AppColors.PRIMARY_DARK};
        }}
        QPushButton:disabled {{
            background: {AppColors.TEXT_DISABLED};
            color: {AppColors.BACKGROUND_LIGHT};
        }}
    """
    
    GROUP_BOX_STYLE = f"""
        QGroupBox {{
            font-weight: bold;
            font-size: 14px;
            color: {AppColors.TEXT_PRIMARY};
            border: 2px solid {AppColors.BORDER_NORMAL};
            border-radius: 8px;
            margin-top: 10px;
            padding-top: 10px;
            background-color: rgba(255, 255, 255, 0.05);
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 15px;
            padding: 0 8px 0 8px;
            color: {AppColors.PRIMARY};
        }}
    """
    
    RADIO_BUTTON_STYLE = f"""
        QRadioButton {{
            color: {AppColors.TEXT_SECONDARY};
            font-size: 13px;
            spacing: 8px;
        }}
        QRadioButton::indicator {{
            width: 16px;
            height: 16px;
        }}
        QRadioButton::indicator:unchecked {{
            border: 2px solid {AppColors.BORDER_NORMAL};
            border-radius: 8px;
            background-color: transparent;
        }}
        QRadioButton::indicator:checked {{
            border: 2px solid {AppColors.PRIMARY};
            border-radius: 8px;
            background-color: {AppColors.PRIMARY};
        }}
        QRadioButton:hover {{
            color: {AppColors.TEXT_PRIMARY};
        }}
    """

class ThemeManager:
    """主题管理器 - 处理深色/浅色主题切换"""
    
    # 浅色主题颜色配置
    LIGHT_COLORS = {
        # 主色调
        'PRIMARY': "#2196F3",
        'PRIMARY_DARK': "#1976D2", 
        'PRIMARY_LIGHT': "#42A5F5",
        
        # 背景色
        'BACKGROUND_DARK': "#f5f5f5",
        'BACKGROUND_MEDIUM': "#ffffff",
        'BACKGROUND_LIGHT': "#fafafa",
        'WIDGET_BACKGROUND': "#ffffff",
        
        # 边框色
        'BORDER_NORMAL': "#e0e0e0",
        'BORDER_HOVER': "#bdbdbd",
        'BORDER_FOCUS': "#2196F3",
        
        # 文字色
        'TEXT_PRIMARY': "#212121",
        'TEXT_SECONDARY': "#757575",
        'TEXT_DISABLED': "#9e9e9e",
        
        # 状态色
        'SUCCESS': "#4caf50",
        'SUCCESS_DARK': "#388e3c",
        'SUCCESS_LIGHT': "#66bb6a",
        'ERROR': "#f44336",
        'ERROR_DARK': "#d32f2f",
        'WARNING': "#ff9800",
        'INFO': "#2196F3"
    }
    
    # 深色主题颜色配置（使用现有的AppColors）
    DARK_COLORS = {
        'PRIMARY': AppColors.PRIMARY,
        'PRIMARY_DARK': AppColors.PRIMARY_DARK,
        'PRIMARY_LIGHT': AppColors.PRIMARY_LIGHT,
        'BACKGROUND_DARK': AppColors.BACKGROUND_DARK,
        'BACKGROUND_MEDIUM': AppColors.BACKGROUND_MEDIUM,
        'BACKGROUND_LIGHT': AppColors.BACKGROUND_LIGHT,
        'WIDGET_BACKGROUND': AppColors.WIDGET_BACKGROUND,
        'BORDER_NORMAL': AppColors.BORDER_NORMAL,
        'BORDER_HOVER': AppColors.BORDER_HOVER,
        'BORDER_FOCUS': AppColors.BORDER_FOCUS,
        'TEXT_PRIMARY': AppColors.TEXT_PRIMARY,
        'TEXT_SECONDARY': AppColors.TEXT_SECONDARY,
        'TEXT_DISABLED': AppColors.TEXT_DISABLED,
        'SUCCESS': AppColors.SUCCESS,
        'SUCCESS_DARK': AppColors.SUCCESS_DARK,
        'SUCCESS_LIGHT': AppColors.SUCCESS_LIGHT,
        'ERROR': AppColors.ERROR,
        'ERROR_DARK': AppColors.ERROR_DARK,
        'WARNING': AppColors.WARNING,
        'INFO': AppColors.INFO
    }
    
    @staticmethod
    def get_theme_styles():
        """获取深色主题样式"""
        colors = ThemeManager.DARK_COLORS
        
        return {
            'main_window': f"""
                QMainWindow {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                        stop:0 {colors['BACKGROUND_DARK']}, stop:1 {colors['BACKGROUND_MEDIUM']});
                    color: {colors['TEXT_PRIMARY']};
                }}
            """,
            
            'button': f"""
                QPushButton {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                        stop:0 {colors['PRIMARY']}, stop:1 {colors['PRIMARY_DARK']});
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 6px;
                    font-weight: bold;
                    font-size: 14px;
                }}
                QPushButton:hover {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                        stop:0 {colors['PRIMARY_LIGHT']}, stop:1 {colors['PRIMARY']});
                }}
                QPushButton:pressed {{
                    background: {colors['PRIMARY_DARK']};
                }}
                QPushButton:disabled {{
                    background: {colors['TEXT_DISABLED']};
                    color: {colors['BACKGROUND_LIGHT']};
                }}
            """,
            
            'group_box': f"""
                QGroupBox {{
                    font-weight: bold;
                    font-size: 14px;
                    color: {colors['TEXT_PRIMARY']};
                    border: 2px solid {colors['BORDER_NORMAL']};
                    border-radius: 8px;
                    margin-top: 10px;
                    padding-top: 10px;
                    background-color: rgba(255, 255, 255, 0.05);
                }}
                QGroupBox::title {{
                    subcontrol-origin: margin;
                    left: 15px;
                    padding: 0 8px 0 8px;
                    color: {colors['PRIMARY']};
                }}
            """,
            
            'radio_button': f"""
                QRadioButton {{
                    color: {colors['TEXT_SECONDARY']};
                    font-size: 13px;
                    spacing: 8px;
                }}
                QRadioButton::indicator {{
                    width: 16px;
                    height: 16px;
                }}
                QRadioButton::indicator:unchecked {{
                    border: 2px solid {colors['BORDER_NORMAL']};
                    border-radius: 8px;
                    background-color: transparent;
                }}
                QRadioButton::indicator:checked {{
                    border: 2px solid {colors['PRIMARY']};
                    border-radius: 8px;
                    background-color: {colors['PRIMARY']};
                }}
                QRadioButton:hover {{
                    color: {colors['TEXT_PRIMARY']};
                }}
            """,
            
            'line_edit': f"""
                QLineEdit {{
                    padding: 10px 12px;
                    border: 2px solid {colors['BORDER_NORMAL']};
                    border-radius: 6px;
                    background-color: {colors['WIDGET_BACKGROUND']};
                    color: {colors['TEXT_PRIMARY']};
                    font-size: 13px;
                    min-height: 16px;
                }}
                QLineEdit:focus {{
                    border: 2px solid {colors['BORDER_FOCUS']};
                    background-color: {colors['BACKGROUND_LIGHT']};
                }}
            """,
            
            'text_edit': f"""
                QTextEdit {{
                    border: 2px solid {colors['BORDER_NORMAL']};
                    border-radius: 6px;
                    padding: 8px;
                    background-color: {colors['BACKGROUND_DARK']};
                    color: {colors['TEXT_SECONDARY']};
                    font-family: 'Consolas', 'Monaco', monospace;
                    font-size: 12px;
                    selection-background-color: {colors['PRIMARY']};
                }}
            """,
            
            'status_bar': f"""
                QStatusBar {{
                    background-color: {colors['BACKGROUND_DARK']};
                    color: {colors['TEXT_SECONDARY']};
                    border-top: 1px solid {colors['BORDER_NORMAL']};
                    font-size: 12px;
                }}
            """,
            
            # 下拉框样式
            'combo_box': f"""
                QComboBox {{
                    background-color: {colors['WIDGET_BACKGROUND']};
                    border: 2px solid {colors['BORDER_NORMAL']};
                    border-radius: 6px;
                    padding: 5px 10px;
                    color: {colors['TEXT_PRIMARY']};
                    font-size: 13px;
                }}
                QComboBox:hover {{
                    border: 2px solid {colors['BORDER_HOVER']};
                }}
                QComboBox::drop-down {{
                    subcontrol-origin: padding;
                    subcontrol-position: top right;
                    width: 15px;
                    border-left-width: 1px;
                    border-left-color: {colors['BORDER_NORMAL']};
                    border-left-style: solid;
                    border-top-right-radius: 6px;
                    border-bottom-right-radius: 6px;
                }}
                QComboBox::down-arrow {{
                    image: none;
                    border-left: 5px solid transparent;
                    border-right: 5px solid transparent;
                    border-top: 5px solid {colors['TEXT_SECONDARY']};
                }}
                QComboBox QAbstractItemView {{
                    background-color: {colors['WIDGET_BACKGROUND']};
                    border: 1px solid {colors['BORDER_NORMAL']};
                    color: {colors['TEXT_PRIMARY']};
                    selection-background-color: {colors['PRIMARY']};
                }}
            """
        }