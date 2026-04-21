import sys
import os
import logging
import warnings
import datetime
import time

import numpy as np

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QGroupBox, QRadioButton, QMessageBox, QLineEdit, QTextEdit, QComboBox, QCheckBox,
                            QTabWidget, QDialog, QSlider, QSpinBox, QDesktopWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl, QTimer
from PyQt5.QtGui import QDesktopServices, QPainter, QPen, QColor, QPainterPath

from models import DemucsModel, ModelNotFoundError
from config import AppColors, AppConfig, UIStyles

warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*sip.*")
warnings.filterwarnings("ignore", message=".*sipPyTypeDict.*")

from utils import setup_detailed_logging, log_system_info, get_audio_file_info, validate_audio_file, get_system_info, cleanup_temp_files
from settings_manager import get_settings_manager

log_file_path = None
logger = logging.getLogger(__name__)
settings = get_settings_manager()


def check_dependencies():
    """检查应用所需的依赖项是否安装"""
    required_packages = [
        ('PyQt5', 'PyQt5'),
        ('numpy', 'numpy'),
        ('librosa', 'librosa'),
        ('torch', 'torch'),
        ('torchaudio', 'torchaudio'),
        ('ffmpeg', 'ffmpeg-python')
    ]
    missing_packages = []

    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)

    if missing_packages:
        error_msg = "检测到缺失的依赖项:\n"
        error_msg += ", ".join(missing_packages) + "\n"
        error_msg += "请使用以下命令安装:\n"
        error_msg += f"pip install {' '.join(missing_packages)}"
        logger.error(error_msg)
        print("\n" + "="*60)
        print("错误: 缺失依赖项")
        print("="*60)
        print(error_msg)
        print("="*60 + "\n")
        return False
    return True


def check_gpu_availability():
    """检查GPU是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"GPU可用: {torch.cuda.get_device_name(0)}")
            return True
        logger.info("未检测到可用GPU，将使用CPU进行处理")
        return False
    except ImportError:
        logger.warning("PyTorch未安装，无法检查GPU可用性")
        return False


if not check_dependencies():
    sys.exit(1)

class AudioLoadThread(QThread):
    """
    音频加载后台线程
    避免UI卡顿
    """
    waveform_loaded = pyqtSignal(np.ndarray, int)  # 波形数据, 采样率
    load_failed = pyqtSignal(str)  # 错误信息
    
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        
    def run(self):
        """后台加载音频文件"""
        try:
            # 尝试使用librosa加载（更好的性能和兼容性）
            try:
                import librosa
                # 加载音频文件，降采样以提高性能
                audio_data, sr = librosa.load(self.file_path, sr=22050, mono=True)
                
                # 进一步降采样用于显示（每1000个样本取一个）
                downsample_factor = max(1, len(audio_data) // 1500)
                waveform_data = audio_data[::downsample_factor]
                
                self.waveform_loaded.emit(waveform_data, sr)
                return
                
            except ImportError:
                pass
            
            # 备用方案：使用wave模块
            try:
                import wave
                import struct
                
                with wave.open(self.file_path, 'rb') as wav_file:
                    frames = wav_file.readframes(-1)
                    sound_info = struct.unpack('<' + ('h' * (len(frames) // 2)), frames)
                    
                    # 转换为numpy数组并归一化
                    audio_data = np.array(sound_info).astype(np.float32)
                    if np.max(np.abs(audio_data)) > 0:
                        audio_data = audio_data / np.max(np.abs(audio_data))
                    
                    # 降采样用于显示
                    downsample_factor = max(1, len(audio_data) // 1500)
                    waveform_data = audio_data[::downsample_factor]
                    
                    self.waveform_loaded.emit(waveform_data, wav_file.getframerate())
                    return
                    
            except Exception:
                pass
                
            # 最后备用方案：使用torchaudio
            try:
                import torchaudio
                import torch
                waveform, sample_rate = torchaudio.load(self.file_path)
                
                # 转换为单声道
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0)
                else:
                    waveform = waveform[0]
                
                # 转换为numpy并降采样
                audio_data = waveform.numpy()
                downsample_factor = max(1, len(audio_data) // 1500)
                waveform_data = audio_data[::downsample_factor]
                
                self.waveform_loaded.emit(waveform_data, sample_rate)
                return
                
            except Exception:
                pass
                
            self.load_failed.emit("无法加载音频文件：不支持的格式或文件损坏")
            
        except Exception as e:
            self.load_failed.emit(f"加载音频文件时出错: {str(e)}")

class AudioWaveformWidget(QWidget):
    """
    音频波形预览组件
    显示音频文件的波形图
    """
    def __init__(self):
        super().__init__()
        self.waveform_data = None
        self.sample_rate = None
        self.is_loading = False
        self.is_processing = False  # 新增：是否正在处理分离
        self.load_thread = None
        self.setMinimumHeight(200)
        self.setStyleSheet("""
            background-color: #1e1e1e;
            border: 2px solid #404040;
            border-radius: 6px;
        """)
        
        # 创建加载动画定时器
        self.loading_timer = QTimer()
        self.loading_timer.timeout.connect(self.update_loading_animation)
        self.loading_angle = 0
        
        # 创建扫描动画定时器
        self.scanning_timer = QTimer()
        self.scanning_timer.timeout.connect(self.update_scanning_animation)
        self.scan_position = 0
        self.scan_direction = 1  # 1为向右，-1为向左
        
    def load_audio_file(self, file_path):
        """使用后台线程加载音频文件，避免UI卡顿"""
        # 如果已有线程在运行，先停止
        if self.load_thread and self.load_thread.isRunning():
            self.load_thread.quit()
            self.load_thread.wait()
        
        # 开始加载状态
        self.is_loading = True
        self.waveform_data = None
        self.loading_timer.start(50)  # 50ms更新一次动画
        self.update()
        
        # 创建并启动加载线程
        self.load_thread = AudioLoadThread(file_path)
        self.load_thread.waveform_loaded.connect(self.on_waveform_loaded)
        self.load_thread.load_failed.connect(self.on_load_failed)
        self.load_thread.start()
        
        return True  # 立即返回，实际加载在后台进行
    
    def on_waveform_loaded(self, waveform_data, sample_rate):
        """波形数据加载完成的回调"""
        self.is_loading = False
        self.loading_timer.stop()
        self.waveform_data = waveform_data
        self.sample_rate = sample_rate
        self.update()
    
    def on_load_failed(self, error_message):
        """波形数据加载失败的回调"""
        self.is_loading = False
        self.loading_timer.stop()
        self.waveform_data = None
        self.update()
        # 使用QMessageBox显示错误信息给用户
        QMessageBox.critical(self, "加载失败", f"音频加载失败: {error_message}")
        logger.error(f"音频加载失败: {error_message}")
    
    def update_loading_animation(self):
        """更新加载动画"""
        self.loading_angle = (self.loading_angle + 10) % 360
        self.update()
    
    def start_processing_animation(self):
        """开始处理扫描动画"""
        if self.waveform_data is not None:
            self.is_processing = True
            self.scan_position = 0
            self.scan_direction = 1
            self.scanning_timer.start(30)  # 30ms更新一次，更流畅
            logger.debug("🎬 开始波形扫描动画")
    
    def stop_processing_animation(self):
        """停止处理扫描动画"""
        self.is_processing = False
        self.scanning_timer.stop()
        self.update()
        logger.debug("⏹️ 停止波形扫描动画")
    
    def update_scanning_animation(self):
        """更新扫描动画"""
        if not self.is_processing:
            return
            
        # 计算扫描线位置
        width = self.width() - 20
        self.scan_position += self.scan_direction * 3  # 每次移动3像素
        
        # 到达边界时反向
        if self.scan_position >= width:
            self.scan_position = width
            self.scan_direction = -1
        elif self.scan_position <= 0:
            self.scan_position = 0
            self.scan_direction = 1
            
        self.update()
    
    def paintEvent(self, event):
        """绘制波形"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 设置背景
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        
        # 如果正在加载，显示加载动画
        if self.is_loading:
            self.draw_loading_animation(painter)
            return
        
        if self.waveform_data is None:
            # 显示提示文本
            painter.setPen(QColor(160, 160, 160))
            painter.drawText(self.rect(), Qt.AlignCenter, "请选择音频文件")
            return
        
        # 绘制波形
        width = self.width() - 20
        height = self.height() - 20
        center_y = height // 2 + 10
        
        # 设置波形颜色
        painter.setPen(QPen(QColor(74, 144, 226), 1))
        
        # 绘制中心线
        painter.setPen(QPen(QColor(64, 64, 64), 1))
        painter.drawLine(10, center_y, width + 10, center_y)
        
        # 绘制波形
        painter.setPen(QPen(QColor(74, 144, 226), 1))
        
        if len(self.waveform_data) > 1:
            # 计算每个像素对应的样本数
            samples_per_pixel = len(self.waveform_data) / width
            
            path = QPainterPath()
            path.moveTo(10, center_y)
            
            for x in range(width):
                # 计算当前像素对应的样本索引
                sample_idx = int(x * samples_per_pixel)
                if sample_idx < len(self.waveform_data):
                    # 获取振幅值并缩放到显示高度
                    amplitude = self.waveform_data[sample_idx]
                    y = center_y - (amplitude * (height // 2 - 10))
                    
                    if x == 0:
                        path.moveTo(10 + x, y)
                    else:
                        path.lineTo(10 + x, y)
            
            painter.drawPath(path)
        
        # 如果正在处理，绘制扫描线
        if self.is_processing:
            self.draw_scanning_line(painter, width, height, center_y)
        
        # 绘制边框
        painter.setPen(QPen(QColor(64, 64, 64), 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
    
    def draw_scanning_line(self, painter, width, height, center_y):
        """绘制扫描线动画"""
        # 计算扫描线位置
        scan_x = 10 + self.scan_position
        
        # 绘制扫描线光晕效果（多层渐变）
        for i in range(5):
            alpha = 120 - i * 20  # 渐变透明度
            line_width = 1 + i  # 渐变宽度
            glow_color = QColor(74, 144, 226, alpha)  # 蓝色光晕
            painter.setPen(QPen(glow_color, line_width))
            painter.drawLine(scan_x, 10, scan_x, height + 10)
        
        # 绘制主扫描线
        scan_color = QColor(74, 144, 226, 200)  # 主蓝色线
        painter.setPen(QPen(scan_color, 2))
        painter.drawLine(scan_x, 10, scan_x, height + 10)
        
        # 绘制扫描区域高亮效果
        if self.waveform_data is not None:
            # 在扫描线附近绘制高亮的波形段
            highlight_width = 20  # 高亮区域宽度
            start_x = max(10, scan_x - highlight_width // 2)
            end_x = min(width + 10, scan_x + highlight_width // 2)
            
            # 绘制高亮波形段
            painter.setPen(QPen(QColor(255, 255, 255, 150), 1))
            samples_per_pixel = len(self.waveform_data) / width
            
            for x in range(int(start_x - 10), int(end_x - 10)):
                if 0 <= x < width:
                    sample_idx = int(x * samples_per_pixel)
                    if sample_idx < len(self.waveform_data):
                        amplitude = self.waveform_data[sample_idx]
                        y = center_y - (amplitude * (height // 2 - 10))
                        painter.drawPoint(10 + x, int(y))
        
        # 移除了扫描状态文字，保持界面简洁
        
        # 绘制进度指示器
        progress_y = height + 15
        painter.setPen(QColor(74, 144, 226, 100))
        painter.drawText(10, progress_y, f"扫描进度: {int((self.scan_position / width) * 100)}%")
    
    def draw_loading_animation(self, painter):
        """绘制加载动画"""
        # 设置加载动画的中心点
        center_x = self.width() // 2
        center_y = self.height() // 2
        
        # 绘制旋转的加载圆圈
        painter.setPen(QPen(QColor(74, 144, 226), 3))
        
        # 绘制加载圆弧
        radius = 30
        start_angle = self.loading_angle
        span_angle = 90
        
        # 绘制圆弧
        rect_x = center_x - radius
        rect_y = center_y - radius
        rect_size = radius * 2
        painter.drawArc(rect_x, rect_y, rect_size, rect_size, start_angle * 16, span_angle * 16)
        
        # 绘制加载文本
        painter.setPen(QColor(160, 160, 160))
        painter.drawText(self.rect(), Qt.AlignCenter, "正在加载...")


        
    def init_ui(self):
        """初始化UI组件"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # 创建标签页
        tab_widget = QTabWidget()
        
        # 音频处理设置
        audio_tab = self.create_audio_settings_tab()
        tab_widget.addTab(audio_tab, "🎵 音频处理")
        
        # 模型设置
        model_tab = self.create_model_settings_tab()
        tab_widget.addTab(model_tab, "🤖 模型配置")
        
        # 性能设置
        performance_tab = self.create_performance_settings_tab()
        tab_widget.addTab(performance_tab, "⚡ 性能优化")
        
        # 输出设置
        output_tab = self.create_output_settings_tab()
        tab_widget.addTab(output_tab, "📁 输出选项")
        
        # 界面设置
        ui_tab = self.create_ui_settings_tab()
        tab_widget.addTab(ui_tab, "🎨 界面设置")
        
        layout.addWidget(tab_widget)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        
        # 预设按钮
        self.preset_btn = QPushButton("📋 预设方案")
        self.preset_btn.clicked.connect(self.show_presets)
        
        self.reset_btn = QPushButton("🔄 重置默认")
        self.reset_btn.clicked.connect(self.reset_to_defaults)
        
        self.apply_btn = QPushButton("✅ 应用并保存")
        self.apply_btn.clicked.connect(self.apply_settings)
        
        self.save_btn = QPushButton("💾 仅保存")
        self.save_btn.clicked.connect(self.save_only)
        
        self.cancel_btn = QPushButton("❌ 取消")
        self.cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(self.preset_btn)
        button_layout.addWidget(self.reset_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
        
        # 在UI创建完成后加载保存的设置
        self.load_settings_from_file()
    
    def create_audio_settings_tab(self):
        """创建音频处理设置标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        
        # 音频质量设置
        quality_group = QGroupBox("🎛️ 音频质量设置")
        quality_layout = QVBoxLayout()
        
        # 采样率设置
        sample_rate_layout = QHBoxLayout()
        sample_rate_layout.addWidget(QLabel("输出采样率:"))
        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.addItems(["22050 Hz", "44100 Hz", "48000 Hz"])
        self.sample_rate_combo.setCurrentText("44100 Hz")
        sample_rate_layout.addWidget(self.sample_rate_combo)
        sample_rate_layout.addStretch()
        quality_layout.addLayout(sample_rate_layout)
        
        # 位深度设置
        bit_depth_layout = QHBoxLayout()
        bit_depth_layout.addWidget(QLabel("位深度:"))
        self.bit_depth_combo = QComboBox()
        self.bit_depth_combo.addItems(["16-bit", "24-bit", "32-bit"])
        self.bit_depth_combo.setCurrentText("16-bit")
        bit_depth_layout.addWidget(self.bit_depth_combo)
        bit_depth_layout.addStretch()
        quality_layout.addLayout(bit_depth_layout)
        
        quality_group.setLayout(quality_layout)
        layout.addWidget(quality_group)
        
        # 分离参数设置
        separation_group = QGroupBox("🎯 分离参数")
        separation_layout = QVBoxLayout()
        
        # 分离质量
        quality_layout_inner = QHBoxLayout()
        quality_layout_inner.addWidget(QLabel("分离质量:"))
        self.quality_slider = QSlider(Qt.Horizontal)
        self.quality_slider.setRange(1, 5)
        self.quality_slider.setValue(3)
        self.quality_slider.setTickPosition(QSlider.TicksBelow)
        self.quality_label = QLabel("标准")
        quality_layout_inner.addWidget(self.quality_slider)
        quality_layout_inner.addWidget(self.quality_label)
        separation_layout.addLayout(quality_layout_inner)
        
        # 连接滑块信号
        self.quality_slider.valueChanged.connect(self.update_quality_label)
        
        separation_group.setLayout(separation_layout)
        layout.addWidget(separation_group)
        
        layout.addStretch()
        return widget
    
    def create_model_settings_tab(self):
        """创建模型配置标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        
        # 模型参数设置
        model_group = QGroupBox("🤖 模型参数")
        model_layout = QVBoxLayout()
        
        # 模型shifts参数
        shifts_layout = QHBoxLayout()
        shifts_layout.addWidget(QLabel("模型Shifts:"))
        self.shifts_spin = QSpinBox()
        self.shifts_spin.setRange(1, 10)
        self.shifts_spin.setValue(1)
        self.shifts_spin.setToolTip("增加shifts可以提高质量但会增加处理时间")
        shifts_layout.addWidget(self.shifts_spin)
        shifts_layout.addWidget(QLabel("(推荐: 1-3)"))
        shifts_layout.addStretch()
        model_layout.addLayout(shifts_layout)
        
        # 重叠参数
        overlap_layout = QHBoxLayout()
        overlap_layout.addWidget(QLabel("音频重叠:"))
        self.overlap_slider = QSlider(Qt.Horizontal)
        self.overlap_slider.setRange(10, 50)
        self.overlap_slider.setValue(25)
        self.overlap_slider.setTickPosition(QSlider.TicksBelow)
        self.overlap_label = QLabel("25%")
        overlap_layout.addWidget(self.overlap_slider)
        overlap_layout.addWidget(self.overlap_label)
        model_layout.addLayout(overlap_layout)
        
        # 连接信号
        self.overlap_slider.valueChanged.connect(self.update_overlap_label)
        
        # 段大小设置
        segment_layout = QHBoxLayout()
        segment_layout.addWidget(QLabel("音频段大小:"))
        self.segment_spin = QSpinBox()
        self.segment_spin.setRange(5, 30)
        self.segment_spin.setValue(10)
        self.segment_spin.setSuffix(" 秒")
        segment_layout.addWidget(self.segment_spin)
        segment_layout.addStretch()
        model_layout.addLayout(segment_layout)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # 高级选项
        advanced_group = QGroupBox("🔬 高级选项")
        advanced_layout = QVBoxLayout()
        
        # TTA (Test Time Augmentation)
        self.tta_checkbox = QCheckBox("启用TTA (测试时增强)")
        self.tta_checkbox.setToolTip("可以提高分离质量，但会显著增加处理时间")
        advanced_layout.addWidget(self.tta_checkbox)
        
        # 后处理
        self.post_process_checkbox = QCheckBox("启用后处理优化")
        self.post_process_checkbox.setChecked(True)
        self.post_process_checkbox.setToolTip("对分离结果进行后处理优化")
        advanced_layout.addWidget(self.post_process_checkbox)
        
        # 批处理大小
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("批处理大小:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 8)
        self.batch_spin.setValue(1)
        batch_layout.addWidget(self.batch_spin)
        batch_layout.addStretch()
        advanced_layout.addLayout(batch_layout)
        
        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)
        
        layout.addStretch()
        return widget
    
    def create_ui_settings_tab(self):
        """创建界面设置标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        
        # 界面主题设置
        theme_group = QGroupBox("🎨 界面主题")
        theme_layout = QVBoxLayout()
        
        # 主题选择
        theme_select_layout = QHBoxLayout()
        theme_select_layout.addWidget(QLabel("主题风格:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["深色主题", "浅色主题", "自动跟随系统"])
        theme_select_layout.addWidget(self.theme_combo)
        theme_select_layout.addStretch()
        theme_layout.addLayout(theme_select_layout)
        
        # 强调色选择
        accent_layout = QHBoxLayout()
        accent_layout.addWidget(QLabel("强调色:"))
        self.accent_combo = QComboBox()
        self.accent_combo.addItems(["蓝色 (默认)", "绿色", "紫色", "橙色", "红色"])
        accent_layout.addWidget(self.accent_combo)
        accent_layout.addStretch()
        theme_layout.addLayout(accent_layout)
        
        theme_group.setLayout(theme_layout)
        layout.addWidget(theme_group)
        
        # 显示设置
        display_group = QGroupBox("📺 显示设置")
        display_layout = QVBoxLayout()
        
        # 字体大小
        font_layout = QHBoxLayout()
        font_layout.addWidget(QLabel("界面字体大小:"))
        self.font_slider = QSlider(Qt.Horizontal)
        self.font_slider.setRange(10, 18)
        self.font_slider.setValue(13)
        self.font_slider.setTickPosition(QSlider.TicksBelow)
        self.font_label = QLabel("13px")
        font_layout.addWidget(self.font_slider)
        font_layout.addWidget(self.font_label)
        display_layout.addLayout(font_layout)
        
        # 连接信号
        self.font_slider.valueChanged.connect(self.update_font_label)
        
        # 波形显示设置
        self.show_waveform_checkbox = QCheckBox("显示音频波形预览")
        self.show_waveform_checkbox.setChecked(True)
        display_layout.addWidget(self.show_waveform_checkbox)
        
        # 动画效果
        self.enable_animations_checkbox = QCheckBox("启用界面动画效果")
        self.enable_animations_checkbox.setChecked(True)
        display_layout.addWidget(self.enable_animations_checkbox)
        
        # 详细日志
        self.verbose_logging_checkbox = QCheckBox("显示详细处理日志")
        self.verbose_logging_checkbox.setChecked(True)
        display_layout.addWidget(self.verbose_logging_checkbox)
        
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        
        # 语言设置
        language_group = QGroupBox("🌐 语言设置")
        language_layout = QVBoxLayout()
        
        lang_layout = QHBoxLayout()
        lang_layout.addWidget(QLabel("界面语言:"))
        self.language_combo = QComboBox()
        self.language_combo.addItems(["简体中文", "English", "日本語"])
        lang_layout.addWidget(self.language_combo)
        lang_layout.addStretch()
        language_layout.addLayout(lang_layout)
        
        language_group.setLayout(language_layout)
        layout.addWidget(language_group)
        
        layout.addStretch()
        return widget
    
    def update_overlap_label(self, value):
        """更新重叠标签"""
        self.overlap_label.setText(f"{value}%")
    
    def update_font_label(self, value):
        """更新字体大小标签"""
        self.font_label.setText(f"{value}px")
    
    def create_performance_settings_tab(self):
        """创建性能设置标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        
        # 处理器设置
        cpu_group = QGroupBox("💻 处理器设置")
        cpu_layout = QVBoxLayout()
        
        # CPU线程数
        thread_layout = QHBoxLayout()
        thread_layout.addWidget(QLabel("CPU线程数:"))
        self.thread_spin = QSpinBox()
        self.thread_spin.setRange(1, 16)
        self.thread_spin.setValue(4)
        thread_layout.addWidget(self.thread_spin)
        thread_layout.addStretch()
        cpu_layout.addLayout(thread_layout)
        
        # GPU加速
        self.gpu_checkbox = QCheckBox("启用GPU加速 (如果可用)")
        self.gpu_checkbox.setChecked(True)
        
        # 检查GPU可用性
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            if not gpu_available:
                # 检查MPS (Apple Silicon GPU)支持
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    gpu_available = True
        except ImportError:
            gpu_available = False
        
        if not gpu_available:
            self.gpu_checkbox.setChecked(False)
            self.gpu_checkbox.setEnabled(False)
            self.gpu_checkbox.setText("启用GPU加速 (不可用 - 未检测到支持的GPU)")
            self.gpu_checkbox.setToolTip("未检测到可用的GPU或PyTorch配置不正确")
        
        cpu_layout.addWidget(self.gpu_checkbox)
        
        cpu_group.setLayout(cpu_layout)
        layout.addWidget(cpu_group)
        
        # 内存设置
        memory_group = QGroupBox("💾 内存管理")
        memory_layout = QVBoxLayout()
        
        # 内存缓存
        cache_layout = QHBoxLayout()
        cache_layout.addWidget(QLabel("内存缓存大小:"))
        self.cache_combo = QComboBox()
        self.cache_combo.addItems(["512 MB", "1 GB", "2 GB", "4 GB"])
        self.cache_combo.setCurrentText("1 GB")
        cache_layout.addWidget(self.cache_combo)
        cache_layout.addStretch()
        memory_layout.addLayout(cache_layout)
        
        # 自动清理
        self.auto_cleanup_checkbox = QCheckBox("处理完成后自动清理内存")
        self.auto_cleanup_checkbox.setChecked(True)
        memory_layout.addWidget(self.auto_cleanup_checkbox)
        
        memory_group.setLayout(memory_layout)
        layout.addWidget(memory_group)
        
        layout.addStretch()
        return widget
    
    def create_output_settings_tab(self):
        """创建输出设置标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        
        # 文件命名设置
        naming_group = QGroupBox("📝 文件命名")
        naming_layout = QVBoxLayout()
        
        # 命名模式
        naming_mode_layout = QHBoxLayout()
        naming_mode_layout.addWidget(QLabel("命名模式:"))
        self.naming_combo = QComboBox()
        self.naming_combo.addItems([
            "原文件名_轨道名",
            "轨道名_原文件名", 
            "时间戳_轨道名",
            "自定义前缀_轨道名"
        ])
        naming_mode_layout.addWidget(self.naming_combo)
        naming_mode_layout.addStretch()
        naming_layout.addLayout(naming_mode_layout)
        
        # 自定义前缀
        prefix_layout = QHBoxLayout()
        prefix_layout.addWidget(QLabel("自定义前缀:"))
        self.prefix_edit = QLineEdit()
        self.prefix_edit.setPlaceholderText("例如: MySong")
        prefix_layout.addWidget(self.prefix_edit)
        naming_layout.addLayout(prefix_layout)
        
        naming_group.setLayout(naming_layout)
        layout.addWidget(naming_group)
        
        # 输出格式设置
        format_group = QGroupBox("🎵 输出格式")
        format_layout = QVBoxLayout()
        
        # 文件格式
        format_layout_inner = QHBoxLayout()
        format_layout_inner.addWidget(QLabel("输出格式:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["WAV", "FLAC", "MP3"])
        format_layout_inner.addWidget(self.format_combo)
        format_layout_inner.addStretch()
        format_layout.addLayout(format_layout_inner)
        
        # 压缩质量 (仅MP3)
        compression_layout = QHBoxLayout()
        compression_layout.addWidget(QLabel("MP3质量:"))
        self.compression_slider = QSlider(Qt.Horizontal)
        self.compression_slider.setRange(128, 320)
        self.compression_slider.setValue(192)
        self.compression_slider.setTickPosition(QSlider.TicksBelow)
        self.compression_label = QLabel("192 kbps")
        compression_layout.addWidget(self.compression_slider)
        compression_layout.addWidget(self.compression_label)
        format_layout.addLayout(compression_layout)
        
        # 连接滑块信号
        self.compression_slider.valueChanged.connect(self.update_compression_label)
        
        format_group.setLayout(format_layout)
        layout.addWidget(format_group)
        
        # 其他选项
        other_group = QGroupBox("🔧 其他选项")
        other_layout = QVBoxLayout()
        
        self.normalize_checkbox = QCheckBox("音频标准化 (统一音量)")
        self.normalize_checkbox.setChecked(True)
        other_layout.addWidget(self.normalize_checkbox)
        
        self.create_subfolder_checkbox = QCheckBox("为每个文件创建子文件夹")
        other_layout.addWidget(self.create_subfolder_checkbox)
        
        other_group.setLayout(other_layout)
        layout.addWidget(other_group)
        
        layout.addStretch()
        return widget
    
    def update_quality_label(self, value):
        """更新质量标签"""
        labels = ["最低", "较低", "标准", "较高", "最高"]
        self.quality_label.setText(labels[value - 1])
    
    def update_compression_label(self, value):
        """更新压缩质量标签"""
        self.compression_label.setText(f"{value} kbps")
    
    def reset_to_defaults(self):
        """重置为默认设置"""
        # 音频设置
        self.sample_rate_combo.setCurrentText("44100 Hz")
        self.bit_depth_combo.setCurrentText("16-bit")
        self.quality_slider.setValue(3)
        
        # 性能设置
        self.thread_spin.setValue(4)
        self.gpu_checkbox.setChecked(True)
        self.cache_combo.setCurrentText("1 GB")
        self.auto_cleanup_checkbox.setChecked(True)
        
        # 输出设置
        self.naming_combo.setCurrentIndex(0)
        self.prefix_edit.clear()
        self.format_combo.setCurrentText("WAV")
        self.compression_slider.setValue(192)
        self.normalize_checkbox.setChecked(True)
        self.create_subfolder_checkbox.setChecked(False)
        
        QMessageBox.information(self, "设置重置", "所有设置已重置为默认值")
    
    def apply_settings(self):
        """应用设置"""
        # 这里可以保存设置到配置文件
        settings = {
            'sample_rate': self.sample_rate_combo.currentText(),
            'bit_depth': self.bit_depth_combo.currentText(),
            'quality': self.quality_slider.value(),
            'threads': self.thread_spin.value(),
            'gpu_enabled': self.gpu_checkbox.isChecked(),
            'cache_size': self.cache_combo.currentText(),
            'auto_cleanup': self.auto_cleanup_checkbox.isChecked(),
            'naming_mode': self.naming_combo.currentText(),
            'custom_prefix': self.prefix_edit.text(),
            'output_format': self.format_combo.currentText(),
            'mp3_quality': self.compression_slider.value(),
            'normalize': self.normalize_checkbox.isChecked(),
            'create_subfolder': self.create_subfolder_checkbox.isChecked()
        }
        
        # 保存设置到文件
        self.save_settings_to_file(settings)
        
        QMessageBox.information(self, "设置已保存", "高级设置已成功应用！")
        self.accept()

    def save_only(self):
        """仅保存设置，不关闭对话框"""
        settings = self.get_current_settings()
        self.save_settings_to_file(settings)
        QMessageBox.information(self, "设置已保存", "设置已保存，下次启动时将自动加载！")

    def load_default_settings(self):
        """加载默认设置"""
        return {
            'sample_rate': '44100 Hz',
            'bit_depth': '16-bit',
            'quality': 3,
            'threads': 4,
            'gpu_enabled': True,
            'cache_size': '1 GB',
            'auto_cleanup': True,
            'naming_mode': '原文件名_轨道名',
            'custom_prefix': '',
            'output_format': 'WAV',
            'mp3_quality': 192,
            'normalize': True,
            'create_subfolder': False,
            'model_shifts': 1,
            'overlap': 0.25,
            'segment_size': 10.0,
            'batch_size': 1,
            'use_tta': False,
            'post_process': True
        }

    def save_settings_to_file(self, settings):
        """保存设置到配置文件"""
        try:
            import json
            # 使用模型目录作为配置文件位置
            config_dir = 'pretrained_models'
            if not os.path.exists(config_dir):
                os.makedirs(config_dir)
            
            config_file = os.path.join(config_dir, 'advanced_settings.json')
            
            # 如果文件已存在，先读取现有内容以保留窗口状态
            existing_data = {}
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                except:
                    pass
            
            # 合并设置，保留窗口状态
            if 'window_state' in existing_data:
                settings['window_state'] = existing_data['window_state']
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            
            logger.info(f"高级设置已保存到: {config_file}")
        except Exception as e:
            logger.error(f"保存设置失败: {e}")

    def load_settings_from_file(self):
        """从配置文件加载设置"""
        try:
            import json
            # 从模型目录加载设置文件
            config_file = os.path.join('pretrained_models', 'advanced_settings.json')
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                self.apply_loaded_settings(settings)
                logger.info("已加载保存的高级设置")
        except Exception as e:
            logger.error(f"加载设置失败: {e}")

    def apply_loaded_settings(self, settings):
        """应用加载的设置到UI控件"""
        try:
            # 音频设置
            if 'sample_rate' in settings:
                self.sample_rate_combo.setCurrentText(settings['sample_rate'])
            if 'bit_depth' in settings:
                self.bit_depth_combo.setCurrentText(settings['bit_depth'])
            if 'quality' in settings:
                self.quality_slider.setValue(settings['quality'])
            
            # 性能设置
            if 'threads' in settings:
                self.thread_spin.setValue(settings['threads'])
            if 'gpu_enabled' in settings:
                self.gpu_checkbox.setChecked(settings['gpu_enabled'])
            if 'cache_size' in settings:
                self.cache_combo.setCurrentText(settings['cache_size'])
            if 'auto_cleanup' in settings:
                self.auto_cleanup_checkbox.setChecked(settings['auto_cleanup'])
            
            # 模型设置
            if 'model_shifts' in settings:
                self.shifts_spin.setValue(settings['model_shifts'])
            if 'overlap' in settings:
                self.overlap_slider.setValue(int(settings['overlap'] * 100))
            if 'segment_size' in settings:
                self.segment_spin.setValue(int(settings['segment_size']))
            if 'batch_size' in settings:
                self.batch_spin.setValue(settings['batch_size'])
            if 'use_tta' in settings:
                self.tta_checkbox.setChecked(settings['use_tta'])
            if 'post_process' in settings:
                self.post_process_checkbox.setChecked(settings['post_process'])
            
            # 输出设置
            if 'naming_mode' in settings:
                self.naming_combo.setCurrentText(settings['naming_mode'])
            if 'custom_prefix' in settings:
                self.prefix_edit.setText(settings['custom_prefix'])
            if 'output_format' in settings:
                self.format_combo.setCurrentText(settings['output_format'])
            if 'mp3_quality' in settings:
                self.compression_slider.setValue(settings['mp3_quality'])
            if 'normalize' in settings:
                self.normalize_checkbox.setChecked(settings['normalize'])
            if 'create_subfolder' in settings:
                self.create_subfolder_checkbox.setChecked(settings['create_subfolder'])
            
            # 界面设置
            if 'theme' in settings:
                self.theme_combo.setCurrentText(settings['theme'])
            if 'accent_color' in settings:
                self.accent_combo.setCurrentText(settings['accent_color'])
            if 'font_size' in settings:
                self.font_slider.setValue(settings['font_size'])
            if 'show_waveform' in settings:
                self.show_waveform_checkbox.setChecked(settings['show_waveform'])
            if 'enable_animations' in settings:
                self.enable_animations_checkbox.setChecked(settings['enable_animations'])
            if 'verbose_logging' in settings:
                self.verbose_logging_checkbox.setChecked(settings['verbose_logging'])
            if 'language' in settings:
                self.language_combo.setCurrentText(settings['language'])
                
        except Exception as e:
            logger.error(f"应用设置失败: {e}")

    def show_presets(self):
        """显示预设方案对话框"""
        presets = {
            "🎵 音质优先": {
                'sample_rate': '48000 Hz',
                'bit_depth': '24-bit',
                'quality': 5,
                'shifts_spin': 3,
                'overlap_slider': 30,
                'tta_checkbox': True,
                'post_process_checkbox': True,
                'normalize': True
            },
            "⚡ 速度优先": {
                'sample_rate': '22050 Hz',
                'bit_depth': '16-bit',
                'quality': 2,
                'shifts_spin': 1,
                'overlap_slider': 20,
                'tta_checkbox': False,
                'post_process_checkbox': False,
                'threads': 8
            },
            "⚖️ 平衡模式": {
                'sample_rate': '44100 Hz',
                'bit_depth': '16-bit',
                'quality': 3,
                'shifts_spin': 1,
                'overlap_slider': 25,
                'tta_checkbox': False,
                'post_process_checkbox': True,
                'normalize': True
            },
            "🎤 人声专用": {
                'sample_rate': '44100 Hz',
                'bit_depth': '16-bit',
                'quality': 4,
                'shifts_spin': 2,
                'overlap_slider': 35,
                'post_process_checkbox': True,
                'normalize': True,
                'output_format': 'WAV'
            }
        }
        
        # 创建预设选择对话框
        preset_dialog = QDialog(self)
        preset_dialog.setWindowTitle("选择预设方案")
        preset_dialog.setFixedSize(400, 300)
        
        layout = QVBoxLayout(preset_dialog)
        
        # 添加说明
        info_label = QLabel("选择一个预设方案快速配置设置:")
        layout.addWidget(info_label)
        
        # 预设按钮
        for preset_name, preset_settings in presets.items():
            btn = QPushButton(preset_name)
            btn.clicked.connect(lambda checked, settings=preset_settings: self.apply_preset(settings, preset_dialog))
            layout.addWidget(btn)
        
        # 取消按钮
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(preset_dialog.reject)
        layout.addWidget(cancel_btn)
        
        preset_dialog.exec_()

    def apply_preset(self, preset_settings, dialog):
        """应用预设设置"""
        try:
            for setting_name, value in preset_settings.items():
                if hasattr(self, setting_name):
                    widget = getattr(self, setting_name)
                    if isinstance(widget, QComboBox):
                        widget.setCurrentText(str(value))
                    elif isinstance(widget, QSlider) or isinstance(widget, QSpinBox):
                        widget.setValue(value)
                    elif isinstance(widget, QCheckBox):
                        widget.setChecked(value)
            
            QMessageBox.information(self, "预设已应用", "预设方案已成功应用到当前设置！")
            dialog.accept()
            
        except Exception as e:
            QMessageBox.warning(self, "应用失败", f"应用预设时出错: {str(e)}")

    def get_current_settings(self):
        """获取当前设置"""
        return {
            # 音频设置
            'sample_rate': self.sample_rate_combo.currentText(),
            'bit_depth': self.bit_depth_combo.currentText(),
            'quality': self.quality_slider.value(),
            
            # 模型设置
            'model_shifts': self.shifts_spin.value(),
            'overlap': self.overlap_slider.value() / 100.0,
            'segment_size': float(self.segment_spin.value()),
            'batch_size': self.batch_spin.value(),
            'use_tta': self.tta_checkbox.isChecked(),
            'post_process': self.post_process_checkbox.isChecked(),
            
            # 性能设置
            'threads': self.thread_spin.value(),
            'gpu_enabled': self.gpu_checkbox.isChecked(),
            'cache_size': self.cache_combo.currentText(),
            'auto_cleanup': self.auto_cleanup_checkbox.isChecked(),
            
            # 输出设置
            'naming_mode': self.naming_combo.currentText(),
            'custom_prefix': self.prefix_edit.text(),
            'output_format': self.format_combo.currentText(),
            'mp3_quality': self.compression_slider.value(),
            'normalize': self.normalize_checkbox.isChecked(),
            'create_subfolder': self.create_subfolder_checkbox.isChecked(),
            
            # 界面设置
            'theme': self.theme_combo.currentText(),
            'accent_color': self.accent_combo.currentText(),
            'font_size': self.font_slider.value(),
            'show_waveform': self.show_waveform_checkbox.isChecked(),
            'enable_animations': self.enable_animations_checkbox.isChecked(),
            'verbose_logging': self.verbose_logging_checkbox.isChecked(),
            'language': self.language_combo.currentText()
        }

class SeparationThread(QThread):
    """
    音频分离后台线程
    用于在后台执行音频分离任务，避免阻塞UI
    """
    progress_updated = pyqtSignal(str)
    separation_completed = pyqtSignal(list, str)
    separation_error = pyqtSignal(str)

    def __init__(self, model_type, audio_path, output_dir):
        super().__init__()
        self.model_type = model_type
        self.audio_path = audio_path
        self.output_dir = output_dir
        self.is_running = True

    def run(self):
        """
        线程执行函数
        初始化模型并执行音频分离
        """
        try:
            # 发送进度更新信号
            self.progress_updated.emit(f"初始化{self.model_type}模型...")

            # 将UI模型类型映射到实际的模型类型
            model_type_mapping = {
                # 经典模型 - 使用Demucs
                "2stems": "demucs:2stems",
                "4stems": "demucs:4stems", 
                "6stems": "demucs:6stems",
                
                # UVR5 模型
                "uvr5_hp2": "uvr5:HP2-人声/伴奏",
                "uvr5_hp3": "uvr5:HP3-人声/伴奏",
                "uvr5_hp5": "uvr5:HP5-人声/伴奏",
                
                # MDX-Net 模型
                "mdx_inst": "mdx:UVR-MDX-NET-Inst_HQ_3",
                "mdx_vocal": "mdx:UVR-MDX-NET-Voc_FT",
                "mdx_kim": "mdx:Kim_Vocal_2",
                
                # BS-RoFormer 模型
                "roformer_2": "bs_roformer:2stems",
                "roformer_4": "bs_roformer:4stems",
                "roformer_5": "bs_roformer:5stems"
            }
            
            # 获取实际的模型类型
            actual_model_type = model_type_mapping.get(self.model_type, f"demucs:{self.model_type}")
            
            # 使用ModelFactory创建模型实例
            from models import ModelFactory
            model = ModelFactory.create_model(actual_model_type)

            self.progress_updated.emit("模型初始化完成，开始分离音频...")

            # 执行音频分离
            output_files = model.separate(self.audio_path, self.output_dir)

            # 发送完成信号
            self.separation_completed.emit(output_files, self.output_dir)

        except ModelNotFoundError as e:
            self.separation_error.emit(f"模型文件不存在: {str(e)}\n请下载: {e.download_url}")
        except FileNotFoundError as e:
            self.separation_error.emit(f"文件不存在: {str(e)}")
        except NotImplementedError as e:
            self.separation_error.emit(f"模型功能开发中: {str(e)}")
        except Exception as e:
            logger.error(f"分离过程出错: {str(e)}", exc_info=True)
            self.separation_error.emit(f"分离失败: {str(e)}")

    def stop(self):
        """
        停止线程执行
        """
        self.is_running = False
        self.wait()

class MusicSeparatorMainWindow(QMainWindow):
    """
    本地音乐多轨分离应用的主窗口类
    继承自QMainWindow，提供应用的基础窗口框架和完整用户交互流程
    """
    def __init__(self):
        super().__init__()
        logger.info("🏗️ 初始化主窗口...")
        
        # 记录系统信息
        try:
            import psutil
            memory_info = psutil.virtual_memory()
            cpu_count = psutil.cpu_count()
            logger.info(f"💾 系统内存: {memory_info.total // (1024**3)}GB (可用: {memory_info.available // (1024**3)}GB)")
            logger.info(f"🖥️ CPU核心数: {cpu_count}")
        except ImportError:
            logger.debug("psutil未安装，跳过系统信息记录")
        
        # 检查依赖文件
        check_dependencies()
        
        self.selected_file_path = None
        self.selected_model = "2stems"  # 默认选择2stems模型
        self.output_dir = ""
        self.separation_thread = None
        
        # 主题管理
        self.is_dark_theme = True  # 默认深色主题
        self.load_theme_preference()  # 加载保存的主题偏好
        
        # 性能监控
        self.operation_start_time = None
        
        logger.debug("📋 初始化窗口属性...")
        self.init_window()
        logger.debug("🎨 初始化布局...")
        self.init_layout()
        logger.debug("🔧 初始化UI组件...")
        self.init_ui_components()
        logger.debug("🎭 应用主题样式...")
        self.apply_theme()
        
        logger.info("✅ 主窗口初始化完成")

    def load_window_state(self):
        """从模型目录的设置文件中加载窗口状态"""
        try:
            import json
            # 从模型目录加载设置文件
            config_file = os.path.join('pretrained_models', 'advanced_settings.json')
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                
                # 获取窗口状态
                window_state = settings.get('window_state', {})
                
                # 恢复窗口位置和大小
                x = window_state.get('x', 100)
                y = window_state.get('y', 100)
                width = window_state.get('width', 1200)
                height = window_state.get('height', 800)
                
                # 确保窗口在屏幕范围内
                from PyQt5.QtWidgets import QDesktopWidget
                desktop = QDesktopWidget()
                screen_rect = desktop.screenGeometry()
                
                # 调整位置确保窗口可见
                if x < 0 or x > screen_rect.width() - 200:
                    x = 100
                if y < 0 or y > screen_rect.height() - 200:
                    y = 100
                
                # 调整大小确保合理
                width = max(1000, min(width, screen_rect.width() - 100))
                height = max(700, min(height, screen_rect.height() - 100))
                
                self.setGeometry(x, y, width, height)
                
                # 恢复窗口状态
                if window_state.get('maximized', False):
                    self.showMaximized()
                
                logger.info(f"已恢复窗口状态: {width}x{height} at ({x}, {y})")
            else:
                # 默认窗口设置
                self.setGeometry(100, 100, 1200, 800)
                logger.info("使用默认窗口设置")
                
        except Exception as e:
            logger.error(f"加载窗口状态失败: {e}")
            # 使用默认设置
            self.setGeometry(100, 100, 1200, 800)

    def save_window_state(self):
        """保存当前窗口状态到模型目录的设置文件中"""
        try:
            import json
            
            # 使用模型目录作为配置文件位置
            config_dir = 'pretrained_models'
            if not os.path.exists(config_dir):
                os.makedirs(config_dir)
            
            config_file = os.path.join(config_dir, 'advanced_settings.json')
            
            # 获取当前窗口状态
            geometry = self.geometry()
            window_state = {
                'x': geometry.x(),
                'y': geometry.y(),
                'width': geometry.width(),
                'height': geometry.height(),
                'maximized': self.isMaximized()
            }
            
            # 读取现有设置文件
            existing_settings = {}
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        existing_settings = json.load(f)
                except:
                    pass
            
            # 更新窗口状态
            existing_settings['window_state'] = window_state
            
            # 保存到文件
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(existing_settings, f, indent=2, ensure_ascii=False)
            
            logger.info(f"窗口状态已保存到: {config_file}")
            logger.info(f"窗口状态: {window_state['width']}x{window_state['height']} at ({window_state['x']}, {window_state['y']})")
            
        except Exception as e:
            logger.error(f"保存窗口状态失败: {e}")

    def closeEvent(self, event):
        """窗口关闭事件，保存窗口状态"""
        self.save_window_state()
        event.accept()

    def init_window(self):
        """
        初始化窗口属性
        设置窗口标题、大小和位置
        """
        self.setWindowTitle("本地音乐多轨分离")
        
        # 加载保存的窗口状态
        self.load_window_state()
        
        self.setMinimumSize(1000, 700)

    def init_layout(self):
        """
        初始化主布局
        创建专业的左右分栏布局
        """
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建主水平布局
        self.main_layout = QHBoxLayout(central_widget)
        self.main_layout.setContentsMargins(15, 15, 15, 15)
        self.main_layout.setSpacing(20)

        # 创建左侧控制面板
        self.left_panel = QWidget()
        self.left_panel.setMinimumWidth(350)  # 设置最小宽度
        self.left_panel.setMaximumWidth(450)  # 设置最大宽度
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_layout.setContentsMargins(0, 0, 0, 0)
        self.left_layout.setSpacing(20)

        # 创建右侧工作区域
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.setSpacing(15)

        # 添加面板到主布局
        self.main_layout.addWidget(self.left_panel)
        self.main_layout.addWidget(self.right_panel)

    def add_welcome_message(self):
        """添加欢迎信息和使用提示"""
        import datetime
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        
        # 添加欢迎横幅
        self.log_text.append("=" * 50)
        self.log_text.append("🎵 本地音乐多轨分离工具")
        self.log_text.append("=" * 50)
        self.log_text.append(f"⏰ 启动时间: {current_time}")
        self.log_text.append("")
        
        # 添加功能介绍
        self.log_text.append("🚀 功能特点:")
        self.log_text.append("   • 支持2/4/5/6轨音频分离")
        self.log_text.append("   • 集成Demucs和Spleeter模型")
        self.log_text.append("   • 实时波形预览和处理动画")
        self.log_text.append("   • 完全本地处理，保护隐私")
        self.log_text.append("")
        
        # 添加使用提示
        self.log_text.append("📋 使用步骤:")
        self.log_text.append("   1️⃣ 选择音频文件")
        self.log_text.append("   2️⃣ 选择分离模式")
        self.log_text.append("   3️⃣ 设置输出目录")
        self.log_text.append("   4️⃣ 点击开始分离")
        self.log_text.append("")
        
        # 添加支持格式

        self.log_text.append("🎯 请选择音频文件开始您的音乐分离之旅!")
        self.log_text.append("=" * 50)

    def add_progress_info(self, message, details=""):
        """添加带时间戳的进度信息"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        if details:
            self.log_text.append(f"          {details}")

    def add_separator(self, title=""):
        """添加分隔线"""
        if title:
            self.log_text.append(f"{'='*15} {title} {'='*15}")
        else:
            self.log_text.append("=" * 50)

    def init_ui_components(self):
        """
        初始化UI组件
        创建专业的左右分栏界面布局
        """
        # === 左侧控制面板 ===
        
        # 标题区域
        title_label = QLabel("本地音乐多轨分离")
        title_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #4a90e2;
            padding: 10px 0;
            border-bottom: 2px solid #404040;
            margin-bottom: 10px;
        """)
        self.left_layout.addWidget(title_label)

        # 输入文件组
        input_group = QGroupBox("📁 音频输入")
        input_layout = QVBoxLayout()
        input_layout.setContentsMargins(15, 20, 15, 15)
        input_layout.setSpacing(12)
        
        # 文件路径显示
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)
        self.file_path_edit.setPlaceholderText("请选择音频文件...")
        
        # 选择文件按钮
        self.select_file_button = QPushButton("🎵 选择音频文件")
        self.select_file_button.clicked.connect(self.select_audio_file)
        
        input_layout.addWidget(QLabel("文件路径:"))
        input_layout.addWidget(self.file_path_edit)
        input_layout.addWidget(self.select_file_button)
        input_group.setLayout(input_layout)

        # 输出设置组
        output_group = QGroupBox("📂 输出设置")
        output_layout = QVBoxLayout()
        output_layout.setContentsMargins(15, 20, 15, 15)
        output_layout.setSpacing(12)
        
        self.output_dir_edit = QLineEdit(self.output_dir)
        self.output_dir_edit.setReadOnly(True)
        self.output_dir_edit.setPlaceholderText("自动设置为输入文件目录...")
        
        self.select_output_btn = QPushButton("📁 选择输出目录")
        self.select_output_btn.clicked.connect(self.select_output_directory)
        
        output_layout.addWidget(QLabel("输出目录:"))
        output_layout.addWidget(self.output_dir_edit)
        output_layout.addWidget(self.select_output_btn)
        output_group.setLayout(output_layout)

        # 分离模式组
        model_group = QGroupBox("🎛️ 分离配置")
        model_layout = QVBoxLayout()
        model_layout.setContentsMargins(15, 20, 15, 15)
        model_layout.setSpacing(10)
        
        # 简化的模型选项 - 只保留核心的2、4、6轨分离
        self.model_radios = {
            "2stems": QRadioButton("🎤 2轨分离 (人声 + 伴奏)"),
            "4stems": QRadioButton("🥁 4轨分离 (人声 + 鼓 + 贝斯 + 其他)"),
            "6stems": QRadioButton("🎸 6轨分离 (人声 + 鼓 + 贝斯 + 钢琴 + 吉他 + 其他)")
        }
        
        self.model_radios["2stems"].setChecked(True)
        for key, radio in self.model_radios.items():
            radio.toggled.connect(lambda checked, k=key: self.on_model_selected(k, checked))
            model_layout.addWidget(radio)
        
        # 添加说明
        info_label = QLabel("💡 使用Demucs模型进行高质量音轨分离")
        info_label.setStyleSheet("color: #a0a0a0; font-size: 11px; margin: 10px 0 5px 0;")
        model_layout.addWidget(info_label)
        model_group.setLayout(model_layout)

        # 控制按钮组
        control_group = QGroupBox("🎯 操作控制")
        control_layout = QVBoxLayout()
        control_layout.setContentsMargins(15, 20, 15, 15)
        control_layout.setSpacing(15)
        
        self.start_btn = QPushButton("🚀 开始分离")
        self.start_btn.setObjectName("startButton")
        self.start_btn.clicked.connect(self.start_separation)
        
        self.open_dir_btn = QPushButton("📂 打开输出目录")
        self.open_dir_btn.clicked.connect(self.open_output_directory)
        
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.open_dir_btn)
        control_group.setLayout(control_layout)

        # 添加组件到左侧面板
        self.left_layout.addWidget(input_group)
        self.left_layout.addWidget(output_group)
        self.left_layout.addWidget(model_group)
        self.left_layout.addWidget(control_group)
        self.left_layout.addStretch()

        # === 右侧工作区域 ===
        
        # 处理状态显示
        status_group = QGroupBox("📊 处理状态")
        status_layout = QVBoxLayout()
        status_layout.setContentsMargins(15, 20, 15, 15)
        
        # 状态信息面板
        self.status_info = QWidget()
        self.status_info.setStyleSheet("""
            background-color: rgba(74, 144, 226, 0.1);
            border: 1px solid #4a90e2;
            border-radius: 8px;
            padding: 15px;
        """)
        status_info_layout = QVBoxLayout(self.status_info)
        
        self.current_file_label = QLabel("当前文件: 未选择")
        self.current_model_label = QLabel("分离模式: 2轨分离")
        self.progress_label = QLabel("状态: 就绪")
        
        # 添加处理动画定时器
        self.processing_timer = QTimer()
        self.processing_timer.timeout.connect(self.update_processing_animation)
        self.processing_dots = 0
        self.base_processing_text = ""
        self.is_processing = False
        
        # 设置状态标签样式
        self.current_file_label.setStyleSheet("color: #e0e0e0; font-size: 13px; margin: 3px 0;")
        self.current_model_label.setStyleSheet("color: #e0e0e0; font-size: 13px; margin: 3px 0;")
        
        # 初始化状态为就绪状态（蓝色）
        self.set_status_with_color("状态: 就绪", "info")
        
        for label in [self.current_file_label, self.current_model_label, self.progress_label]:
            status_info_layout.addWidget(label)
        

        
        
        status_layout.addWidget(self.status_info)
        status_group.setLayout(status_layout)

        # 处理日志区域
        log_group = QGroupBox("📝 处理状态")
        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(15, 20, 15, 15)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(300)
        self.log_text.setPlaceholderText("处理状态将在这里显示...")
        
        # 添加初始提示
        self.add_welcome_message()
        self.log_text.append("  支持格式: MP3, WAV, FLAC, M4A, OGG")
        
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)

        # 音频波形预览区域
        waveform_group = QGroupBox("🎵 音频波形预览")
        waveform_layout = QVBoxLayout()
        waveform_layout.setContentsMargins(15, 20, 15, 15)
        
        # 创建音频波形显示组件
        self.waveform_widget = AudioWaveformWidget()
        self.waveform_widget.setMinimumHeight(200)
        self.waveform_widget.setMaximumHeight(250)
        
        # 音频信息标签
        self.audio_info_label = QLabel("请选择音频文件")
        self.audio_info_label.setStyleSheet("""
            color: #a0a0a0;
            font-size: 12px;
            padding: 5px;
            text-align: center;
        """)
        self.audio_info_label.setAlignment(Qt.AlignCenter)
        
        waveform_layout.addWidget(self.waveform_widget)
        waveform_layout.addWidget(self.audio_info_label)
        waveform_group.setLayout(waveform_layout)

        # 添加组件到右侧面板
        self.right_layout.addWidget(status_group)
        self.right_layout.addWidget(waveform_group)
        self.right_layout.addWidget(log_group)

    def on_model_selected(self, model_type, checked):
        """处理模型选择事件"""
        if checked:
            self.selected_model = model_type
            
            # 定义所有模型的显示名称
            model_names = {
                # 经典模型
                "2stems": "2轨分离 (人声 + 伴奏)",
                "4stems": "4轨分离 (人声 + 鼓 + 贝斯 + 其他)",
                "6stems": "6轨分离 (人声 + 鼓 + 贝斯 + 钢琴 + 吉他 + 其他)",
              
            }
            
            display_name = model_names.get(model_type, model_type)
            self.current_model_label.setText(f"分离模式: {display_name}")
            self.add_progress_info("🎛️ 分离模式已更新", display_name)
            
            # 记录模型选择到日志
            logger.info(f"🎛️ 用户选择了分离模式: {display_name} ({model_type})")

    def set_model(self, model):
        """
        设置选中的分离模型
        :param model: 模型名称 (2stems, 4stems, 6stems)
        """
        if model == "2stems" and hasattr(self, 'radio_2stems') and self.radio_2stems.isChecked():
            self.selected_model = "2stems"
            if hasattr(self, 'status_label'):
                self.status_label.setText(f"已选择模型: 2stems (人声 + 伴奏)")
        elif model == "4stems" and hasattr(self, 'radio_4stems') and self.radio_4stems.isChecked():
            self.selected_model = "4stems"
            if hasattr(self, 'status_label'):
                self.status_label.setText(f"已选择模型: 4stems (人声 + 鼓 + 贝斯 + 其他)")

    def select_audio_file(self):
        """
        打开文件选择对话框，让用户选择音频文件
        支持的格式：MP3, WAV, FLAC, M4A
        """
        logger.info("🎵 用户开始选择音频文件...")
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择音频文件", "", "音频文件 (*.mp3 *.wav *.flac *.m4a *.ogg);;所有文件 (*)"
        )

        if file_path:
            logger.info(f"📁 用户选择了文件: {file_path}")
            
            # 记录文件详细信息
            try:
                file_info = get_audio_file_info(file_path)
                if file_info:
                    logger.info(f"📊 文件信息 - 大小: {file_info['size_formatted']}, 格式: {file_info['extension']}")
                    logger.debug(f"📅 文件修改时间: {file_info['modified']}")
            except Exception as e:
                logger.warning(f"⚠️ 无法获取文件详细信息: {e}")
            
            self.selected_file_path = file_path
            # 设置输出目录为输入文件所在目录
            self.output_dir = os.path.dirname(file_path)
            self.output_dir_edit.setText(self.output_dir)
            logger.info(f"📂 自动设置输出目录: {self.output_dir}")
            
            # 显示文件路径
            display_path = file_path if len(file_path) < 45 else f"...{file_path[-42:]}"
            self.file_path_edit.setText(display_path)
            
            # 更新状态信息
            filename = os.path.basename(file_path)
            self.current_file_label.setText(f"当前文件: {filename}")
            self.set_status_with_color("状态: 文件已选择，准备就绪", "success")
            
            # 显示音频文件信息
            logger.debug("🌊 开始加载音频文件信息和波形预览...")
            self.display_audio_info(file_path)
            
            # 添加日志
            self.add_progress_info("🎵 音频文件已选择", filename)
            self.statusBar().showMessage(f"已选择文件: {filename}")
            logger.info(f"✅ 文件选择成功: {filename}")
        else:
            logger.info("❌ 用户取消了文件选择")

    def display_audio_info(self, file_path):
        """显示音频文件信息和波形预览"""
        try:
            # 获取文件基本信息
            file_info = get_audio_file_info(file_path)
            if file_info:
                filename = os.path.basename(file_path)
                file_ext = file_info['extension']
                
                # 更新音频信息标签
                info_text = f"文件: {filename} | 格式: {file_ext[1:]} | 大小: {file_info['size_formatted']}"
                self.audio_info_label.setText(info_text)
                
                # 加载音频波形
                if self.waveform_widget.load_audio_file(file_path):
                    self.add_progress_info("🌊 波形预览", "加载成功")
                else:
                    self.add_progress_info("⚠️ 波形预览", "加载失败，但不影响分离功能")
            
        except Exception as e:
            self.audio_info_label.setText(f"无法读取文件信息: {str(e)}")
            self.log_text.append(f"⚠️ 文件信息读取失败: {str(e)}")

    def select_output_directory(self):
        """
        打开目录选择对话框，让用户选择输出目录
        """
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择输出目录", self.output_dir, QFileDialog.ShowDirsOnly
        )

        if dir_path:
            self.output_dir = dir_path
            # 显示短路径，过长时截断
            display_dir = dir_path if len(dir_path) < 45 else f"...{dir_path[-42:]}"
            self.output_dir_edit.setText(display_dir)
            self.add_progress_info("📁 输出目录已设置", os.path.basename(dir_path))
            self.statusBar().showMessage(f"已选择输出目录: {os.path.basename(dir_path)}")

    def open_output_directory(self):
        """打开输出目录"""
        if os.path.exists(self.output_dir):
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.output_dir))
        else:
            QMessageBox.warning(self, "错误", "输出目录不存在！")

    def start_separation(self):
        """
        开始音乐分离处理
        检查必要条件，调用分离逻辑
        """
        logger.info("🚀 用户启动音频分离...")
        
        # 记录操作开始时间
        self.operation_start_time = time.time()
        
        # 检查是否选择了文件
        if not self.selected_file_path:
            logger.warning("❌ 用户未选择音频文件")
            self.set_status_with_color("状态: 错误 - 请先选择音频文件", "error")
            self.log_text.append("❌ 错误: 请先选择音频文件")
            QMessageBox.warning(self, "警告", "请先选择音频文件！")
            return

        # 记录分离参数
        logger.info(f"📁 输入文件: {self.selected_file_path}")
        logger.info(f"📂 输出目录: {self.output_dir}")
        logger.info(f"🎛️ 分离模式: {self.selected_model}")
        
        # 检查输入文件是否存在
        if not os.path.exists(self.selected_file_path):
            logger.error(f"❌ 输入文件不存在: {self.selected_file_path}")
            self.set_status_with_color("状态: 错误 - 输入文件不存在", "error")
            self.log_text.append("❌ 错误: 输入文件不存在")
            QMessageBox.critical(self, "错误", "输入文件不存在！")
            return

        # 检查输出目录是否存在
        if not os.path.exists(self.output_dir):
            try:
                logger.info(f"📁 创建输出目录: {self.output_dir}")
                os.makedirs(self.output_dir)
                self.add_progress_info("📁 输出目录已创建", os.path.basename(self.output_dir))
                logger.info("✅ 输出目录创建成功")
            except Exception as e:
                error_msg = f"错误: 无法创建输出目录 - {str(e)}"
                logger.error(f"❌ 输出目录创建失败: {e}")
                self.set_status_with_color(f"状态: {error_msg}", "error")
                self.log_text.append(f"❌ {error_msg}")
                QMessageBox.critical(self, "错误", error_msg)
                return

        # 检查磁盘空间
        try:
            import shutil
            free_space = shutil.disk_usage(self.output_dir).free / (1024**3)  # GB
            input_size = os.path.getsize(self.selected_file_path) / (1024**3)  # GB
            estimated_output_size = input_size * 3  # 估算输出大小
            
            logger.info(f"💾 磁盘可用空间: {free_space:.2f}GB")
            logger.info(f"📊 输入文件大小: {input_size:.2f}GB")
            logger.info(f"📈 预估输出大小: {estimated_output_size:.2f}GB")
            
            if free_space < estimated_output_size:
                logger.warning(f"⚠️ 磁盘空间可能不足")
                self.log_text.append(f"⚠️ 警告: 磁盘空间可能不足")
        except Exception as e:
            logger.debug(f"无法检查磁盘空间: {e}")

        # 更新状态
        self.set_status_with_color("状态: 开始分离...", "processing")
        self.add_separator("开始音频分离")
        self.add_progress_info("🚀 启动分离任务", f"文件: {os.path.basename(self.selected_file_path)}")
        self.add_progress_info("🎛️ 分离模式", self.selected_model)
        self.add_progress_info("📁 输出目录", os.path.basename(self.output_dir))

        # 禁用UI控件
        logger.debug("🔒 禁用UI控件...")
        self.select_file_button.setEnabled(False)
        self.select_output_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        # 禁用所有分离类型单选按钮
        for radio in self.model_radios.values():
            radio.setEnabled(False)

        # 创建并启动分离线程
        self.separation_thread = SeparationThread(
            self.selected_model,
            self.selected_file_path,
            self.output_dir
        )

        # 连接线程信号

        self.separation_thread.separation_completed.connect(self.on_separation_completed)
        self.separation_thread.separation_error.connect(self.on_separation_error)

        # 启动波形扫描动画
        self.waveform_widget.start_processing_animation()

        # 启动线程
        self.separation_thread.start()



    def update_progress(self, message):
        """
        更新进度信息
        :param message: 进度消息
        """
        # 如果是处理中的消息，启动动画
        if "初始化" in message or "开始" in message or "处理" in message:
            self.start_processing_animation(f"状态: {message}")
        else:
            self.set_status_with_color(f"状态: {message}", "processing")
        
        self.log_text.append(f"🔄 {message}")
        self.statusBar().showMessage(message)

    def update_status(self, message):
        """
        更新状态标签
        :param message: 状态消息
        """
        self.statusBar().showMessage(message)

    def on_separation_completed(self, output_files, output_dir):
        """
        分离完成处理
        :param output_files: 分离后的文件路径列表
        :param output_dir: 输出目录
        """
        # 停止处理动画
        self.stop_processing_animation()
        
        self.set_status_with_color(f"状态: 分离完成! 共生成{len(output_files)}个文件", "success")
        self.add_separator("分离完成")
        self.add_progress_info("🎉 任务完成", f"成功生成 {len(output_files)} 个音轨文件")
        self.add_progress_info("📂 输出位置", os.path.basename(output_dir))
        
        # 添加文件列表
        for i, file_path in enumerate(output_files, 1):
            filename = os.path.basename(file_path)
            file_size = os.path.getsize(file_path) // (1024 * 1024)  # MB
            self.add_progress_info(f"   {i}️⃣ {filename}", f"大小: {file_size}MB")
        
        self.add_separator()
        
        # 列出生成的文件
        for i, file_path in enumerate(output_files, 1):
            filename = os.path.basename(file_path)
            self.log_text.append(f"   {i}. {filename}")
        
        self.statusBar().showMessage(f"分离完成! 共生成{len(output_files)}个文件")
        
        # 停止波形扫描动画
        self.waveform_widget.stop_processing_animation()
        
        # 先重置UI状态，确保线程正确清理
        self.reset_ui_state()



    def on_separation_error(self, error_message):
        """
        分离错误处理
        :param error_message: 错误消息
        """
        # 停止处理动画
        self.stop_processing_animation()
        
        # 停止波形扫描动画
        self.waveform_widget.stop_processing_animation()
        
        self.set_status_with_color(f"状态: 分离失败", "error")
        self.log_text.append(f"❌ 分离失败: {error_message}")
        self.statusBar().showMessage(f"错误: {error_message}")
        self.reset_ui_state()

        # 显示错误对话框
        QMessageBox.critical(self, "❌ 分离失败", f"音频分离过程中发生错误:\n\n{error_message}\n\n请检查:\n• 音频文件是否完整\n• 模型文件是否存在\n• 输出目录是否有写入权限")

    

    def reset_ui_state(self):
        """
        重置UI控件状态
        """
        # 启用UI控件
        self.select_file_button.setEnabled(True)
        self.select_output_btn.setEnabled(True)
        self.start_btn.setEnabled(True)
        # 启用所有分离类型单选按钮
        for radio in self.model_radios.values():
            radio.setEnabled(True)

        # 正确清理线程
        if self.separation_thread and self.separation_thread.isRunning():
            self.separation_thread.quit()
            self.separation_thread.wait()
        self.separation_thread = None

    def closeEvent(self, event):
        """
        窗口关闭事件处理
        确保线程正确清理，防止闪退
        """
        try:
            # 清理分离线程
            if self.separation_thread and self.separation_thread.isRunning():
                self.separation_thread.is_running = False
                self.separation_thread.quit()
                self.separation_thread.wait(3000)  # 等待最多3秒
            
            # 清理音频加载线程
            if hasattr(self, 'waveform_widget') and hasattr(self.waveform_widget, 'load_thread'):
                if self.waveform_widget.load_thread and self.waveform_widget.load_thread.isRunning():
                    self.waveform_widget.load_thread.quit()
                    self.waveform_widget.load_thread.wait(2000)
            
            event.accept()
        except Exception as e:
            print(f"关闭窗口时出错: {e}")
            event.accept()

    def update_processing_animation(self):
        """更新处理动画效果"""
        if hasattr(self, 'is_processing') and self.is_processing:
            # 循环显示不同数量的点
            dots = "." * (self.processing_dots % 4)
            if self.processing_dots % 4 == 0:
                dots = ""
            
            # 更新状态标签的文字
            animated_text = self.base_processing_text + dots
            if hasattr(self, 'progress_label'):
                self.progress_label.setText(animated_text)
            
            self.processing_dots += 1
    
    def start_processing_animation(self, base_text):
        """开始处理动画"""
        self.is_processing = True
        self.base_processing_text = base_text
        self.processing_dots = 0
        if hasattr(self, 'processing_timer'):
            self.processing_timer.start(500)  # 每500ms更新一次
    
    def stop_processing_animation(self):
        """停止处理动画"""
        self.is_processing = False
        if hasattr(self, 'processing_timer'):
            self.processing_timer.stop()

    def apply_theme(self):
        """应用深色主题样式"""
        from config import ThemeManager
        
        # 获取深色主题的样式
        theme_styles = ThemeManager.get_theme_styles()
        
        # 应用主窗口样式
        self.setStyleSheet(theme_styles['main_window'])
        
        # 应用各个组件的样式
        if hasattr(self, 'start_btn'):
            self.start_btn.setStyleSheet(theme_styles['button'])
        if hasattr(self, 'open_dir_btn'):
            self.open_dir_btn.setStyleSheet(theme_styles['button'])
        if hasattr(self, 'select_file_button'):
            self.select_file_button.setStyleSheet(theme_styles['button'])
        if hasattr(self, 'select_output_btn'):
            self.select_output_btn.setStyleSheet(theme_styles['button'])
        if hasattr(self, 'settings_btn'):
            self.settings_btn.setStyleSheet(theme_styles['button'])
        
        # 应用分组框样式
        if hasattr(self, 'left_panel'):
            for child in self.left_panel.findChildren(QGroupBox):
                child.setStyleSheet(theme_styles['group_box'])
        
        if hasattr(self, 'right_panel'):
            for child in self.right_panel.findChildren(QGroupBox):
                child.setStyleSheet(theme_styles['group_box'])
        
        # 应用文本输入框样式
        if hasattr(self, 'file_path_edit'):
            self.file_path_edit.setStyleSheet(theme_styles['line_edit'])
        if hasattr(self, 'output_dir_edit'):
            self.output_dir_edit.setStyleSheet(theme_styles['line_edit'])
        
        # 应用文本编辑框样式
        if hasattr(self, 'log_text'):
            self.log_text.setStyleSheet(theme_styles['text_edit'])
        
        # 应用单选按钮样式
        if hasattr(self, 'model_radios'):
            for radio in self.model_radios.values():
                radio.setStyleSheet(theme_styles['radio_button'])
        
        # 应用状态栏样式
        if hasattr(self, 'statusBar'):
            self.statusBar().setStyleSheet(theme_styles['status_bar'])
        
        # 应用标签样式
        if hasattr(self, 'left_panel'):
            for child in self.left_panel.findChildren(QLabel):
                if not child.styleSheet():  # 只对没有自定义样式的标签应用样式
                    child.setStyleSheet(f"color: {ThemeManager.DARK_COLORS['TEXT_PRIMARY']}; font-size: 13px;")
        
        if hasattr(self, 'right_panel'):
            for child in self.right_panel.findChildren(QLabel):
                if not child.styleSheet():  # 只对没有自定义样式的标签应用样式
                    child.setStyleSheet(f"color: {ThemeManager.DARK_COLORS['TEXT_PRIMARY']}; font-size: 13px;")
        
        # 更新状态信息面板的样式
        if hasattr(self, 'status_info'):
            self.status_info.setStyleSheet("""
                background-color: rgba(74, 144, 226, 0.1);
                border: 1px solid #4a90e2;
                border-radius: 8px;
                padding: 15px;
            """)
        
        # 更新波形显示区域样式
        if hasattr(self, 'waveform_widget'):
            self.waveform_widget.setStyleSheet("""
                background-color: #2a2a2a;
                border: 1px solid #404040;
                border-radius: 6px;
            """)
        
        # 更新进度条样式
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #404040;
                    border-radius: 6px;
                    text-align: center;
                    color: #ffffff;
                    background-color: #2a2a2a;
                }
                QProgressBar::chunk {
                    background-color: #4a90e2;
                    border-radius: 5px;
                }
            """)
        
        # 更新下拉框样式
        if hasattr(self, 'left_panel'):
            for child in self.left_panel.findChildren(QComboBox):
                child.setStyleSheet(theme_styles.get('combo_box', f"""
                    QComboBox {{
                        background-color: {ThemeManager.DARK_COLORS['WIDGET_BACKGROUND']};
                        border: 2px solid {ThemeManager.DARK_COLORS['BORDER_NORMAL']};
                        border-radius: 6px;
                        padding: 5px 10px;
                        color: {ThemeManager.DARK_COLORS['TEXT_PRIMARY']};
                        font-size: 13px;
                    }}
                    QComboBox:hover {{
                        border: 2px solid {ThemeManager.DARK_COLORS['BORDER_HOVER']};
                    }}
                    QComboBox::drop-down {{
                        subcontrol-origin: padding;
                        subcontrol-position: top right;
                        width: 15px;
                        border-left-width: 1px;
                        border-left-color: {ThemeManager.DARK_COLORS['BORDER_NORMAL']};
                        border-left-style: solid;
                        border-top-right-radius: 6px;
                        border-bottom-right-radius: 6px;
                    }}
                    QComboBox::down-arrow {{
                        image: none;
                        border-left: 5px solid transparent;
                        border-right: 5px solid transparent;
                        border-top: 5px solid {ThemeManager.DARK_COLORS['TEXT_SECONDARY']};
                    }}
                    QComboBox QAbstractItemView {{
                        background-color: {ThemeManager.DARK_COLORS['WIDGET_BACKGROUND']};
                        border: 1px solid {ThemeManager.DARK_COLORS['BORDER_NORMAL']};
                        color: {ThemeManager.DARK_COLORS['TEXT_PRIMARY']};
                        selection-background-color: {ThemeManager.DARK_COLORS['PRIMARY']};
                    }}
                """))
        
        if hasattr(self, 'right_panel'):
            for child in self.right_panel.findChildren(QComboBox):
                child.setStyleSheet(theme_styles.get('combo_box', f"""
                    QComboBox {{
                        background-color: {ThemeManager.DARK_COLORS['WIDGET_BACKGROUND']};
                        border: 2px solid {ThemeManager.DARK_COLORS['BORDER_NORMAL']};
                        border-radius: 6px;
                        padding: 5px 10px;
                        color: {ThemeManager.DARK_COLORS['TEXT_PRIMARY']};
                        font-size: 13px;
                    }}
                    QComboBox:hover {{
                        border: 2px solid {ThemeManager.DARK_COLORS['BORDER_HOVER']};
                    }}
                    QComboBox::drop-down {{
                        subcontrol-origin: padding;
                        subcontrol-position: top right;
                        width: 15px;
                        border-left-width: 1px;
                        border-left-color: {ThemeManager.DARK_COLORS['BORDER_NORMAL']};
                        border-left-style: solid;
                        border-top-right-radius: 6px;
                        border-bottom-right-radius: 6px;
                    }}
                    QComboBox::down-arrow {{
                        image: none;
                        border-left: 5px solid transparent;
                        border-right: 5px solid transparent;
                        border-top: 5px solid {ThemeManager.DARK_COLORS['TEXT_SECONDARY']};
                    }}
                    QComboBox QAbstractItemView {{
                        background-color: {ThemeManager.DARK_COLORS['WIDGET_BACKGROUND']};
                        border: 1px solid {ThemeManager.DARK_COLORS['BORDER_NORMAL']};
                        color: {ThemeManager.DARK_COLORS['TEXT_PRIMARY']};
                        selection-background-color: {ThemeManager.DARK_COLORS['PRIMARY']};
                    }}
                """))
        
        # 更新复选框样式
        if hasattr(self, 'left_panel'):
            for child in self.left_panel.findChildren(QCheckBox):
                child.setStyleSheet(f"""
                    QCheckBox {{
                        color: {ThemeManager.DARK_COLORS['TEXT_PRIMARY']};
                        font-size: 13px;
                        spacing: 8px;
                    }}
                    QCheckBox::indicator {{
                        width: 16px;
                        height: 16px;
                    }}
                    QCheckBox::indicator:unchecked {{
                        border: 2px solid {ThemeManager.DARK_COLORS['BORDER_NORMAL']};
                        border-radius: 3px;
                        background-color: {ThemeManager.DARK_COLORS['WIDGET_BACKGROUND']};
                    }}
                    QCheckBox::indicator:checked {{
                        border: 2px solid {ThemeManager.DARK_COLORS['PRIMARY']};
                        border-radius: 3px;
                        background-color: {ThemeManager.DARK_COLORS['PRIMARY']};
                        image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iOSIgdmlld0JveD0iMCAwIDEyIDkiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik0xMSAwLjVMMy41IDhMMSA1LjUiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+Cjwvc3ZnPgo=);
                    }}
                    QCheckBox:hover {{
                        color: {ThemeManager.DARK_COLORS['PRIMARY']};
                    }}
                """)
        
        if hasattr(self, 'right_panel'):
            for child in self.right_panel.findChildren(QCheckBox):
                child.setStyleSheet(f"""
                    QCheckBox {{
                        color: {ThemeManager.DARK_COLORS['TEXT_PRIMARY']};
                        font-size: 13px;
                        spacing: 8px;
                    }}
                    QCheckBox::indicator {{
                        width: 16px;
                        height: 16px;
                    }}
                    QCheckBox::indicator:unchecked {{
                        border: 2px solid {ThemeManager.DARK_COLORS['BORDER_NORMAL']};
                        border-radius: 3px;
                        background-color: {ThemeManager.DARK_COLORS['WIDGET_BACKGROUND']};
                    }}
                    QCheckBox::indicator:checked {{
                        border: 2px solid {ThemeManager.DARK_COLORS['PRIMARY']};
                        border-radius: 3px;
                        background-color: {ThemeManager.DARK_COLORS['PRIMARY']};
                        image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iOSIgdmlld0JveD0iMCAwIDEyIDkiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik0xMSAwLjVMMy41IDhMMSA1LjUiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+Cjwvc3ZnPgo=);
                    }}
                    QCheckBox:hover {{
                        color: {ThemeManager.DARK_COLORS['PRIMARY']};
                    }}
                """)

    def save_theme_preference(self):
        """保存主题偏好到配置文件"""
        try:
            import json
            config_file = os.path.join('pretrained_models', 'theme_preference.json')
            
            theme_config = {
                'is_dark_theme': self.is_dark_theme,
                'theme': 'dark' if self.is_dark_theme else 'light'
            }
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(theme_config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"主题偏好已保存: {'dark' if self.is_dark_theme else 'light'}")
        except Exception as e:
            logger.error(f"保存主题偏好失败: {e}")

    def load_theme_preference(self):
        """从配置文件加载主题偏好"""
        try:
            import json
            config_file = os.path.join('pretrained_models', 'theme_preference.json')
            
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    theme_config = json.load(f)
                
                self.is_dark_theme = theme_config.get('is_dark_theme', True)
                logger.info(f"已加载主题偏好: {'dark' if self.is_dark_theme else 'light'}")
            else:
                # 默认深色主题
                self.is_dark_theme = True
                logger.info("使用默认深色主题")
        except Exception as e:
            logger.error(f"加载主题偏好失败: {e}")
            self.is_dark_theme = True

    def set_status_with_color(self, status_text, status_type="info"):
        """
        设置状态文字并根据状态类型应用不同颜色
        :param status_text: 状态文字
        :param status_type: 状态类型 - "info"(蓝色), "success"(绿色), "warning"(橙色), "error"(红色), "processing"(黄色)
        """
        from config import ThemeManager
        
        # 获取当前主题的颜色
        colors = ThemeManager.DARK_COLORS if self.is_dark_theme else ThemeManager.LIGHT_COLORS
        
        color_map = {
            "info": colors['INFO'],      # 蓝色 - 信息状态
            "success": colors['SUCCESS'],   # 绿色 - 成功状态
            "warning": colors['WARNING'],   # 橙色 - 警告状态
            "error": colors['ERROR'],     # 红色 - 错误状态
            "processing": "#fd7e14" # 橙色 - 处理中状态（保持通用）
        }
        
        color = color_map.get(status_type, colors['INFO'])
        
        # 更新状态标签的颜色和文字
        if hasattr(self, 'progress_label'):
            self.progress_label.setStyleSheet(f"color: {color}; font-size: 13px; margin: 3px 0; font-weight: bold;")
            self.progress_label.setText(status_text)

if __name__ == "__main__":
    """
    应用入口函数
    创建应用实例和主窗口，并启动事件循环
    """
    app = QApplication(sys.argv)

    # 检查依赖项
    if not check_dependencies():
        sys.exit(1)

    log_file_path = setup_detailed_logging()
    log_system_info()
    
    # 检查GPU可用性
    gpu_available = check_gpu_availability()
    app_config = AppConfig()
    AppConfig.ENABLE_GPU = gpu_available
    
    window = MusicSeparatorMainWindow()
    window.show()
    sys.exit(app.exec_())
