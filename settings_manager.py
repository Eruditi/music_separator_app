# -*- coding: utf-8 -*-
"""
设置管理模块 - 用户配置持久化
支持JSON格式的配置文件读写
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from config import AppConfig

logger = logging.getLogger(__name__)


class SettingsManager:
    """设置管理器 - 处理用户配置的保存和加载"""
    
    def __init__(self) -> None:
        """初始化设置管理器"""
        self.config_file: str = os.path.join(AppConfig.BASE_DIR, 'settings.json')
        self.default_settings: Dict[str, Any] = self._get_default_settings()
        self.current_settings: Dict[str, Any] = self.load_settings()
    
    def _get_default_settings(self) -> Dict[str, Any]:
        """获取默认设置"""
        return {
            "audio": {
                "sample_rate": 44100,
                "bit_depth": 16,
                "quality": 3
            },
            "model": {
                "shifts": 1,
                "overlap": 0.25,
                "segment": 10,
                "tta": False,
                "post_process": True,
                "batch_size": 1
            },
            "performance": {
                "use_gpu": True,
                "threads": 4,
                "cache_size": "1 GB"
            },
            "output": {
                "format": "wav",
                "auto_normalize": True,
                "save_metadata": True
            },
            "ui": {
                "theme": "dark",
                "language": "zh_CN",
                "show_waveform": True,
                "auto_check_update": True
            },
            "paths": {
                "last_input_dir": "",
                "last_output_dir": "",
                "default_output_dir": ""
            }
        }
    
    def load_settings(self) -> Dict[str, Any]:
        """
        从配置文件加载设置
        
        Returns:
            Dict[str, Any]: 当前设置字典
        """
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_settings: Dict[str, Any] = json.load(f)
                
                # 合并默认设置和加载的设置（处理新增配置项）
                merged_settings = self._merge_settings(self.default_settings, loaded_settings)
                logger.info(f"✅ 设置已加载: {self.config_file}")
                return merged_settings
            else:
                logger.info("📄 配置文件不存在，使用默认设置")
                return self.default_settings.copy()
                
        except json.JSONDecodeError as e:
            logger.error(f"❌ 配置文件格式错误: {str(e)}，使用默认设置")
            return self.default_settings.copy()
        except Exception as e:
            logger.error(f"❌ 加载设置失败: {str(e)}，使用默认设置")
            return self.default_settings.copy()
    
    def save_settings(self, settings: Optional[Dict[str, Any]] = None) -> bool:
        """
        保存设置到配置文件
        
        Args:
            settings: 要保存的设置字典，如果为None则保存当前设置
            
        Returns:
            bool: 保存是否成功
        """
        try:
            settings_to_save = settings if settings is not None else self.current_settings
            
            # 确保目录存在
            config_dir = os.path.dirname(self.config_file)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir, exist_ok=True)
            
            # 保存到文件
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(settings_to_save, f, ensure_ascii=False, indent=2)
            
            self.current_settings = settings_to_save
            logger.info(f"💾 设置已保存: {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 保存设置失败: {str(e)}")
            return False
    
    def get_setting(self, key_path: Union[str, List[str]], default: Any = None) -> Any:
        """
        获取指定路径的设置值
        
        Args:
            key_path: 设置路径，如 "audio.sample_rate" 或 ["audio", "sample_rate"]
            default: 默认值
            
        Returns:
            Any: 设置值或默认值
        """
        try:
            if isinstance(key_path, str):
                keys = key_path.split('.')
            else:
                keys = key_path
            
            value: Any = self.current_settings
            for key in keys:
                value = value[key]
            return value
            
        except (KeyError, TypeError):
            return default
    
    def set_setting(self, key_path: Union[str, List[str]], value: Any) -> bool:
        """
        设置指定路径的设置值
        
        Args:
            key_path: 设置路径，如 "audio.sample_rate" 或 ["audio", "sample_rate"]
            value: 要设置的值
            
        Returns:
            bool: 设置是否成功
        """
        try:
            if isinstance(key_path, str):
                keys = key_path.split('.')
            else:
                keys = key_path
            
            # 遍历到倒数第二个key
            target: Dict[str, Any] = self.current_settings
            for key in keys[:-1]:
                if key not in target:
                    target[key] = {}
                target = target[key]
            
            # 设置最终值
            target[keys[-1]] = value
            return True
            
        except Exception as e:
            logger.error(f"❌ 设置值失败 {key_path}: {str(e)}")
            return False
    
    def reset_to_defaults(self) -> bool:
        """
        重置为默认设置
        
        Returns:
            bool: 重置是否成功
        """
        self.current_settings = self.default_settings.copy()
        return self.save_settings()
    
    def _merge_settings(self, default: Dict[str, Any], loaded: Dict[str, Any]) -> Dict[str, Any]:
        """
        递归合并设置，确保新增的配置项也被包含
        
        Args:
            default: 默认设置
            loaded: 已加载的设置
            
        Returns:
            Dict[str, Any]: 合并后的设置
        """
        result = default.copy()
        
        for key, value in loaded.items():
            if key in result:
                if isinstance(value, dict) and isinstance(result[key], dict):
                    result[key] = self._merge_settings(result[key], value)
                else:
                    result[key] = value
            else:
                result[key] = value
        
        return result
    
    def validate_settings(self) -> Tuple[bool, List[str]]:
        """
        验证当前设置的有效性
        
        Returns:
            Tuple[bool, List[str]]: (是否有效, 错误信息列表)
        """
        errors: List[str] = []
        
        # 验证音频设置
        audio: Dict[str, Any] = self.current_settings.get('audio', {})
        if audio.get('sample_rate') not in [22050, 44100, 48000]:
            errors.append("无效的采样率设置")
        if audio.get('bit_depth') not in [16, 24, 32]:
            errors.append("无效的位深度设置")
        
        # 验证模型设置
        model: Dict[str, Any] = self.current_settings.get('model', {})
        if not (1 <= model.get('shifts', 1) <= 10):
            errors.append("shifts值必须在1-10之间")
        if not (0.1 <= model.get('overlap', 0.25) <= 0.5):
            errors.append("overlap值必须在0.1-0.5之间")
        
        # 验证性能设置
        performance: Dict[str, Any] = self.current_settings.get('performance', {})
        if not (1 <= performance.get('threads', 4) <= 16):
            errors.append("线程数必须在1-16之间")
        
        return len(errors) == 0, errors


# 全局设置管理器实例
_settings_manager: Optional[SettingsManager] = None

def get_settings_manager() -> SettingsManager:
    """获取全局设置管理器实例（单例模式）"""
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = SettingsManager()
    return _settings_manager
