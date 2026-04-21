"""模型实现"""
import os
import sys
import logging
import shutil
from pathlib import Path

# 设置ffmpeg路径
ffmpeg_path = os.path.join(os.path.dirname(__file__), 'ffmpeg', 'bin')
if os.path.exists(ffmpeg_path):
    os.environ['PATH'] = ffmpeg_path + os.pathsep + os.environ['PATH']

# 导入Demucs相关模块
try:
    from demucs import pretrained
    from demucs.apply import apply_model
    from demucs.audio import AudioFile, save_audio
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False
    logging.warning("Demucs模块未安装，部分功能将不可用")

class ModelNotFoundError(Exception):
    """模型未找到异常"""
    def __init__(self, message, model_type=None, download_url=None):
        super().__init__(message)
        self.model_type = model_type
        self.download_url = download_url

class BaseModel:
    """基础模型类"""
    
    def __init__(self, model_type):
        """初始化模型

        Args:
            model_type: 模型类型
        """
        self.model_type = model_type
        self.logger = logging.getLogger(__name__)

    def separate(self, audio_path, output_dir):
        """分离音频

        Args:
            audio_path: 输入音频文件路径
            output_dir: 输出目录路径

        Returns:
            分离后的音频文件路径列表
        """
        raise NotImplementedError("子类必须实现separate方法")

class DemucsModel(BaseModel):
    """Demucs模型实现"""
    
    # 模型类型映射
    MODEL_TYPE_MAP = {
        2: 'htdemucs_ft',
        4: 'htdemucs',
        6: 'htdemucs_6s'
    }
    
    # 轨道名称映射
    TRACK_NAMES = {
        2: ['vocals', 'accompaniment'],
        4: ['drums', 'bass', 'other', 'vocals'],
        6: ['drums', 'bass', 'other', 'vocals', 'piano', 'guitar']
    }
    
    def __init__(self, model_type):
        """初始化Demucs模型

        Args:
            model_type: 模型类型，格式为"demucs:stems"
        """
        if not DEMUCS_AVAILABLE:
            raise ImportError("Demucs模块未安装，无法使用Demucs模型")
            
        super().__init__(model_type)
        
        # 解析模型类型
        self.model_name, self.stems = self._parse_model_type(model_type)
        
        # 验证轨道数
        if self.stems not in self.MODEL_TYPE_MAP:
            raise ValueError(f"Demucs模型只支持{list(self.MODEL_TYPE_MAP.keys())}轨分离，当前: {self.stems}轨")
        
        self.model_type = self.MODEL_TYPE_MAP[self.stems]
        self.model = self._load_model()
        self.model_path = self._get_model_path()
        
        self.logger.info(f"成功初始化Demucs {self.stems}轨模型")
        self.logger.info(f"模型路径: {self.model_path}")
    
    def _parse_model_type(self, model_type):
        """解析模型类型字符串"""
        if ':' not in model_type:
            raise ValueError(f"无效的模型类型: {model_type}，格式应为 'demucs:stems'")
        
        parts = model_type.split(':')
        if len(parts) != 2:
            raise ValueError(f"无效的模型类型: {model_type}")
        
        model_name = parts[0]
        stems_str = parts[1].replace('stems', '')
        
        try:
            stems = int(stems_str)
        except ValueError:
            raise ValueError(f"无效的轨道数: {parts[1]}")
        
        return model_name, stems
    
    def _load_model(self):
        """加载Demucs模型"""
        # 确保在项目文件夹内创建pretrained_models目录
        pretrained_models_dir = os.path.join(os.path.dirname(__file__), 'pretrained_models')
        os.makedirs(pretrained_models_dir, exist_ok=True)
        
        # 尝试复制远程模型配置文件到本地目录
        remote_dir = Path(__file__).parent.parent / 'Lib' / 'site-packages' / 'demucs' / 'remote'
        if remote_dir.exists():
            for file in remote_dir.glob('*.txt'):
                dest_file = Path(pretrained_models_dir) / file.name
                if not dest_file.exists():
                    shutil.copy(file, dest_file)
                    self.logger.info(f"复制模型配置文件: {file.name}")
        
        # 使用项目的pretrained_models目录作为模型库
        repo_path = Path(pretrained_models_dir)
        self.logger.info(f"使用模型库路径: {repo_path}")
        
        # 尝试使用本地模型库
        try:
            model = pretrained.get_model(self.model_type, repo=repo_path)
            self.logger.info(f"模型加载成功: {self.model_type}")
            return model
        except Exception as e:
            # 如果本地模型库加载失败，尝试不指定repo参数
            self.logger.warning(f"本地模型库加载失败: {str(e)}，尝试从远程下载")
            model = pretrained.get_model(self.model_type)
            self.logger.info(f"模型加载成功: {self.model_type}")
            return model
    
    def _get_model_path(self):
        """获取模型路径"""
        try:
            if hasattr(self.model, 'path'):
                return self.model.path
            elif hasattr(self.model, 'repo'):
                return self.model.repo
            else:
                return f"Demucs模型: {self.model_type}"
        except Exception:
            return f"Demucs模型: {self.model_type}"

    def separate(self, audio_path, output_dir, progress_callback=None):
        """使用Demucs模型分离音频

        Args:
            audio_path: 输入音频文件路径
            output_dir: 输出目录路径
            progress_callback: 进度回调函数，接收(当前步骤, 总步骤, 描述)参数

        Returns:
            分离后的音频文件路径列表
        """
        import os
        import time
        import torch
        import gc
        
        def report_progress(step, total, description):
            """报告进度"""
            if progress_callback:
                progress_callback(step, total, description)
            self.logger.info(f"[进度] {step}/{total} - {description}")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        self.logger.info(f"开始分离音频: {audio_path}")
        self.logger.info(f"使用模型: {self.model_type}")
        self.logger.info(f"输出目录: {output_dir}")

        try:
            start_time = time.time()
            total_steps = 5
            current_step = 0
            
            # 步骤1: 验证音频文件
            current_step += 1
            report_progress(current_step, total_steps, "验证音频文件...")
            
            from utils import validate_audio_file
            is_valid, error_msg, file_info = validate_audio_file(audio_path)
            if not is_valid:
                raise ValueError(f"音频文件验证失败: {error_msg}")
            
            self.logger.info(f"[文件信息] 大小: {file_info['size_formatted']}, "
                           f"时长: {file_info.get('duration', '未知')}秒")
            
            # 步骤2: 加载模型
            current_step += 1
            report_progress(current_step, total_steps, "加载AI模型...")
            self.logger.info("[模型加载] 开始加载模型...")
            self.logger.debug(f"[模型加载] 模型类型: {self.model_type}")
            self.logger.info(f"[模型加载] 成功加载 {self.model_type} 模型")
        
            # 步骤3: 读取音频文件
            current_step += 1
            report_progress(current_step, total_steps, "读取音频文件...")
            self.logger.info("[音频处理] 开始读取音频文件...")
            self.logger.debug(f"[音频处理] 输入文件: {audio_path}")
            
            try:
                f = AudioFile(audio_path)
                wav = f.read(streams=0, samplerate=self.model.samplerate)
                wav = wav.mean(0, keepdim=True)
            except Exception as e:
                raise RuntimeError(f"读取音频文件失败: {str(e)}")
        
            # 步骤4: 执行分离
            current_step += 1
            report_progress(current_step, total_steps, "AI推理分离音轨...")
            
            # 移动到GPU（如果可用）
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.logger.info(f"[设备信息] 使用设备: {device}")
            wav = wav.to(device)
        
            # 添加batch维度
            wav = wav.unsqueeze(0)
        
            # 执行分离
            self.logger.info("[音频分离] 开始Demucs分离处理...")
            
            # 使用配置参数
            shifts = getattr(self, 'shifts', 1)
            overlap = getattr(self, 'overlap', 0.25)
            
            sources = apply_model(
                self.model,
                wav,
                device=device,
                shifts=shifts,
                overlap=overlap,
                progress=True
            )
            
            # 确保sources是列表
            if not isinstance(sources, list):
                sources = [sources]
            
            # 确保sources是二维的
            if sources and len(sources[0].shape) == 3:
                sources = sources[0]
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"[音频分离] 模型推理完成，分离出{len(sources)}个轨道")
            
            # 步骤5: 保存输出文件
            current_step += 1
            report_progress(current_step, total_steps, "保存分离后的音轨...")
            
            # 生成输出文件
            output_files = []
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
        
            # 根据轨道数处理输出
            if self.stems == 2:
                # 2轨模式：人声 + 伴奏
                output_files = self._save_2stems(sources, base_name, output_dir)
            else:
                # 4轨或6轨模式：分别保存每个轨道
                output_files = self._save_multistem(sources, base_name, output_dir)
        
            self.logger.info(f"[处理总结] 音频分离完成，耗时: {elapsed_time:.2f}秒")
            self.logger.info(f"[处理总结] 共生成{len(output_files)}个轨道文件，保存至: {output_dir}")
            
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            return output_files
            
        except KeyboardInterrupt:
            self.logger.info("用户中断了处理过程")
            return []
        except Exception as e:
            self.logger.error(f"Demucs处理失败: {str(e)}", exc_info=True)
            raise
    
    def _save_2stems(self, sources, base_name, output_dir):
        """保存2轨分离结果（人声+伴奏）
        
        Args:
            sources: 分离后的音频源列表
            base_name: 输出文件基础名称
            output_dir: 输出目录
            
        Returns:
            list: 输出文件路径列表
        """
        import torch
        output_files = []
        
        # 假设第一个轨道是人声，其余轨道合并为伴奏
        vocals = sources[0]
        accompaniment = sum(sources[1:]) if len(sources) > 1 else torch.zeros_like(vocals)
        
        # 保存人声轨道
        vocals_path = os.path.join(output_dir, f"{base_name}_vocals.wav")
        save_audio(vocals.cpu(), vocals_path, samplerate=self.model.samplerate)
        output_files.append(vocals_path)
        self.logger.info(f"[轨道生成] vocals: {vocals_path}")
        
        # 保存伴奏轨道
        accompaniment_path = os.path.join(output_dir, f"{base_name}_accompaniment.wav")
        save_audio(accompaniment.cpu(), accompaniment_path, samplerate=self.model.samplerate)
        output_files.append(accompaniment_path)
        self.logger.info(f"[轨道生成] accompaniment: {accompaniment_path}")
        
        return output_files
    
    def _save_multistem(self, sources, base_name, output_dir):
        """保存多轨分离结果
        
        Args:
            sources: 分离后的音频源列表
            base_name: 输出文件基础名称
            output_dir: 输出目录
            
        Returns:
            list: 输出文件路径列表
        """
        output_files = []
        
        # 获取轨道名称
        track_names = self.TRACK_NAMES.get(self.stems, [])
        if not track_names:
            track_names = [f"stem_{i}" for i in range(len(sources))]
        
        for i, stem_name in enumerate(track_names):
            if i < len(sources):
                stem_audio = sources[i].cpu()
                stem_path = os.path.join(output_dir, f"{base_name}_{stem_name}.wav")
                save_audio(stem_audio, stem_path, samplerate=self.model.samplerate)
                output_files.append(stem_path)
                self.logger.info(f"[轨道生成] {stem_name}: {stem_path}")
        
        return output_files

class ModelFactory:
    @staticmethod
    def create_model(model_type):
        """根据模型类型创建相应的模型实例

        Args:
            model_type: 模型类型字符串，格式为"model_name:stems"或"model_name:model_name"

        Returns:
            模型实例
        """
        if not model_type:
            raise ValueError("模型类型不能为空")

        # 处理不同的模型类型格式
        if ':' in model_type:
            model_name, param = model_type.split(':', 1)
        else:
            # 对于某些模型，可能没有冒号分隔符
            model_name = model_type
            param = None

        # 标准化模型名称
        model_name = model_name.lower()

        # 根据模型名称创建相应的模型实例
        if model_name == 'demucs':
            return DemucsModel(model_type)
        else:
            raise ValueError(f"不支持的模型类型: {model_name}")

    @staticmethod
    def get_available_models():
        """获取所有可用的模型类型和描述
        
        Returns:
            dict: 模型类型到描述的映射
        """
        return {
            # Demucs 模型
            "demucs:2stems": "Demucs 2轨 (人声 + 伴奏)",
            "demucs:4stems": "Demucs 4轨 (人声 + 鼓 + 贝斯 + 其他)",
            "demucs:6stems": "Demucs 6轨 (人声 + 鼓 + 贝斯 + 钢琴 + 吉他 + 其他)"
        }
