#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from pathlib import Path
import subprocess
import torch
import numpy as np
from datetime import datetime
import json
import time
import psutil

class DRKGEmbeddingTrainer:
    """DRKG嵌入模型训练器"""
    
    def __init__(self, data_dir='data', model_name='TransE', 
                 embedding_dim=400, batch_size=1024, 
                 neg_samples=256, lr=0.001, max_epochs=500,
                 gpu_id=-1, save_dir='results/models',
                 use_mix_cpu_gpu=False):
        """
        初始化DRKG嵌入模型训练器
        
        Args:
            data_dir (str): 数据目录
            model_name (str): 模型名称 ('TransE', 'RotatE', 'DistMult', 'ComplEx')
            embedding_dim (int): 嵌入维度
            batch_size (int): 批次大小
            neg_samples (int): 负采样数量
            lr (float): 学习率
            max_epochs (int): 最大训练轮数
            gpu_id (int): GPU ID，-1表示使用CPU
            save_dir (str): 模型保存目录
            use_mix_cpu_gpu (bool): 是否使用混合CPU-GPU模式
        """
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / 'processed'
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.neg_samples = neg_samples
        self.lr = lr
        self.max_epochs = max_epochs
        self.gpu_id = gpu_id
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_mix_cpu_gpu = use_mix_cpu_gpu
        
        # 生成模型ID
        self.model_id = f"{model_name}_{embedding_dim}d_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.model_save_path = self.save_dir / self.model_id
        
        # 日志文件
        self.log_file = self.save_dir / f"{self.model_id}_training.log"
        
        # 检测GPU信息
        self.detect_gpu()
        
        # 确保数据已处理
        if not self.processed_dir.exists() or not (self.processed_dir / 'train.tsv').exists():
            raise FileNotFoundError(f"找不到处理后的数据文件。请先运行数据处理步骤。")
    
    def detect_gpu(self):
        """检测GPU信息并优化配置"""
        if self.gpu_id >= 0 and torch.cuda.is_available():
            try:
                # 获取GPU信息
                self.gpu_name = torch.cuda.get_device_name(self.gpu_id)
                self.gpu_memory = torch.cuda.get_device_properties(self.gpu_id).total_memory / (1024**3)  # GB
                
                print(f"使用GPU: {self.gpu_name}，显存: {self.gpu_memory:.2f} GB")
                
                # 如果是RTX 4090（24GB），优化配置
                if "4090" in self.gpu_name and self.gpu_memory > 20:
                    # 为24GB显存的4090优化
                    if self.embedding_dim <= 400:
                        print("检测到RTX 4090 (24GB)，进行配置优化")
                        # 根据模型设置最优参数
                        if self.model_name == 'TransE' or self.model_name == 'DistMult':
                            self.batch_size = min(self.batch_size, 8192)
                            self.neg_samples = min(self.neg_samples, 512)
                        elif self.model_name == 'RotatE' or self.model_name == 'ComplEx':
                            self.batch_size = min(self.batch_size, 4096)
                            self.neg_samples = min(self.neg_samples, 256)
                    # 对于较大维度，需要减小批次大小
                    elif self.embedding_dim <= 800:
                        if self.model_name == 'TransE' or self.model_name == 'DistMult':
                            self.batch_size = min(self.batch_size, 4096)
                        else:
                            self.batch_size = min(self.batch_size, 2048)
                            
                    print(f"优化后的配置: batch_size={self.batch_size}, neg_samples={self.neg_samples}")
                    
                    # 对于大模型，自动开启混合CPU-GPU模式
                    if self.embedding_dim >= 800:
                        self.use_mix_cpu_gpu = True
                        print("自动开启混合CPU-GPU模式以支持大维度嵌入")
            except Exception as e:
                print(f"GPU检测出错: {e}")
    
    def prepare_training_config(self):
        """准备DGL-KE的训练配置"""
        
        # 检查系统内存
        system_memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"系统内存: {system_memory_gb:.2f} GB")
        
        # DGL-KE配置
        config = {
            '--model_name': self.model_name,
            '--data_path': str(self.processed_dir),
            '--dataset': 'DRKG',
            '--format': 'raw_udd_hrt',
            '--data_files': ['train.tsv', 'valid.tsv', 'test.tsv'],
            '--save_path': str(self.model_save_path),
            '--batch_size': self.batch_size,
            '--neg_sample_size': self.neg_samples,
            '--hidden_dim': self.embedding_dim,
            '--gamma': 12.0,  # RotatE模型的margin
            '--lr': self.lr,
            '--max_step': self.max_epochs,
            '--log_interval': 1000,
            '--eval_interval': 10000,
            '--no_eval_filter': True,
            '--gpu': self.gpu_id,
            '--regularization_coef': 1e-9,
            '--double_entity_emb': self.model_name == 'RotatE' or self.model_name == 'ComplEx',
            '--double_relation_emb': self.model_name == 'ComplEx',
            '--seed': 42,
            '--mix_cpu_gpu': self.use_mix_cpu_gpu,
            '--num_proc': min(4, os.cpu_count())
        }
        
        # 特定模型的参数
        if self.model_name == 'RotatE':
            config['--gamma'] = 9.0
            config['--negative_sample_type'] = 'both'
            config['--adversarial_temperature'] = 1.0
        elif self.model_name == 'TransE':
            config['--gamma'] = 9.0
            config['--negative_sample_type'] = 'both'
            config['--adversarial_temperature'] = 1.0
        elif self.model_name == 'DistMult':
            config['--gamma'] = 143.0
            config['--negative_sample_type'] = 'corrupt_head,corrupt_tail'
            config['--adversarial_temperature'] = 0.5
        elif self.model_name == 'ComplEx':
            config['--gamma'] = 143.0
            config['--negative_sample_type'] = 'corrupt_head,corrupt_tail'
            config['--adversarial_temperature'] = 0.5
            
        return config
    
    def save_config(self):
        """保存训练配置"""
        config = self.prepare_training_config()
        config_file = self.model_save_path / 'config.json'
        
        # 确保目录存在
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        # 将配置保存为JSON
        with open(config_file, 'w') as f:
            # 转换为可序列化的格式
            serializable_config = {}
            for k, v in config.items():
                if isinstance(v, Path):
                    serializable_config[k] = str(v)
                elif isinstance(v, list):
                    serializable_config[k] = [str(x) if isinstance(x, Path) else x for x in v]
                else:
                    serializable_config[k] = v
            
            json.dump(serializable_config, f, indent=4)
            
        print(f"配置已保存到 {config_file}")
        
    def train_dglke(self):
        """使用DGL-KE训练知识图谱嵌入模型"""
        print(f"开始训练 {self.model_name} 模型...")
        
        # 获取训练配置
        config = self.prepare_training_config()
        
        # 保存配置
        self.save_config()
        
        # 构建命令行参数
        cmd_args = []
        for k, v in config.items():
            if isinstance(v, bool):
                if v:
                    cmd_args.append(k)
            elif isinstance(v, list):
                cmd_args.append(k)
                cmd_args.append(','.join([str(x) for x in v]))
            else:
                cmd_args.append(k)
                cmd_args.append(str(v))
        
        # 记录开始时间
        start_time = time.time()
        
        # 使用DGL-KE训练
        try:
            from dglke.train import main
            main(cmd_args)
        except ImportError:
            print("无法导入dglke模块。尝试使用命令行方式运行...")
            
            # 使用命令行运行DGL-KE
            cmd = ['dglke_train'] + cmd_args
            try:
                # 记录训练日志
                with open(self.log_file, 'w') as f:
                    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    for line in process.stdout:
                        f.write(line)
                        print(line, end='')
                    process.wait()
                    if process.returncode != 0:
                        print(f"训练失败，退出代码: {process.returncode}")
                        sys.exit(1)
            except (subprocess.SubprocessError, FileNotFoundError) as e:
                print(f"运行dglke_train失败: {e}")
                print("请确保DGL-KE已正确安装。")
                sys.exit(1)
        
        # 记录结束时间
        end_time = time.time()
        training_time = end_time - start_time
        
        # 记录训练时间
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"{self.model_name}模型训练完成！")
        print(f"训练时间: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
        print(f"模型已保存到 {self.model_save_path}")
        
        # 将训练时间写入配置文件
        config_file = self.model_save_path / 'config.json'
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            config_data['训练时间_小时'] = hours
            config_data['训练时间_分钟'] = minutes
            config_data['训练时间_秒'] = seconds
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=4)
        
    def load_embeddings(self):
        """加载训练好的嵌入向量"""
        entity_file = self.model_save_path / 'DRKG_entity.npy'
        relation_file = self.model_save_path / 'DRKG_relation.npy'
        
        if not entity_file.exists() or not relation_file.exists():
            print(f"找不到嵌入文件: {entity_file} 或 {relation_file}")
            return None, None
        
        print(f"加载嵌入文件: {entity_file} 和 {relation_file}")
        entity_emb = np.load(entity_file)
        relation_emb = np.load(relation_file)
        
        return entity_emb, relation_emb
    
    def run(self):
        """运行训练过程"""
        # 训练模型
        self.train_dglke()
        
        # 加载嵌入
        entity_emb, relation_emb = self.load_embeddings()
        
        # 返回结果
        return {
            'model_id': self.model_id,
            'model_path': self.model_save_path,
            'entity_embedding': entity_emb,
            'relation_embedding': relation_emb
        }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练DRKG知识图谱嵌入模型')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='数据目录')
    parser.add_argument('--model', type=str, default='TransE',
                       choices=['TransE', 'RotatE', 'DistMult', 'ComplEx'],
                       help='知识图谱嵌入模型类型')
    parser.add_argument('--dim', type=int, default=400,
                       help='嵌入维度')
    parser.add_argument('--batch_size', type=int, default=1024,
                       help='批次大小')
    parser.add_argument('--neg_samples', type=int, default=256,
                       help='负采样数量')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--epochs', type=int, default=500,
                       help='训练轮数')
    parser.add_argument('--gpu', type=int, default=-1,
                       help='GPU ID，-1表示使用CPU')
    parser.add_argument('--mix_cpu_gpu', action='store_true',
                       help='是否使用混合CPU-GPU模式（对于大模型有用）')
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = DRKGEmbeddingTrainer(
        data_dir=args.data_dir,
        model_name=args.model,
        embedding_dim=args.dim,
        batch_size=args.batch_size,
        neg_samples=args.neg_samples,
        lr=args.lr,
        max_epochs=args.epochs,
        gpu_id=args.gpu,
        use_mix_cpu_gpu=args.mix_cpu_gpu
    )
    
    # 运行训练
    trainer.run()

if __name__ == "__main__":
    main() 