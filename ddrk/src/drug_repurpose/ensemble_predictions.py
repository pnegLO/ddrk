#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_prediction_files(file_paths):
    """
    加载多个预测文件
    
    Args:
        file_paths (list): 预测文件路径列表
        
    Returns:
        list: 预测数据框列表
    """
    predictions = []
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            # 确保有药物和分数列
            if 'drug' not in df.columns or 'score' not in df.columns:
                print(f"警告: {file_path} 缺少必要的列 (drug, score)")
                continue
            predictions.append(df)
            print(f"成功加载 {file_path}, 包含 {len(df)} 个预测结果")
        except Exception as e:
            print(f"无法加载 {file_path}: {e}")
    
    return predictions

def normalize_scores(df):
    """
    归一化分数到 [0, 1] 范围
    
    Args:
        df (DataFrame): 包含预测分数的数据框
        
    Returns:
        DataFrame: 归一化后的数据框
    """
    min_score = df['score'].min()
    max_score = df['score'].max()
    
    # 避免除以零
    if max_score == min_score:
        df['norm_score'] = 1.0
    else:
        df['norm_score'] = (df['score'] - min_score) / (max_score - min_score)
    
    return df

def ensemble_predictions(prediction_dfs, method='mean', weights=None):
    """
    集成多个模型的预测结果
    
    Args:
        prediction_dfs (list): 预测数据框列表
        method (str): 集成方法，'mean'、'max' 或 'weighted'
        weights (list): 权重列表，用于加权平均
        
    Returns:
        DataFrame: 集成后的预测结果
    """
    # 如果只有一个预测，直接返回
    if len(prediction_dfs) == 1:
        return prediction_dfs[0]
    
    # 归一化所有分数
    normalized_dfs = [normalize_scores(df.copy()) for df in prediction_dfs]
    
    # 获取所有唯一的药物
    all_drugs = set()
    for df in normalized_dfs:
        all_drugs.update(df['drug'].tolist())
    
    print(f"总共找到 {len(all_drugs)} 个唯一药物")
    
    # 创建集成结果数据框
    ensemble_results = []
    
    for drug in all_drugs:
        # 收集每个模型对该药物的分数
        drug_scores = []
        for df in normalized_dfs:
            if drug in df['drug'].values:
                score = df.loc[df['drug'] == drug, 'norm_score'].iloc[0]
                drug_scores.append(score)
            else:
                drug_scores.append(0.0)  # 如果模型没有预测该药物，分数为0
        
        # 根据方法计算集成分数
        if method == 'mean':
            ensemble_score = np.mean(drug_scores)
        elif method == 'max':
            ensemble_score = np.max(drug_scores)
        elif method == 'weighted':
            if weights is None:
                weights = [1.0] * len(drug_scores)
            ensemble_score = np.average(drug_scores, weights=weights)
        else:
            raise ValueError(f"不支持的集成方法: {method}")
        
        ensemble_results.append({
            'drug': drug,
            'score': ensemble_score,
            'num_models': sum(1 for s in drug_scores if s > 0)
        })
    
    # 创建数据框并排序
    ensemble_df = pd.DataFrame(ensemble_results)
    ensemble_df = ensemble_df.sort_values('score', ascending=False).reset_index(drop=True)
    
    return ensemble_df

def plot_top_drugs(drug_df, title='Top Ensemble Predicted Drugs for COVID-19', 
                  save_path=None, top_n=20):
    """
    绘制前N个药物的条形图
    
    Args:
        drug_df (DataFrame): 药物预测结果DataFrame
        title (str): 图表标题
        save_path (str): 保存路径，None表示不保存
        top_n (int): 显示前N个药物
    """
    # 只显示前top_n个药物
    plot_df = drug_df.head(top_n).copy()
    
    # 提取药物名称（去掉前缀）
    plot_df['drug_name'] = plot_df['drug'].apply(lambda x: x.split('::')[-1])
    
    plt.figure(figsize=(14, 10))
    
    # 创建条形图
    ax = sns.barplot(x='score', y='drug_name', data=plot_df)
    
    # 添加模型数量信息
    if 'num_models' in plot_df.columns:
        for i, (_, row) in enumerate(plot_df.iterrows()):
            ax.text(row['score'] + 0.01, i, f"({row['num_models']}个模型)", 
                    va='center', fontsize=10)
    
    plt.title(title, fontsize=16)
    plt.xlabel('集成预测分数', fontsize=14)
    plt.ylabel('药物名称', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        # 创建目录（如果不存在）
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到 {save_path}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='集成多个模型的COVID-19药物预测结果')
    parser.add_argument('--pred_files', type=str, required=True,
                       help='预测文件路径，逗号分隔')
    parser.add_argument('--output', type=str, default='results/predictions/ensemble_predictions.csv',
                       help='输出文件路径')
    parser.add_argument('--method', type=str, default='mean',
                       choices=['mean', 'max', 'weighted'],
                       help='集成方法: mean, max, weighted')
    parser.add_argument('--weights', type=str, default=None,
                       help='模型权重，逗号分隔（用于加权平均）')
    parser.add_argument('--top_k', type=int, default=200,
                       help='输出前k个候选药物')
    parser.add_argument('--plot', action='store_true',
                       help='是否绘制图表')
    
    args = parser.parse_args()
    
    # 解析文件路径
    pred_files = [path.strip() for path in args.pred_files.split(',')]
    
    # 加载预测文件
    prediction_dfs = load_prediction_files(pred_files)
    
    if not prediction_dfs:
        print("错误: 没有成功加载任何预测文件")
        sys.exit(1)
    
    # 解析权重（如果提供）
    weights = None
    if args.weights:
        try:
            weights = [float(w.strip()) for w in args.weights.split(',')]
            if len(weights) != len(prediction_dfs):
                print(f"警告: 权重数量 ({len(weights)}) 与模型数量 ({len(prediction_dfs)}) 不匹配")
                weights = None
        except ValueError:
            print(f"警告: 无法解析权重 '{args.weights}'，使用均等权重")
    
    # 集成预测
    ensemble_df = ensemble_predictions(prediction_dfs, method=args.method, weights=weights)
    
    # 只保留前top_k个结果
    ensemble_df = ensemble_df.head(args.top_k)
    
    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ensemble_df.to_csv(output_path, index=False)
    print(f"集成预测结果已保存到 {output_path}")
    
    # 打印前10个结果
    print("\n前10个集成预测结果:")
    for i, (_, row) in enumerate(ensemble_df.head(10).iterrows()):
        print(f"{i+1}. {row['drug']}: {row['score']:.4f} (由 {row['num_models']} 个模型预测)")
    
    # 绘制图表
    if args.plot:
        plot_path = output_path.with_suffix('.png')
        plot_top_drugs(
            ensemble_df, 
            title=f'基于{len(prediction_dfs)}个模型的集成预测COVID-19候选药物',
            save_path=str(plot_path)
        )

if __name__ == "__main__":
    main() 