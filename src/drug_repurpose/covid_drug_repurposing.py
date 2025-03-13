#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime
import time

sys.path.append(str(Path(__file__).parent.parent))
from utils.data_utils import DRKGProcessor

class DrugRepurposingModel:
    """COVID-19药物重新定位模型"""
    
    def __init__(self, model_dir, entity_map_file='data/entity_idx_map.txt'):
        """
        初始化药物重新定位模型
        
        Args:
            model_dir (str): 模型目录路径
            entity_map_file (str): 实体映射文件路径
        """
        self.model_dir = Path(model_dir)
        self.entity_map_file = Path(entity_map_file)
        
        # 加载实体映射
        self.load_entity_mapping()
        
        # 加载嵌入
        self.load_embeddings()
        
        # 检测可用设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 将嵌入加载到设备上
        self.entity_emb_tensor = self.entity_emb_tensor.to(self.device)
        self.relation_emb_tensor = self.relation_emb_tensor.to(self.device)
    
    def load_entity_mapping(self):
        """加载实体映射"""
        print(f"加载实体映射: {self.entity_map_file}")
        
        # 检查文件是否存在
        if not self.entity_map_file.exists():
            raise FileNotFoundError(f"找不到实体映射文件: {self.entity_map_file}")
        
        # 加载映射
        self.entity_map = {}
        self.idx_to_entity = {}
        
        with open(self.entity_map_file, 'r') as f:
            for line in f:
                entity, idx = line.strip().split('\t')
                self.entity_map[entity] = int(idx)
                self.idx_to_entity[int(idx)] = entity
                
        print(f"加载了 {len(self.entity_map)} 个实体映射")
        
    def load_embeddings(self):
        """加载模型嵌入"""
        entity_file = self.model_dir / 'DRKG_entity.npy'
        relation_file = self.model_dir / 'DRKG_relation.npy'
        
        # 检查文件是否存在
        if not entity_file.exists() or not relation_file.exists():
            raise FileNotFoundError(f"找不到嵌入文件: {entity_file} 或 {relation_file}")
        
        print(f"加载嵌入文件: {entity_file} 和 {relation_file}")
        
        # 加载嵌入
        self.entity_emb = np.load(entity_file)
        self.relation_emb = np.load(relation_file)
        
        print(f"实体嵌入形状: {self.entity_emb.shape}, 关系嵌入形状: {self.relation_emb.shape}")
        
        # 提取模型信息
        self.embedding_dim = self.entity_emb.shape[1]
        print(f"嵌入维度: {self.embedding_dim}")
        
        # 提取模型类型
        config_file = self.model_dir / 'config.json'
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
                self.model_type = config.get('--model_name', 'Unknown')
        else:
            # 从目录名称猜测模型类型
            dirname = os.path.basename(self.model_dir)
            if 'TransE' in dirname:
                self.model_type = 'TransE'
            elif 'RotatE' in dirname:
                self.model_type = 'RotatE'
            elif 'DistMult' in dirname:
                self.model_type = 'DistMult'
            elif 'ComplEx' in dirname:
                self.model_type = 'ComplEx'
            else:
                self.model_type = 'Unknown'
        
        print(f"模型类型: {self.model_type}")
        
        # 转换为张量
        self.entity_emb_tensor = torch.tensor(self.entity_emb)
        self.relation_emb_tensor = torch.tensor(self.relation_emb)
    
    def find_entity_by_name(self, name, partial_match=True):
        """
        通过名称查找实体
        
        Args:
            name (str): 实体名称
            partial_match (bool): 是否允许部分匹配
            
        Returns:
            list: 匹配的实体列表
        """
        matches = []
        
        if partial_match:
            for entity in self.entity_map.keys():
                if name.lower() in entity.lower():
                    matches.append(entity)
        else:
            if name in self.entity_map:
                matches.append(name)
                
        return matches
    
    def get_entity_embedding(self, entity):
        """
        获取实体的嵌入向量
        
        Args:
            entity (str): 实体名称
            
        Returns:
            ndarray: 嵌入向量
        """
        if entity not in self.entity_map:
            raise ValueError(f"找不到实体: {entity}")
            
        idx = self.entity_map[entity]
        return self.entity_emb[idx]
    
    def get_relation_embedding(self, relation):
        """
        获取关系的嵌入向量
        
        Args:
            relation (str): 关系名称
            
        Returns:
            ndarray: 嵌入向量
        """
        # 这里需要一个关系到索引的映射，暂时使用实体映射
        if relation not in self.entity_map:
            raise ValueError(f"找不到关系: {relation}")
            
        idx = self.entity_map[relation]
        return self.relation_emb[idx]
    
    def predict_triples(self, head, relation, candidates=None, top_k=100):
        """
        预测三元组（头实体，关系，?）
        
        Args:
            head (str): 头实体
            relation (str): 关系
            candidates (list): 候选尾实体列表，None表示使用所有实体
            top_k (int): 返回前k个预测结果
            
        Returns:
            list: 预测结果列表，每个元素为(尾实体，分数)
        """
        if head not in self.entity_map:
            raise ValueError(f"找不到头实体: {head}")
            
        # 获取头实体和关系的嵌入
        head_idx = self.entity_map[head]
        head_emb = self.entity_emb_tensor[head_idx].to(self.device)
        
        # 如果关系不在映射中，尝试查找或报错
        if relation not in self.entity_map:
            # 尝试查找类似的关系
            related_relations = [r for r in self.entity_map.keys() if relation in r]
            if related_relations:
                print(f"找不到关系: {relation}，使用相关关系: {related_relations[0]}")
                relation = related_relations[0]
            else:
                raise ValueError(f"找不到关系: {relation}")
        
        relation_idx = self.entity_map[relation]
        relation_emb = self.relation_emb_tensor[relation_idx].to(self.device)
        
        # 根据不同模型类型预测尾实体
        if self.model_type == 'TransE':
            # TransE: 头 + 关系 ≈ 尾
            tail_emb_pred = head_emb + relation_emb
            
            # 计算余弦相似度
            similarities = torch.nn.functional.cosine_similarity(
                tail_emb_pred.unsqueeze(0), 
                self.entity_emb_tensor
            )
        elif self.model_type == 'DistMult':
            # DistMult: 头 * 关系 * 尾 (这里我们计算 头 * 关系 与每个尾的点积)
            head_rel = head_emb * relation_emb
            similarities = torch.matmul(self.entity_emb_tensor, head_rel)
        elif self.model_type == 'RotatE':
            # RotatE比较复杂，需要在复数空间操作，使用近似计算
            # 对于初步筛查，可以简化为TransE类似的方法
            tail_emb_pred = head_emb + relation_emb
            similarities = torch.nn.functional.cosine_similarity(
                tail_emb_pred.unsqueeze(0), 
                self.entity_emb_tensor
            )
        elif self.model_type == 'ComplEx':
            # ComplEx也在复数空间操作，使用近似计算
            # 将实部和虚部分开处理
            dim = self.embedding_dim // 2
            head_re, head_im = head_emb[:dim], head_emb[dim:]
            rel_re, rel_im = relation_emb[:dim], relation_emb[dim:]
            
            # 计算复数乘积: (head_re + i*head_im) * (rel_re + i*rel_im)
            # = (head_re * rel_re - head_im * rel_im) + i*(head_re * rel_im + head_im * rel_re)
            prod_re = head_re * rel_re - head_im * rel_im
            prod_im = head_re * rel_im + head_im * rel_re
            
            # 计算每个实体的得分
            entity_re = self.entity_emb_tensor[:, :dim]
            entity_im = self.entity_emb_tensor[:, dim:]
            
            # 复数内积: (a+bi)(c-di) = ac + bd + i(bc - ad)，取实部 ac + bd
            similarities = torch.sum(entity_re * prod_re, dim=1) + torch.sum(entity_im * prod_im, dim=1)
        else:
            # 默认使用TransE
            tail_emb_pred = head_emb + relation_emb
            similarities = torch.nn.functional.cosine_similarity(
                tail_emb_pred.unsqueeze(0), 
                self.entity_emb_tensor
            )
        
        # 获取相似度最高的实体
        if candidates:
            # 只考虑候选实体
            candidate_indices = torch.tensor([self.entity_map[c] for c in candidates if c in self.entity_map],
                                           device=self.device)
            if len(candidate_indices) == 0:
                return []
                
            candidate_similarities = similarities[candidate_indices]
            top_indices = torch.argsort(candidate_similarities, descending=True)[:top_k]
            top_entities = [candidates[i] for i in top_indices.cpu().numpy()]
            top_scores = candidate_similarities[top_indices].cpu().tolist()
        else:
            # 考虑所有实体
            top_indices = torch.argsort(similarities, descending=True)[:top_k]
            top_entities = [self.idx_to_entity[i.item()] for i in top_indices]
            top_scores = similarities[top_indices].cpu().tolist()
        
        return list(zip(top_entities, top_scores))
    
    def predict_batch_triples(self, heads, relation, candidates=None, top_k=100, batch_size=32):
        """
        批量预测三元组（头实体，关系，?）
        
        Args:
            heads (list): 头实体列表
            relation (str): 关系
            candidates (list): 候选尾实体列表，None表示使用所有实体
            top_k (int): 返回前k个预测结果
            batch_size (int): 批处理大小
            
        Returns:
            dict: 头实体到预测结果的映射
        """
        results = {}
        
        # 如果头实体少，不需要批处理
        if len(heads) <= batch_size:
            for head in tqdm(heads, desc="预测三元组"):
                try:
                    results[head] = self.predict_triples(head, relation, candidates, top_k)
                except Exception as e:
                    print(f"预测 {head} 时出错: {e}")
            return results
        
        # 批处理预测
        # 获取关系的嵌入
        if relation not in self.entity_map:
            # 尝试查找类似的关系
            related_relations = [r for r in self.entity_map.keys() if relation in r]
            if related_relations:
                print(f"找不到关系: {relation}，使用相关关系: {related_relations[0]}")
                relation = related_relations[0]
            else:
                raise ValueError(f"找不到关系: {relation}")
                
        relation_idx = self.entity_map[relation]
        relation_emb = self.relation_emb_tensor[relation_idx].to(self.device)
        
        # 获取候选实体索引
        if candidates:
            candidate_indices = torch.tensor([self.entity_map[c] for c in candidates if c in self.entity_map],
                                           device=self.device)
            candidate_entities = [c for c in candidates if c in self.entity_map]
            if len(candidate_indices) == 0:
                return {}
        
        # 批量处理
        for i in tqdm(range(0, len(heads), batch_size), desc="批处理预测"):
            batch_heads = heads[i:i+batch_size]
            
            # 获取有效的头实体
            valid_heads = []
            valid_indices = []
            
            for head in batch_heads:
                if head in self.entity_map:
                    valid_heads.append(head)
                    valid_indices.append(self.entity_map[head])
            
            if not valid_heads:
                continue
                
            # 转换为张量
            head_indices = torch.tensor(valid_indices, device=self.device)
            head_embs = self.entity_emb_tensor[head_indices]
            
            # 根据不同模型类型批量预测
            if self.model_type == 'TransE':
                # TransE: 头 + 关系 ≈ 尾
                tail_emb_pred = head_embs + relation_emb.unsqueeze(0)
                
                # 批量计算相似度
                if candidates:
                    # 只考虑候选实体
                    candidate_embs = self.entity_emb_tensor[candidate_indices]
                    similarities = torch.matmul(tail_emb_pred, candidate_embs.t())
                else:
                    # 考虑所有实体
                    similarities = torch.matmul(tail_emb_pred, self.entity_emb_tensor.t())
            elif self.model_type == 'DistMult':
                # DistMult: 头 * 关系 * 尾
                head_rel = head_embs * relation_emb.unsqueeze(0)
                
                if candidates:
                    candidate_embs = self.entity_emb_tensor[candidate_indices]
                    similarities = torch.matmul(head_rel, candidate_embs.t())
                else:
                    similarities = torch.matmul(head_rel, self.entity_emb_tensor.t())
            else:
                # 其他模型简化为TransE
                tail_emb_pred = head_embs + relation_emb.unsqueeze(0)
                
                if candidates:
                    candidate_embs = self.entity_emb_tensor[candidate_indices]
                    similarities = torch.matmul(tail_emb_pred, candidate_embs.t())
                else:
                    similarities = torch.matmul(tail_emb_pred, self.entity_emb_tensor.t())
            
            # 获取每个头实体的预测结果
            for j, head in enumerate(valid_heads):
                # 获取top_k
                head_similarities = similarities[j]
                top_indices = torch.argsort(head_similarities, descending=True)[:top_k]
                
                # 获取实体和分数
                if candidates:
                    top_entities = [candidate_entities[i.item()] for i in top_indices]
                else:
                    top_entities = [self.idx_to_entity[candidate_indices[i.item()].item() 
                                                   if candidates else i.item()] 
                                 for i in top_indices]
                
                top_scores = head_similarities[top_indices].cpu().tolist()
                
                # 保存结果
                results[head] = list(zip(top_entities, top_scores))
        
        return results
    
    def get_drug_candidates(self, disease='Disease::COVID-19', 
                          relation='Hetionet::CtD::TREATS', top_k=100):
        """
        获取疾病的候选药物
        
        Args:
            disease (str): 疾病实体名称
            relation (str): 关系名称，默认为'Hetionet::CtD::TREATS'
            top_k (int): 返回前k个候选药物
            
        Returns:
            DataFrame: 候选药物及其分数
        """
        start_time = time.time()
        
        # 查找COVID-19相关实体
        covid_entities = self.find_entity_by_name(disease, partial_match=False)
        
        if not covid_entities:
            covid_entities = self.find_entity_by_name('COVID-19')
            
        if not covid_entities:
            raise ValueError("找不到COVID-19相关实体")
        
        # 获取所有药物实体
        drug_entities = []
        for entity in self.entity_map.keys():
            if entity.startswith('Compound::') or entity.startswith('Drug::'):
                drug_entities.append(entity)
                
        print(f"找到 {len(drug_entities)} 个药物实体")
        
        # 对每个COVID-19实体进行预测
        batch_results = self.predict_batch_triples(
            covid_entities, relation, 
            candidates=drug_entities, 
            top_k=top_k
        )
        
        # 整合结果
        all_results = []
        for entity, results in batch_results.items():
            for drug, score in results:
                all_results.append((drug, score, entity))
                
        # 转换为DataFrame
        if all_results:
            df = pd.DataFrame(all_results, columns=['drug', 'score', 'disease'])
            # 按分数降序排序
            df = df.sort_values('score', ascending=False).reset_index(drop=True)
            # 只保留前top_k个结果
            df = df.head(top_k)
        else:
            df = pd.DataFrame(columns=['drug', 'score', 'disease'])
        
        end_time = time.time()
        print(f"预测耗时: {end_time - start_time:.2f} 秒")
        
        return df
    
    def plot_top_drugs(self, drug_df, title='Top Predicted Drugs for COVID-19', 
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
        
        # 为每种药物分配不同颜色
        drug_types = plot_df['drug'].apply(lambda x: x.split('::')[0] if '::' in x else '其他')
        unique_types = drug_types.unique()
        
        # 创建调色板
        palette = sns.color_palette("husl", len(unique_types))
        color_map = dict(zip(unique_types, palette))
        colors = [color_map[t] for t in drug_types]
        
        # 创建图表
        plt.figure(figsize=(14, 10))
        
        # 创建条形图
        bars = plt.barh(plot_df['drug_name'], plot_df['score'], color=colors)
        
        # 添加条形图标签
        for bar, score in zip(bars, plot_df['score']):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', va='center')
        
        # 添加图例
        legend_handles = [plt.Rectangle((0,0),1,1, color=color_map[t]) for t in unique_types]
        plt.legend(legend_handles, unique_types, loc='lower right')
        
        # 设置标题和轴标签
        plt.title(f"{title}\n使用{self.model_type}模型 ({self.embedding_dim}维)", fontsize=16)
        plt.xlabel('预测分数', fontsize=14)
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
        
        # 创建网络视图（如果药物数量适中）
        if len(plot_df) <= 30:
            try:
                self.plot_network_view(plot_df, save_path.replace('.png', '_network.png') if save_path else None)
            except Exception as e:
                print(f"创建网络视图失败: {e}")
    
    def plot_network_view(self, drug_df, save_path=None):
        """
        创建网络视图，显示疾病和药物之间的关系
        
        Args:
            drug_df (DataFrame): 药物预测结果DataFrame
            save_path (str): 保存路径，None表示不保存
        """
        # 需要安装networkx
        try:
            import networkx as nx
            
            # 创建有向图
            G = nx.DiGraph()
            
            # 添加节点
            diseases = drug_df['disease'].unique()
            drugs = drug_df['drug'].unique()
            
            # 添加疾病节点
            for disease in diseases:
                G.add_node(disease, type='disease')
                
            # 添加药物节点
            for drug in drugs:
                G.add_node(drug, type='drug')
                
            # 添加边
            for _, row in drug_df.iterrows():
                G.add_edge(row['disease'], row['drug'], weight=row['score'])
            
            # 创建图表
            plt.figure(figsize=(18, 12))
            
            # 设置节点位置
            pos = nx.spring_layout(G, seed=42, k=0.5)
            
            # 绘制节点
            disease_nodes = [node for node, attr in G.nodes(data=True) if attr.get('type') == 'disease']
            drug_nodes = [node for node, attr in G.nodes(data=True) if attr.get('type') == 'drug']
            
            # 绘制疾病节点
            nx.draw_networkx_nodes(G, pos, nodelist=disease_nodes, node_color='red', node_size=800, alpha=0.8)
            
            # 绘制药物节点
            # 根据得分为药物节点着色
            drug_scores = {drug: drug_df[drug_df['drug'] == drug]['score'].max() for drug in drug_nodes}
            drug_colors = [drug_scores[drug] for drug in drug_nodes]
            
            nodes = nx.draw_networkx_nodes(G, pos, nodelist=drug_nodes, node_color=drug_colors, 
                                          cmap=plt.cm.Blues, node_size=400, alpha=0.8)
            
            # 添加颜色条
            plt.colorbar(nodes, label='预测分数')
            
            # 绘制边
            edges = G.edges(data=True)
            edge_colors = [d['weight'] for _, _, d in edges]
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, 
                                  edge_cmap=plt.cm.Reds, width=2, alpha=0.6)
            
            # 绘制标签
            # 获取简化标签
            labels = {}
            for node in G.nodes():
                if '::' in node:
                    labels[node] = node.split('::')[-1]
                else:
                    labels[node] = node
            
            # 分开绘制疾病和药物标签
            disease_labels = {node: labels[node] for node in disease_nodes}
            drug_labels = {node: labels[node] for node in drug_nodes}
            
            nx.draw_networkx_labels(G, pos, labels=disease_labels, font_size=12, font_weight='bold')
            nx.draw_networkx_labels(G, pos, labels=drug_labels, font_size=10)
            
            # 设置标题
            plt.title(f"{self.model_type}模型预测的COVID-19候选药物网络视图", fontsize=16)
            plt.axis('off')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"网络视图已保存到 {save_path}")
                
            plt.close()
        except ImportError:
            print("未安装networkx，无法创建网络视图")
    
    def save_predictions(self, drug_df, output_file='results/predictions/covid_drug_predictions.csv'):
        """
        保存预测结果
        
        Args:
            drug_df (DataFrame): 药物预测结果DataFrame
            output_file (str): 输出文件路径
        """
        # 创建输出目录
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存结果
        drug_df.to_csv(output_path, index=False)
        print(f"预测结果已保存到 {output_path}")
        
        # 保存一份excel格式
        try:
            excel_path = output_path.with_suffix('.xlsx')
            drug_df.to_excel(excel_path, index=False)
            print(f"预测结果已保存到Excel: {excel_path}")
        except:
            print("无法保存为Excel格式，需要安装openpyxl")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='COVID-19药物重新定位')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='模型目录')
    parser.add_argument('--entity_map', type=str, default='data/entity_idx_map.txt',
                       help='实体映射文件')
    parser.add_argument('--disease', type=str, default='Disease::COVID-19',
                       help='疾病实体名称')
    parser.add_argument('--relation', type=str, default='Hetionet::CtD::TREATS',
                       help='关系名称')
    parser.add_argument('--top_k', type=int, default=100,
                       help='返回前k个候选药物')
    parser.add_argument('--output', type=str, default='results/predictions/covid_drug_predictions.csv',
                       help='输出文件路径')
    parser.add_argument('--plot', action='store_true',
                       help='是否绘制图表')
    
    args = parser.parse_args()
    
    # 创建药物重新定位模型
    model = DrugRepurposingModel(
        model_dir=args.model_dir,
        entity_map_file=args.entity_map
    )
    
    # 获取COVID-19的候选药物
    drug_df = model.get_drug_candidates(
        disease=args.disease,
        relation=args.relation,
        top_k=args.top_k
    )
    
    # 保存预测结果
    model.save_predictions(drug_df, args.output)
    
    # 绘制图表
    if args.plot:
        save_path = str(Path(args.output).with_suffix('.png'))
        model.plot_top_drugs(drug_df, save_path=save_path, top_n=min(args.top_k, 30))
        
    print(f"找到 {len(drug_df)} 个候选药物")
    print(f"前10个候选药物:")
    for i, (drug, score) in enumerate(zip(drug_df['drug'].head(10), drug_df['score'].head(10))):
        print(f"{i+1}. {drug}: {score:.4f}")

if __name__ == "__main__":
    main() 