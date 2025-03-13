import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

class DRKGProcessor:
    """DRKG数据处理工具"""
    
    def __init__(self, data_dir='data'):
        """
        初始化数据处理器
        
        Args:
            data_dir (str): 数据目录路径
        """
        self.data_dir = Path(data_dir)
        self.drkg_file = self.data_dir / 'drkg.tsv'
        self.entity_map_file = self.data_dir / 'entity_idx_map.txt'
        self.relation_map_file = self.data_dir / 'relation_idx_map.txt'
        
        if not self.drkg_file.exists():
            raise FileNotFoundError(f"找不到DRKG数据文件: {self.drkg_file}")
            
    def load_drkg(self):
        """加载DRKG数据集"""
        print("加载DRKG数据集...")
        self.drkg_df = pd.read_csv(self.drkg_file, sep='\t', header=None,
                                 names=['source', 'relation', 'target'])
        print(f"加载了 {len(self.drkg_df)} 条三元组")
        return self.drkg_df
    
    def load_mappings(self):
        """加载实体和关系映射"""
        print("加载实体和关系映射...")
        self.entity_map = pd.read_csv(self.entity_map_file, sep='\t', header=None,
                                    names=['entity', 'idx'])
        self.relation_map = pd.read_csv(self.relation_map_file, sep='\t', header=None,
                                      names=['relation', 'idx'])
        print(f"加载了 {len(self.entity_map)} 个实体和 {len(self.relation_map)} 种关系")
        return self.entity_map, self.relation_map
    
    def get_entity_types(self):
        """获取实体类型统计"""
        print("分析实体类型...")
        entities = set(self.drkg_df['source'].unique()) | set(self.drkg_df['target'].unique())
        entity_types = {}
        
        for entity in entities:
            entity_type = entity.split('::')[0]
            if entity_type not in entity_types:
                entity_types[entity_type] = 0
            entity_types[entity_type] += 1
        
        return entity_types
    
    def get_relation_types(self):
        """获取关系类型统计"""
        print("分析关系类型...")
        relation_types = {}
        
        for relation in self.drkg_df['relation'].unique():
            relation_parts = relation.split('::')
            relation_type = relation_parts[-1] if len(relation_parts) > 1 else relation
            if relation_type not in relation_types:
                relation_types[relation_type] = 0
            relation_types[relation_type] += 1
        
        return relation_types
    
    def get_covid_related_triplets(self):
        """获取与COVID-19相关的三元组"""
        print("提取COVID-19相关三元组...")
        covid_terms = ['COVID-19', 'SARS-CoV-2', 'Coronavirus', '2019-nCoV', 'SARS-CoV', 'MERS-CoV']
        
        covid_df = self.drkg_df[
            self.drkg_df['source'].apply(lambda x: any(term.lower() in str(x).lower() for term in covid_terms)) |
            self.drkg_df['target'].apply(lambda x: any(term.lower() in str(x).lower() for term in covid_terms))
        ]
        
        print(f"找到 {len(covid_df)} 条COVID-19相关三元组")
        return covid_df
    
    def split_dataset(self, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, seed=42):
        """
        分割数据集为训练集、验证集和测试集
        
        Args:
            train_ratio (float): 训练集比例
            valid_ratio (float): 验证集比例
            test_ratio (float): 测试集比例
            seed (int): 随机种子
        
        Returns:
            tuple: (训练集, 验证集, 测试集)
        """
        print("分割数据集...")
        # 随机打乱数据
        df = self.drkg_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        n = len(df)
        train_end = int(n * train_ratio)
        valid_end = train_end + int(n * valid_ratio)
        
        train_df = df[:train_end]
        valid_df = df[train_end:valid_end]
        test_df = df[valid_end:]
        
        print(f"训练集: {len(train_df)}, 验证集: {len(valid_df)}, 测试集: {len(test_df)}")
        return train_df, valid_df, test_df
    
    def save_datasets(self, train_df, valid_df, test_df, output_dir='data/processed'):
        """
        保存分割后的数据集
        
        Args:
            train_df (DataFrame): 训练集
            valid_df (DataFrame): 验证集
            test_df (DataFrame): 测试集
            output_dir (str): 输出目录
        """
        print("保存数据集...")
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        train_df.to_csv(out_dir / 'train.tsv', sep='\t', header=False, index=False)
        valid_df.to_csv(out_dir / 'valid.tsv', sep='\t', header=False, index=False)
        test_df.to_csv(out_dir / 'test.tsv', sep='\t', header=False, index=False)
        
        print(f"数据集已保存到 {out_dir}")
    
    def create_graph_visualization(self, triplets=None, max_nodes=100, output_file='results/figures/kg_sample.png'):
        """
        创建知识图谱可视化
        
        Args:
            triplets (DataFrame): 三元组数据，默认为None使用全部数据
            max_nodes (int): 最大节点数
            output_file (str): 输出文件路径
        """
        print("创建知识图谱可视化...")
        if triplets is None:
            if len(self.drkg_df) > 10000:
                # 采样一部分三元组进行可视化
                triplets = self.drkg_df.sample(n=min(10000, len(self.drkg_df)), random_state=42)
            else:
                triplets = self.drkg_df
                
        # 创建图
        G = nx.DiGraph()
        
        # 添加边
        for _, row in triplets.iterrows():
            G.add_edge(row['source'], row['target'], label=row['relation'])
            
        # 限制节点数量
        if len(G.nodes) > max_nodes:
            # 选择度最高的节点
            degrees = dict(G.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            top_node_ids = [node[0] for node in top_nodes]
            G = G.subgraph(top_node_ids)
        
        # 设置颜色映射
        node_colors = []
        for node in G.nodes():
            entity_type = node.split('::')[0]
            # 根据实体类型设置颜色
            if 'Gene' in entity_type:
                node_colors.append('blue')
            elif 'Compound' in entity_type or 'Drug' in entity_type:
                node_colors.append('green')
            elif 'Disease' in entity_type:
                node_colors.append('red')
            elif 'COVID' in node or 'SARS' in node:
                node_colors.append('purple')
            else:
                node_colors.append('gray')
                
        # 可视化图
        plt.figure(figsize=(20, 20))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=False, node_color=node_colors, 
                node_size=100, edge_color='gray', arrows=True, alpha=0.7)
        
        # 为重要节点添加标签
        labels = {}
        for node in G.nodes():
            if 'COVID' in node or 'SARS' in node or G.degree(node) > 5:
                labels[node] = node.split('::')[-1]
        
        nx.draw_networkx_labels(G, pos, labels, font_size=10)
        
        # 保存图
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"图已保存到 {output_file}")
        return G
    
    def analyze_and_plot_statistics(self, output_dir='results/figures'):
        """
        分析并绘制DRKG统计信息
        
        Args:
            output_dir (str): 输出目录
        """
        print("分析DRKG统计信息...")
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # 实体类型分布
        entity_types = self.get_entity_types()
        plt.figure(figsize=(14, 7))
        sns.barplot(x=list(entity_types.keys()), y=list(entity_types.values()))
        plt.title('DRKG实体类型分布')
        plt.xlabel('实体类型')
        plt.ylabel('数量')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(out_dir / 'entity_type_distribution.png')
        plt.close()
        
        # 关系类型分布 (只显示top 15)
        relation_types = self.get_relation_types()
        top_relations = sorted(relation_types.items(), key=lambda x: x[1], reverse=True)[:15]
        plt.figure(figsize=(14, 7))
        sns.barplot(x=[r[0] for r in top_relations], y=[r[1] for r in top_relations])
        plt.title('DRKG前15种关系类型分布')
        plt.xlabel('关系类型')
        plt.ylabel('数量')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(out_dir / 'relation_type_distribution.png')
        plt.close()
        
        # 节点度分布
        source_counts = self.drkg_df['source'].value_counts()
        target_counts = self.drkg_df['target'].value_counts()
        all_entities = pd.concat([source_counts, target_counts]).groupby(level=0).sum()
        
        plt.figure(figsize=(14, 7))
        plt.hist(np.log10(all_entities.values), bins=50)
        plt.title('DRKG节点度分布（对数尺度）')
        plt.xlabel('节点度（log10）')
        plt.ylabel('节点数量')
        plt.tight_layout()
        plt.savefig(out_dir / 'node_degree_distribution.png')
        plt.close()
        
        print(f"统计图表已保存到 {out_dir}")
        
if __name__ == "__main__":
    processor = DRKGProcessor()
    processor.load_drkg()
    processor.load_mappings()
    
    # 获取并分析COVID-19相关三元组
    covid_df = processor.get_covid_related_triplets()
    
    # 分割数据集并保存
    train_df, valid_df, test_df = processor.split_dataset()
    processor.save_datasets(train_df, valid_df, test_df)
    
    # 创建知识图谱可视化
    processor.create_graph_visualization(covid_df, max_nodes=150)
    
    # 分析并绘制统计信息
    processor.analyze_and_plot_statistics() 