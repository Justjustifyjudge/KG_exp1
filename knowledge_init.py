"""
台风知识图谱初始化工具
从CSV文件批量导入台风数据到知识图谱
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Tuple
import re

# 导入知识图谱核心类（假设在同目录下）
from knowledge_edit_sql import KnowledgeGraphCore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TyphoonKGInitializer:
    """台风知识图谱初始化器"""
    
    def __init__(self, kg: KnowledgeGraphCore):
        self.kg = kg
        self.typhoon_cache = {}  # 缓存已创建的台风实体
        self.region_cache = {}   # 缓存已创建的地区实体
    
    def load_from_csv(self, csv_path: str, sample_size: int = None) -> Dict:
        """
        从CSV文件加载台风数据
        
        Args:
            csv_path: CSV文件路径
            sample_size: 采样大小（None表示全部加载）
        
        Returns:
            统计信息字典
        """
        logger.info(f"📥 开始从CSV加载台风数据: {csv_path}")
        
        # 读取CSV
        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
        except:
            df = pd.read_csv(csv_path, encoding='gbk')
        
        logger.info(f"  总记录数: {len(df)}")
        
        # 采样（如果需要）
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            logger.info(f"  采样后记录数: {len(df)}")
        
        # 数据清洗
        df = self._clean_data(df)
        
        # 按台风编号分组
        typhoon_groups = df.groupby('台风编号')
        logger.info(f"  唯一台风数: {len(typhoon_groups)}")
        
        # 统计信息
        stats = {
            "total_records": len(df),
            "unique_typhoons": len(typhoon_groups),
            "entities_added": 0,
            "relations_added": 0,
            "time_start": datetime.now()
        }
        
        # 处理每个台风
        for typhoon_id, group in typhoon_groups:
            try:
                self._process_typhoon(typhoon_id, group, stats)
            except Exception as e:
                logger.error(f"  ✗ 处理台风 {typhoon_id} 失败: {e}")
        
        stats["time_end"] = datetime.now()
        stats["duration"] = (stats["time_end"] - stats["time_start"]).total_seconds()
        
        logger.info(f"\n✅ 数据加载完成!")
        logger.info(f"  实体总数: {stats['entities_added']}")
        logger.info(f"  关系总数: {stats['relations_added']}")
        logger.info(f"  耗时: {stats['duration']:.2f} 秒")
        
        return stats
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清洗"""
        logger.info("  🧹 数据清洗中...")
        
        # 替换'-'为NaN
        df = df.replace('-', np.nan)
        
        # 转换数值列
        numeric_cols = ['经度', '纬度', '台风等级', '风速', '气压', '移动速度']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 去除完全空白的行
        df = df.dropna(how='all')
        
        return df
    
    def _process_typhoon(self, typhoon_id: str, group: pd.DataFrame, stats: Dict):
        """
        处理单个台风的所有数据
        
        Args:
            typhoon_id: 台风编号（如194501）
            group: 该台风的所有记录
            stats: 统计信息字典
        """
        # 获取台风基本信息（从第一条记录）
        first_record = group.iloc[0]
        
        # 提取年份
        year = int(typhoon_id[:4])
        
        # 获取中文名称
        cn_name = first_record.get('台风中文名称')
        if pd.isna(cn_name) or cn_name == '-':
            cn_name = None
        
        # 获取英文名称
        en_name = first_record.get('台风英文名称')
        if pd.isna(en_name) or en_name == '-':
            en_name = None
        
        # 构建台风名称
        if cn_name:
            typhoon_name = f"台风{cn_name}"
        elif en_name:
            typhoon_name = f"台风{en_name}"
        else:
            typhoon_name = f"台风{typhoon_id}"
        
        # 计算统计信息
        max_wind_speed = group['风速'].max()
        min_pressure = group['气压'].min()
        max_intensity = self._get_max_intensity(group)
        
        # 时间信息
        start_time = first_record.get('台风起始时间')
        end_time = first_record.get('台风结束时间')
        
        # 构建属性字典
        properties = {
            "year": year,
            "typhoon_id": typhoon_id,
            "start_time": str(start_time) if pd.notna(start_time) else None,
            "end_time": str(end_time) if pd.notna(end_time) else None,
            "record_count": len(group)
        }
        
        if cn_name:
            properties["chinese_name"] = cn_name
        if en_name:
            properties["english_name"] = en_name
        if pd.notna(max_wind_speed):
            properties["max_wind_speed"] = int(max_wind_speed)
        if pd.notna(min_pressure):
            properties["min_pressure"] = int(min_pressure)
        if max_intensity:
            properties["max_intensity"] = max_intensity
        
        # 创建台风实体
        entity_id = self.kg.add_entity_smart(
            typhoon_name,
            "台风",
            properties
        )
        
        if entity_id:
            stats["entities_added"] += 1
            self.typhoon_cache[typhoon_id] = entity_id
            
            # 处理轨迹和地理关系
            self._process_trajectory(typhoon_id, entity_id, group, stats)
    
    def _get_max_intensity(self, group: pd.DataFrame) -> str:
        """获取最大强度"""
        intensity_order = [
            "超强台风",
            "强台风", 
            "台风(TY)",
            "强热带风暴(STS)",
            "热带风暴(TS)",
            "热带低压(TD)"
        ]
        
        intensities = group['台风强度'].dropna().unique()
        
        for intensity in intensity_order:
            if intensity in intensities:
                return intensity
        
        return None
    
    def _process_trajectory(self, typhoon_id: str, entity_id: str, 
                           group: pd.DataFrame, stats: Dict):
        """
        处理台风轨迹，提取地理位置关系
        
        Args:
            typhoon_id: 台风编号
            entity_id: 台风实体ID
            group: 台风数据
            stats: 统计信息
        """
        # 提取轨迹点（每隔n个点采样，避免过多）
        sample_interval = max(1, len(group) // 20)  # 最多20个轨迹点
        trajectory = group.iloc[::sample_interval]
        
        # 判断登陆地区（根据经纬度）
        landfall_regions = self._detect_landfall_regions(trajectory)
        
        for region in landfall_regions:
            # 创建地区实体
            region_id = self._get_or_create_region(region, stats)
            
            if region_id:
                # 创建"登陆于"关系
                success = self.kg.add_relation(entity_id, "登陆于", region_id)
                if success:
                    stats["relations_added"] += 1
    
    def _detect_landfall_regions(self, trajectory: pd.DataFrame) -> List[str]:
        """
        根据轨迹坐标判断登陆地区
        
        简化版本：根据经纬度范围判断
        """
        regions = set()
        
        # 中国沿海省份经纬度范围（简化版）
        region_bounds = {
            "海南": {"lon": (108, 111), "lat": (18, 20)},
            "广东": {"lon": (109, 117), "lat": (20, 25)},
            "广西": {"lon": (104, 112), "lat": (20, 26)},
            "福建": {"lon": (115, 120), "lat": (23, 28)},
            "浙江": {"lon": (118, 123), "lat": (27, 31)},
            "江苏": {"lon": (116, 122), "lat": (30, 35)},
            "上海": {"lon": (120, 122), "lat": (30, 32)},
            "山东": {"lon": (114, 123), "lat": (34, 38)},
            "台湾": {"lon": (119, 122), "lat": (21, 26)},
        }
        
        for _, point in trajectory.iterrows():
            lon = point.get('经度')
            lat = point.get('纬度')
            
            if pd.isna(lon) or pd.isna(lat):
                continue
            
            # 检查是否在某个地区范围内
            for region, bounds in region_bounds.items():
                if (bounds["lon"][0] <= lon <= bounds["lon"][1] and 
                    bounds["lat"][0] <= lat <= bounds["lat"][1]):
                    regions.add(region)
        
        return list(regions)
    
    def _get_or_create_region(self, region_name: str, stats: Dict) -> str:
        """获取或创建地区实体"""
        # 检查缓存
        if region_name in self.region_cache:
            return self.region_cache[region_name]
        
        # 创建新的地区实体
        entity_id = self.kg.add_entity_smart(
            region_name,
            "地区",
            {
                "province": region_name,
                "coastal": True
            }
        )
        
        if entity_id:
            self.region_cache[region_name] = entity_id
            stats["entities_added"] += 1
        
        return entity_id
    
    def load_with_filters(self, csv_path: str, 
                         year_start: int = None,
                         year_end: int = None,
                         has_chinese_name: bool = False) -> Dict:
        """
        带过滤条件的加载
        
        Args:
            csv_path: CSV文件路径
            year_start: 起始年份
            year_end: 结束年份
            has_chinese_name: 是否只加载有中文名的台风
        
        Returns:
            统计信息
        """
        logger.info(f"📥 开始加载台风数据（带过滤）")
        logger.info(f"  年份范围: {year_start or '无限制'} - {year_end or '无限制'}")
        logger.info(f"  只加载有名称: {has_chinese_name}")
        
        # 读取CSV
        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
        except:
            df = pd.read_csv(csv_path, encoding='gbk')
        
        # 数据清洗
        df = self._clean_data(df)
        
        # 过滤
        if year_start or year_end:
            df['year'] = df['台风编号'].astype(str).str[:4].astype(int)
            if year_start:
                df = df[df['year'] >= year_start]
            if year_end:
                df = df[df['year'] <= year_end]
        
        if has_chinese_name:
            df = df[df['台风中文名称'].notna() & (df['台风中文名称'] != '-')]
        
        logger.info(f"  过滤后记录数: {len(df)}")
        
        # 处理
        stats = {
            "total_records": len(df),
            "unique_typhoons": 0,
            "entities_added": 0,
            "relations_added": 0,
            "time_start": datetime.now()
        }
        
        typhoon_groups = df.groupby('台风编号')
        stats["unique_typhoons"] = len(typhoon_groups)
        
        logger.info(f"  唯一台风数: {stats['unique_typhoons']}")
        
        for typhoon_id, group in typhoon_groups:
            try:
                self._process_typhoon(typhoon_id, group, stats)
            except Exception as e:
                logger.error(f"  ✗ 处理台风 {typhoon_id} 失败: {e}")
        
        stats["time_end"] = datetime.now()
        stats["duration"] = (stats["time_end"] - stats["time_start"]).total_seconds()
        
        logger.info(f"\n✅ 数据加载完成!")
        logger.info(f"  实体总数: {stats['entities_added']}")
        logger.info(f"  关系总数: {stats['relations_added']}")
        logger.info(f"  耗时: {stats['duration']:.2f} 秒")
        
        return stats


# ============ 使用示例 ============

def initialize_full_database():
    """完整数据库初始化（全量数据）"""
    print("\n" + "="*70)
    print("🌊 台风知识图谱 - 完整数据库初始化")
    print("="*70)
    
    # 初始化知识图谱
    kg = KnowledgeGraphCore("typhoon_kg_full.db")
    initializer = TyphoonKGInitializer(kg)
    
    # 加载数据
    stats = initializer.load_from_csv(
        "typhoon_data.csv"
    )
    
    print("\n" + "="*70)
    print("✨ 初始化完成！")
    print("="*70)
    print(f"📊 统计信息:")
    print(f"  总记录数: {stats['total_records']}")
    print(f"  台风数量: {stats['unique_typhoons']}")
    print(f"  实体总数: {stats['entities_added']}")
    print(f"  关系总数: {stats['relations_added']}")
    print(f"  耗时: {stats['duration']:.2f} 秒")


def initialize_recent_typhoons():
    """初始化近年台风（2000年至今）"""
    print("\n" + "="*70)
    print("🌊 台风知识图谱 - 初始化近年数据（2000-2024）")
    print("="*70)
    
    # 初始化知识图谱
    kg = KnowledgeGraphCore("typhoon_kg_recent.db")
    initializer = TyphoonKGInitializer(kg)
    
    # 加载2000年以后的数据
    stats = initializer.load_with_filters(
        "typhoon_data.csv",
        year_start=2000,
        year_end=2024,
        has_chinese_name=True  # 只加载有中文名的
    )
    
    print("\n" + "="*70)
    print("✨ 初始化完成！")
    print("="*70)
    print(f"📊 统计信息:")
    print(f"  总记录数: {stats['total_records']}")
    print(f"  台风数量: {stats['unique_typhoons']}")
    print(f"  实体总数: {stats['entities_added']}")
    print(f"  关系总数: {stats['relations_added']}")
    print(f"  耗时: {stats['duration']:.2f} 秒")


def initialize_sample():
    """初始化样本数据（快速测试）"""
    print("\n" + "="*70)
    print("🌊 台风知识图谱 - 初始化样本数据")
    print("="*70)
    
    # 初始化知识图谱
    kg = KnowledgeGraphCore("typhoon_kg_sample.db")
    initializer = TyphoonKGInitializer(kg)
    
    # 只加载1000条记录做测试
    stats = initializer.load_from_csv(
        "typhoon_data.csv",
        sample_size=1000
    )
    
    print("\n" + "="*70)
    print("✨ 初始化完成！")
    print("="*70)
    print(f"📊 统计信息:")
    print(f"  总记录数: {stats['total_records']}")
    print(f"  台风数量: {stats['unique_typhoons']}")
    print(f"  实体总数: {stats['entities_added']}")
    print(f"  关系总数: {stats['relations_added']}")
    print(f"  耗时: {stats['duration']:.2f} 秒")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "full":
            initialize_full_database()
        elif mode == "recent":
            initialize_recent_typhoons()
        elif mode == "sample":
            initialize_sample()
        else:
            print("用法: python typhoon_kg_initializer.py [full|recent|sample]")
    else:
        # 默认使用样本模式
        print("提示: 使用样本模式（快速测试）")
        print("  完整加载: python typhoon_kg_initializer.py full")
        print("  近年数据: python typhoon_kg_initializer.py recent")
        print("  样本数据: python typhoon_kg_initializer.py sample")
        print()
        initialize_sample()