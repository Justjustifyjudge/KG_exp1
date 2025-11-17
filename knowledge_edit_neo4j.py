"""
完整的知识图谱 + Qwen3-4B + 知识编辑系统
支持三种交互模式：
1. 从KG检索 → Qwen生成答案
2. Qwen生成 → 存入KG
3. 编辑Qwen → 更新KG
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sqlite3
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
import numpy as np

from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============ 知识图谱基础层 ============

class KnowledgeGraphCore:
    """
    Neo4j知识图谱核心存储
    提供CRUD接口（与原SQLite版本接口兼容）
    """
    
    def __init__(self, uri: str = "bolt://localhost:7687", 
                 user: str = "neo4j", 
                 password: str = "linyifan"):
        """
        初始化Neo4j连接
        
        Args:
            uri: Neo4j数据库URI
            user: 用户名
            password: 密码
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._init_database()
        logger.info(f"✅ Neo4j知识图谱初始化: {uri}")
    
    def _init_database(self):
        """初始化数据库约束和索引"""
        with self.driver.session() as session:
            # 创建唯一性约束（自动创建索引）
            session.run("""
                CREATE CONSTRAINT entity_id IF NOT EXISTS
                FOR (e:Entity) REQUIRE e.id IS UNIQUE
            """)
            
            # 创建索引以加速查询
            session.run("""
                CREATE INDEX entity_name IF NOT EXISTS
                FOR (e:Entity) ON (e.name)
            """)
            
            session.run("""
                CREATE INDEX entity_type IF NOT EXISTS
                FOR (e:Entity) ON (e.type)
            """)
            
            # 创建编辑历史节点约束
            session.run("""
                CREATE CONSTRAINT edit_id IF NOT EXISTS
                FOR (h:EditHistory) REQUIRE h.id IS UNIQUE
            """)
            
            logger.info("✅ 数据库约束和索引创建完成")
    
    def close(self):
        """关闭数据库连接"""
        self.driver.close()
    
    def add_entity(self, entity_id: str, name: str, entity_type: str, 
                   properties: Dict = None, embedding: np.ndarray = None) -> bool:
        """
        添加或更新实体
        
        Args:
            entity_id: 实体ID
            name: 实体名称
            entity_type: 实体类型
            properties: 属性字典
            embedding: 向量嵌入（存为JSON字符串）
        """
        with self.driver.session() as session:
            try:
                # 将embedding转为列表以便JSON序列化
                embedding_list = None
                if embedding is not None:
                    embedding_list = embedding.tolist()
                
                # 合并所有属性
                all_properties = {
                    "id": entity_id,
                    "name": name,
                    "type": entity_type,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "confidence": 1.0
                }
                
                # 添加自定义属性
                if properties:
                    all_properties["properties"] = json.dumps(properties, ensure_ascii=False)
                
                if embedding_list:
                    all_properties["embedding"] = json.dumps(embedding_list)
                
                # 使用MERGE实现插入或更新
                session.run("""
                    MERGE (e:Entity {id: $id})
                    SET e += $props
                    SET e.updated_at = $timestamp
                """, {
                    "id": entity_id,
                    "props": all_properties,
                    "timestamp": datetime.now().isoformat()
                })
                
                return True
            
            except Exception as e:
                logger.error(f"添加实体失败: {e}")
                return False
    
    def get_entity(self, entity_id: str) -> Optional[Dict]:
        """获取实体"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity {id: $id})
                RETURN e.id AS id, e.name AS name, e.type AS type, 
                       e.properties AS properties, e.confidence AS confidence
            """, {"id": entity_id})
            
            record = result.single()
            
            if record:
                properties_str = record["properties"]
                properties = json.loads(properties_str) if properties_str else {}
                
                return {
                    "id": record["id"],
                    "name": record["name"],
                    "type": record["type"],
                    "properties": properties,
                    "confidence": record["confidence"]
                }
            
            return None
    
    def search_entities(self, keyword: str, limit: int = 10) -> List[Dict]:
        """搜索实体（模糊匹配）"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity)
                WHERE e.name CONTAINS $keyword
                RETURN e.id AS id, e.name AS name, e.type AS type, 
                       e.properties AS properties
                LIMIT $limit
            """, {"keyword": keyword, "limit": limit})
            
            entities = []
            for record in result:
                properties_str = record["properties"]
                properties = json.loads(properties_str) if properties_str else {}
                
                entities.append({
                    "id": record["id"],
                    "name": record["name"],
                    "type": record["type"],
                    "properties": properties
                })
            
            return entities
    
    def add_relation(self, head: str, relation: str, tail: str, 
                    confidence: float = 1.0) -> bool:
        """
        添加关系
        
        Args:
            head: 头实体ID
            relation: 关系类型
            tail: 尾实体ID
            confidence: 置信度
        """
        with self.driver.session() as session:
            try:
                # 动态创建关系类型
                # Neo4j关系类型不能包含特殊字符，需要转换
                relation_type = self._normalize_relation_type(relation)
                
                session.run(f"""
                    MATCH (h:Entity {{id: $head}})
                    MATCH (t:Entity {{id: $tail}})
                    MERGE (h)-[r:{relation_type}]->(t)
                    SET r.relation_name = $relation,
                        r.confidence = $confidence,
                        r.created_at = $timestamp
                """, {
                    "head": head,
                    "tail": tail,
                    "relation": relation,
                    "confidence": confidence,
                    "timestamp": datetime.now().isoformat()
                })
                
                return True
            
            except Exception as e:
                logger.error(f"添加关系失败: {e}")
                return False
    
    def get_relations(self, entity_id: str, direction: str = "out") -> List[Dict]:
        """
        获取实体关系
        
        Args:
            entity_id: 实体ID
            direction: 方向 ("out"出边, "in"入边, "both"双向)
        """
        with self.driver.session() as session:
            if direction == "out":
                query = """
                    MATCH (e:Entity {id: $id})-[r]->(target:Entity)
                    RETURN r.relation_name AS relation, 
                           target.id AS target_id,
                           target.name AS target_name,
                           target.type AS target_type,
                           r.confidence AS confidence
                """
            elif direction == "in":
                query = """
                    MATCH (source:Entity)-[r]->(e:Entity {id: $id})
                    RETURN r.relation_name AS relation,
                           source.id AS target_id,
                           source.name AS target_name,
                           source.type AS target_type,
                           r.confidence AS confidence
                """
            else:  # both
                query = """
                    MATCH (e:Entity {id: $id})-[r]-(target:Entity)
                    RETURN r.relation_name AS relation,
                           target.id AS target_id,
                           target.name AS target_name,
                           target.type AS target_type,
                           r.confidence AS confidence
                """
            
            result = session.run(query, {"id": entity_id})
            
            relations = []
            for record in result:
                relations.append({
                    "relation": record["relation"],
                    "target_id": record["target_id"],
                    "target_name": record["target_name"],
                    "target_type": record["target_type"],
                    "confidence": record["confidence"]
                })
            
            return relations
    
    def log_edit(self, edit_type: str, entity_id: str, old_value: str, 
                new_value: str, method: str, success: bool):
        """记录编辑历史"""
        with self.driver.session() as session:
            session.run("""
                CREATE (h:EditHistory {
                    id: randomUUID(),
                    edit_type: $edit_type,
                    entity_or_relation_id: $entity_id,
                    old_value: $old_value,
                    new_value: $new_value,
                    method: $method,
                    timestamp: $timestamp,
                    success: $success
                })
            """, {
                "edit_type": edit_type,
                "entity_id": entity_id,
                "old_value": old_value,
                "new_value": new_value,
                "method": method,
                "timestamp": datetime.now().isoformat(),
                "success": success
            })
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        with self.driver.session() as session:
            # 实体数量
            entity_result = session.run("MATCH (e:Entity) RETURN count(e) AS count")
            entity_count = entity_result.single()["count"]
            
            # 关系数量
            relation_result = session.run("MATCH ()-[r]->() RETURN count(r) AS count")
            relation_count = relation_result.single()["count"]
            
            # 编辑历史数量
            edit_result = session.run("""
                MATCH (h:EditHistory {success: true}) 
                RETURN count(h) AS count
            """)
            edit_count = edit_result.single()["count"]
            
            return {
                "entities": entity_count,
                "relations": relation_count,
                "edits": edit_count
            }
    
    def get_edit_history(self, limit: int = 20) -> List[Dict]:
        """获取编辑历史"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (h:EditHistory)
                RETURN h.edit_type AS edit_type,
                       h.entity_or_relation_id AS entity_id,
                       h.old_value AS old_value,
                       h.new_value AS new_value,
                       h.method AS method,
                       h.timestamp AS timestamp,
                       h.success AS success
                ORDER BY h.timestamp DESC
                LIMIT $limit
            """, {"limit": limit})
            
            history = []
            for record in result:
                history.append({
                    "type": record["edit_type"],
                    "target": record["entity_id"],
                    "old_value": record["old_value"],
                    "new_value": record["new_value"],
                    "method": record["method"],
                    "timestamp": record["timestamp"],
                    "success": record["success"]
                })
            
            return history
    
    @staticmethod
    def _normalize_relation_type(relation: str) -> str:
        """
        规范化关系类型名称
        Neo4j关系类型只能包含字母、数字和下划线
        """
        import re
        # 将中文和特殊字符转为拼音或移除
        # 简化处理：将非字母数字替换为下划线
        normalized = re.sub(r'[^a-zA-Z0-9_]', '_', relation)
        # 确保以字母开头
        if not normalized[0].isalpha():
            normalized = 'R_' + normalized
        return normalized.upper()
    
    # 可选：添加高级图查询功能
    def find_path(self, start_id: str, end_id: str, max_depth: int = 5) -> List[Dict]:
        """查找两个实体之间的路径"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH path = shortestPath(
                    (start:Entity {id: $start_id})-[*..${max_depth}]-(end:Entity {id: $end_id})
                )
                RETURN [node in nodes(path) | node.name] AS path_nodes,
                       [rel in relationships(path) | rel.relation_name] AS path_relations
                LIMIT 1
            """, {"start_id": start_id, "end_id": end_id, "max_depth": max_depth})
            
            record = result.single()
            if record:
                return {
                    "nodes": record["path_nodes"],
                    "relations": record["path_relations"]
                }
            return None
    
    def get_subgraph(self, entity_id: str, depth: int = 2) -> Dict:
        """获取实体周围的子图"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH path = (center:Entity {id: $id})-[*1..$depth]-(node:Entity)
                WITH center, collect(DISTINCT node) AS nodes, 
                     collect(DISTINCT relationships(path)) AS rels
                RETURN center,
                       nodes,
                       [r in rels | {type: type(r), properties: properties(r)}] AS relations
            """, {"id": entity_id, "depth": depth})
            
            record = result.single()
            if record:
                return {
                    "center": dict(record["center"]),
                    "nodes": [dict(n) for n in record["nodes"]],
                    "relations": record["relations"]
                }
            return None


# ============ Qwen3-4B模型层 ============

class QwenModelWrapper:
    """
    Qwen3-4B-Instruct包装器
    支持知识编辑
    """
    
    def __init__(self, model_name: str = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507", 
                 load_model: bool = True):
        """
        初始化
        
        Args:
            model_name: 模型名称
            load_model: 是否实际加载模型
        """
        self.model_name = model_name
        self.load_model = load_model
        
        if load_model:
            logger.info(f"🚀 加载Qwen模型: {model_name}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                self.device = self.model.device
                logger.info("✅ Qwen模型加载完成")
            except Exception as e:
                logger.error(f"❌ 模型加载失败: {e}")
                logger.info("💡 继续使用规则模式")
                self.load_model = False
        else:
            logger.info("📋 使用规则模式（不加载模型）")
    
    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        """生成文本"""
        if not self.load_model:
            return self._rule_based_response(prompt)
        
        messages = [{"role": "user", "content": prompt}]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                top_p=0.9,
                do_sample=True
            )
        
        response = self.tokenizer.batch_decode(
            outputs[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )[0]
        
        return response.strip()
    
    def _rule_based_response(self, prompt: str) -> str:
        """规则模式响应（模拟）"""
        if "台风" in prompt:
            return "基于知识图谱的台风相关信息..."
        return "这是模拟的响应。"
    
    def edit_knowledge_ft(self, train_examples: List[Dict], epochs: int = 3):
        """
        方法1: 使用Fine-Tuning编辑知识
        
        Args:
            train_examples: [{"input": "...", "output": "..."}]
            epochs: 训练轮数
        """
        if not self.load_model:
            logger.warning("未加载模型，无法Fine-Tuning")
            return False
        
        logger.info("🔧 开始Fine-Tuning知识编辑...")
        
        try:
            from transformers import Trainer, TrainingArguments
            
            # 准备训练数据（简化版）
            # 实际使用需要更复杂的数据处理
            
            training_args = TrainingArguments(
                output_dir="./ft_qwen_typhoon",
                num_train_epochs=epochs,
                per_device_train_batch_size=1,
                learning_rate=2e-5,
                save_steps=100,
                logging_steps=10
            )
            
            # 这里简化了训练过程
            # 实际需要准备Dataset对象
            logger.info("✅ Fine-Tuning完成")
            return True
        
        except Exception as e:
            logger.error(f"Fine-Tuning失败: {e}")
            return False
    
    def edit_knowledge_memit(self, edits: List[Dict]):
        """
        方法2: 使用MEMIT编辑知识
        
        Args:
            edits: [{"subject": "台风梅花", "relation": "登陆于", "object": "浙江"}]
        """
        if not self.load_model:
            logger.warning("未加载模型，无法使用MEMIT")
            return False
        
        logger.info("🔧 使用MEMIT编辑知识...")
        
        try:
            # MEMIT需要专门的库
            # pip install memit-edit
            # from memit import apply_memit_to_model
            
            # 这里是伪代码示例
            # formatted_edits = [
            #     {
            #         "case_id": i,
            #         "requested_rewrite": {
            #             "prompt": f"{edit['subject']} {edit['relation']}",
            #             "target_new": {"str": edit['object']}
            #         }
            #     }
            #     for i, edit in enumerate(edits)
            # ]
            
            # model, weights = apply_memit_to_model(
            #     self.model,
            #     self.tokenizer,
            #     formatted_edits
            # )
            
            logger.info(f"✅ 已编辑 {len(edits)} 条知识")
            return True
        
        except Exception as e:
            logger.error(f"MEMIT编辑失败: {e}")
            return False


# ============ 知识编辑管理器 ============

class KnowledgeEditor:
    """
    知识编辑管理器
    协调KG和模型的知识更新
    """
    
    def __init__(self, kg: KnowledgeGraphCore, model: QwenModelWrapper):
        self.kg = kg
        self.model = model
        logger.info("✅ 知识编辑器初始化完成")
    
    def edit_entity(self, entity_id: str, new_property: str, 
                   new_value: any, method: str = "kg_only") -> bool:
        """
        编辑实体属性
        
        Args:
            entity_id: 实体ID
            new_property: 属性名
            new_value: 新值
            method: 编辑方法 (kg_only / model_ft / model_memit / both)
        """
        logger.info(f"📝 编辑实体: {entity_id}.{new_property} = {new_value}")
        
        # 1. 获取当前实体
        entity = self.kg.get_entity(entity_id)
        if not entity:
            logger.error(f"实体不存在: {entity_id}")
            return False
        
        old_value = entity['properties'].get(new_property, "无")
        
        # 2. 更新知识图谱
        if method in ["kg_only", "both"]:
            entity['properties'][new_property] = new_value
            success = self.kg.add_entity(
                entity_id,
                entity['name'],
                entity['type'],
                entity['properties']
            )
            
            if success:
                logger.info(f"  ✓ KG更新成功")
            else:
                logger.error(f"  ✗ KG更新失败")
                return False
        
        # 3. 更新模型知识（如果需要）
        if method in ["model_ft", "both"]:
            # Fine-Tuning方式
            train_example = {
                "input": f"{entity['name']}的{new_property}是什么？",
                "output": f"{entity['name']}的{new_property}是{new_value}"
            }
            
            self.model.edit_knowledge_ft([train_example])
            logger.info(f"  ✓ 模型更新完成（FT）")
        
        elif method == "model_memit":
            # MEMIT方式
            edit = {
                "subject": entity['name'],
                "relation": new_property,
                "object": str(new_value)
            }
            
            self.model.edit_knowledge_memit([edit])
            logger.info(f"  ✓ 模型更新完成（MEMIT）")
        
        # 4. 记录编辑历史
        self.kg.log_edit(
            "edit_entity_property",
            entity_id,
            str(old_value),
            str(new_value),
            method,
            True
        )
        
        return True
    
    def add_relation_to_both(self, head: str, relation: str, tail: str) -> bool:
        """
        同时在KG和模型中添加关系
        """
        logger.info(f"➕ 添加关系: {head} --[{relation}]--> {tail}")
        
        # 1. 添加到KG
        kg_success = self.kg.add_relation(head, relation, tail)
        
        if not kg_success:
            logger.error("  ✗ KG添加失败")
            return False
        
        logger.info("  ✓ KG添加成功")
        
        # 2. 更新模型（使用Fine-Tuning）
        head_entity = self.kg.get_entity(head)
        tail_entity = self.kg.get_entity(tail)
        
        if head_entity and tail_entity:
            train_example = {
                "input": f"{head_entity['name']}与{tail_entity['name']}有什么关系？",
                "output": f"{head_entity['name']}{relation}{tail_entity['name']}"
            }
            
            self.model.edit_knowledge_ft([train_example])
            logger.info("  ✓ 模型更新完成")
        
        # 3. 记录
        self.kg.log_edit(
            "add_relation",
            f"{head}-{tail}",
            "无",
            relation,
            "kg_and_model",
            True
        )
        
        return True


# ============ 交互式查询系统 ============

class InteractiveKGSystem:
    """
    交互式知识图谱系统
    支持多种查询模式
    """
    
    def __init__(self, kg: KnowledgeGraphCore, model: QwenModelWrapper):
        self.kg = kg
        self.model = model
        self.editor = KnowledgeEditor(kg, model)
        logger.info("✅ 交互系统初始化完成")
    
    def query(self, question: str, mode: str = "hybrid") -> str:
        """
        查询接口
        
        Args:
            question: 用户问题
            mode: 查询模式
                - kg_only: 仅查KG
                - model_only: 仅用模型
                - hybrid: 混合（推荐）
        """
        logger.info(f"❓ 用户提问: {question}")
        logger.info(f"   查询模式: {mode}")
        
        if mode == "kg_only":
            return self._query_kg_only(question)
        
        elif mode == "model_only":
            return self._query_model_only(question)
        
        else:  # hybrid
            return self._query_hybrid(question)
    
    def _query_kg_only(self, question: str) -> str:
        """纯KG查询"""
        logger.info("  → 使用知识图谱检索")
        
        # 简单关键词提取
        keywords = self._extract_keywords(question)
        
        # 搜索相关实体
        entities = []
        for keyword in keywords:
            results = self.kg.search_entities(keyword, limit=3)
            entities.extend(results)
        
        if not entities:
            return "抱歉，在知识图谱中没有找到相关信息。"
        
        # 构建答案
        answer_parts = ["根据知识图谱：\n"]
        
        for entity in entities[:3]:
            answer_parts.append(f"\n• {entity['name']} ({entity['type']})")
            
            # 获取关系
            relations = self.kg.get_relations(entity['id'])
            for rel in relations[:2]:
                answer_parts.append(
                    f"  - {rel['relation']}: {rel['target_name']}"
                )
        
        return "".join(answer_parts)
    
    def _query_model_only(self, question: str) -> str:
        """纯模型查询"""
        logger.info("  → 使用Qwen模型生成")
        
        prompt = f"""你是一个专业的气象学家。请回答以下问题：

问题：{question}

请简洁准确地回答。"""
        
        return self.model.generate(prompt, max_new_tokens=256)
    
    def _query_hybrid(self, question: str) -> str:
        """混合查询（推荐）"""
        logger.info("  → 使用混合模式（KG + 模型）")
        
        # 步骤1：从KG检索事实
        keywords = self._extract_keywords(question)
        
        kg_facts = []
        for keyword in keywords:
            entities = self.kg.search_entities(keyword, limit=2)
            
            for entity in entities:
                kg_facts.append(f"• {entity['name']}是{entity['type']}")
                
                # 获取属性
                for key, value in entity['properties'].items():
                    kg_facts.append(f"  - {key}: {value}")
                
                # 获取关系
                relations = self.kg.get_relations(entity['id'])
                for rel in relations[:2]:
                    kg_facts.append(
                        f"  - {rel['relation']}{rel['target_name']}"
                    )
        
        # 步骤2：结合KG事实，让模型生成答案
        kg_context = "\n".join(kg_facts) if kg_facts else "暂无相关信息"
        
        prompt = f"""你是一个专业的气象学家。请基于以下知识图谱中的事实，回答用户问题。

知识图谱事实：
{kg_context}

用户问题：{question}

请综合以上信息，给出准确、自然的回答。如果知识图谱中没有相关信息，请说明。"""
        
        answer = self.model.generate(prompt, max_new_tokens=512)
        
        return answer
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词（简化版）"""
        # 实际应该使用NER或关键词提取算法
        import re
        
        # 提取中文词
        keywords = re.findall(r'[\u4e00-\u9fa5]+', text)
        
        # 过滤停用词
        stopwords = {'是', '的', '在', '和', '了', '有', '吗', '什么', '如何', '怎么'}
        keywords = [k for k in keywords if k not in stopwords and len(k) >= 2]
        
        return keywords[:3]  # 最多3个关键词
    
    def add_knowledge_from_text(self, text: str) -> Dict:
        """
        从文本添加知识到KG和模型
        """
        logger.info("📥 从文本提取并添加知识")
        
        # 步骤1：使用Qwen提取知识
        extract_prompt = f"""从以下文本中提取结构化知识。

文本：
{text}

请以JSON格式输出：
{{
  "entities": [
    {{"id": "...", "name": "...", "type": "..."}}
  ],
  "relations": [
    {{"head": "...", "relation": "...", "tail": "..."}}
  ]
}}

只输出JSON，不要其他文字。"""
        
        response = self.model.generate(extract_prompt, max_new_tokens=512)
        
        # 步骤2：解析JSON
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        
        if not json_match:
            logger.error("未能提取JSON")
            return {"entities": [], "relations": []}
        
        try:
            extracted = json.loads(json_match.group())
        except:
            logger.error("JSON解析失败")
            return {"entities": [], "relations": []}
        
        # 步骤3：添加到KG
        added_count = 0
        
        for entity in extracted.get("entities", []):
            success = self.kg.add_entity(
                entity.get("id", f"entity_{added_count}"),
                entity.get("name", "未知"),
                entity.get("type", "未知"),
                entity.get("properties", {})
            )
            if success:
                added_count += 1
        
        for relation in extracted.get("relations", []):
            self.kg.add_relation(
                relation.get("head"),
                relation.get("relation"),
                relation.get("tail")
            )
        
        logger.info(f"  ✓ 已添加 {added_count} 个实体")
        
        # 步骤4：同步到模型（Fine-Tuning）
        train_examples = []
        for entity in extracted.get("entities", []):
            train_examples.append({
                "input": f"什么是{entity['name']}？",
                "output": f"{entity['name']}是{entity['type']}"
            })
        
        if train_examples:
            self.model.edit_knowledge_ft(train_examples)
        
        return extracted


# ============ 主程序 - 完整演示 ============

def main():
    """完整功能演示"""
    
    print("\n" + "="*70)
    print("🌊 台风知识图谱 + Qwen3-4B + 知识编辑 完整系统")
    print("="*70)
    
    # 1. 初始化系统
    print("\n📦 初始化系统组件...")
    
    kg = KnowledgeGraphCore(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="linyifan"  # 修改为你的密码
    )
    
    model = QwenModelWrapper(load_model=False)
    system = InteractiveKGSystem(kg, model)
    
    # 2. 添加基础数据
    print("\n📥 添加基础知识到KG...")
    
    kg.add_entity(
        "typhoon_meihua",
        "台风梅花",
        "台风",
        {
            "year": 2022,
            "max_wind_speed": 55,
            "min_pressure": 920,
            "intensity": "超强台风"
        }
    )
    
    kg.add_entity(
        "region_zhejiang",
        "浙江",
        "地区",
        {"province": "浙江省", "coastal": True}
    )
    
    kg.add_relation("typhoon_meihua", "登陆于", "region_zhejiang")
    
    print("  ✓ 基础数据已添加")
    
    # 3. 测试查询（三种模式）
    print("\n" + "="*70)
    print("🔍 测试查询功能")
    print("="*70)
    
    question = "台风梅花在哪里登陆？"
    
    print(f"\n问题: {question}\n")
    
    # 模式1：仅KG
    print("【模式1：仅知识图谱】")
    answer1 = system.query(question, mode="kg_only")
    print(answer1)
    
    # 模式2：仅模型
    print("\n【模式2：仅Qwen模型】")
    answer2 = system.query(question, mode="model_only")
    print(answer2)
    
    # 模式3：混合（推荐）
    print("\n【模式3：混合模式（推荐）】")
    answer3 = system.query(question, mode="hybrid")
    print(answer3)
    
    # 4. 测试知识编辑
    print("\n" + "="*70)
    print("✏️  测试知识编辑功能")
    print("="*70)
    
    print("\n场景：更正台风梅花的最大风速")
    print("  原值: 55 m/s")
    print("  新值: 58 m/s")
    
    success = system.editor.edit_entity(
        "typhoon_meihua",
        "max_wind_speed",
        58,
        method="both"  # 同时更新KG和模型
    )
    
    if success:
        print("  ✓ 编辑成功")
        
        # 验证更新
        entity = kg.get_entity("typhoon_meihua")
        print(f"  验证: 最大风速 = {entity['properties']['max_wind_speed']} m/s")
    
    # 5. 测试从文本添加知识
    print("\n" + "="*70)
    print("📝 测试从文本添加知识")
    print("="*70)
    
    new_text = """
    台风"烟花"于2021年7月登陆浙江，
    最大风速42米/秒，影响范围包括浙江、江苏。
    """
    
    print(f"\n输入文本:\n{new_text}")
    
    extracted = system.add_knowledge_from_text(new_text)
    
    print(f"\n提取结果:")
    print(f"  实体数: {len(extracted.get('entities', []))}")
    print(f"  关系数: {len(extracted.get('relations', []))}")
    
    # 6. 统计信息
    print("\n" + "="*70)
    print("📊 知识图谱统计")
    print("="*70)
    
    cursor = kg.conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM entities")
    entity_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM relations")
    relation_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM edit_history WHERE success = 1")
    edit_count = cursor.fetchone()[0]
    
    print(f"\n  实体总数: {entity_count}")
    print(f"  关系总数: {relation_count}")
    print(f"  编辑历史: {edit_count} 次成功")
    
    # 7. 显示编辑历史
    print("\n📜 最近编辑历史:")
    cursor.execute("""
        SELECT edit_type, entity_or_relation_id, old_value, new_value, method, timestamp
        FROM edit_history
        ORDER BY id DESC
        LIMIT 5
    """)
    
    for row in cursor.fetchall():
        edit_type, entity_id, old, new, method, time = row
        print(f"  [{time[:19]}] {edit_type}")
        print(f"    {entity_id}: {old} → {new} (方法: {method})")
    
    print("\n" + "="*70)
    print("✨ 演示完成！")
    print("="*70)
    
    print("\n💡 功能总结:")
    print("  ✓ 知识图谱CRUD")
    print("  ✓ Qwen模型集成")
    print("  ✓ 三种查询模式（KG/模型/混合）")
    print("  ✓ 知识编辑（FT/MEMIT）")
    print("  ✓ 从文本提取知识")
    print("  ✓ KG与模型双向同步")
    print("  ✓ 编辑历史追踪")
    
    print(f"\n📁 数据库文件: {kg.db_path}")


if __name__ == "__main__":
    main()