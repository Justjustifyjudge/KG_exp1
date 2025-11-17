"""
知识图谱 + Qwen + 知识编辑 REST API服务
提供完整的Web接口
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# 导入核心系统
from knowledge_edit_sql import (
    KnowledgeGraphCore,
    QwenModelWrapper,
    InteractiveKGSystem
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KGEditingAPI:
    """知识图谱编辑API服务"""
    
    def __init__(self, db_path: str = "typhoon_kg_with_editing.db",
                 load_model: bool = False,
                 host: str = "0.0.0.0",
                 port: int = 5000):
        
        self.app = Flask(__name__)
        CORS(self.app)
        
        # 初始化核心系统
        logger.info("🚀 初始化知识图谱系统...")
        self.kg = KnowledgeGraphCore(db_path)
        self.model = QwenModelWrapper(load_model=load_model)
        self.system = InteractiveKGSystem(self.kg, self.model)
        
        self.host = host
        self.port = port
        
        # 注册路由
        self._register_routes()
        
        logger.info("✅ API服务初始化完成")
    
    def _register_routes(self):
        """注册所有API路由"""
        
        # ============ 基础接口 ============
        
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            """健康检查"""
            return jsonify({
                "status": "healthy",
                "service": "KG + Qwen + Editing System",
                "model_loaded": self.model.load_model,
                "timestamp": datetime.now().isoformat()
            })
        
        @self.app.route('/api/stats', methods=['GET'])
        def get_stats():
            """获取统计信息"""
            cursor = self.kg.conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM entities")
            entity_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM relations")
            relation_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM edit_history WHERE success = 1")
            edit_count = cursor.fetchone()[0]
            
            return jsonify({
                "success": True,
                "data": {
                    "entities": entity_count,
                    "relations": relation_count,
                    "edits": edit_count
                }
            })
        
        # ============ 知识图谱接口 ============
        
        @self.app.route('/api/entity', methods=['POST'])
        def add_entity():
            """添加实体"""
            data = request.get_json()
            
            success = self.kg.add_entity(
                data['id'],
                data['name'],
                data['type'],
                data.get('properties', {})
            )
            
            return jsonify({
                "success": success,
                "entity_id": data['id']
            })
        
        @self.app.route('/api/entity/<entity_id>', methods=['GET'])
        def get_entity(entity_id):
            """获取实体"""
            entity = self.kg.get_entity(entity_id)
            
            if entity:
                # 获取关系
                relations = self.kg.get_relations(entity_id)
                entity['relations'] = relations
                
                return jsonify({
                    "success": True,
                    "data": entity
                })
            else:
                return jsonify({
                    "success": False,
                    "error": "Entity not found"
                }), 404
        
        @self.app.route('/api/search', methods=['GET'])
        def search_entities():
            """搜索实体"""
            keyword = request.args.get('q', '')
            limit = int(request.args.get('limit', 10))
            
            results = self.kg.search_entities(keyword, limit)
            
            return jsonify({
                "success": True,
                "data": results,
                "count": len(results)
            })
        
        @self.app.route('/api/relation', methods=['POST'])
        def add_relation():
            """添加关系"""
            data = request.get_json()
            
            success = self.kg.add_relation(
                data['head'],
                data['relation'],
                data['tail'],
                data.get('confidence', 1.0)
            )
            
            return jsonify({"success": success})
        
        # ============ 查询接口 ============
        
        @self.app.route('/api/query', methods=['POST'])
        def query():
            """
            智能查询
            
            请求体:
            {
                "question": "台风梅花在哪里登陆？",
                "mode": "hybrid"  // kg_only / model_only / hybrid
            }
            """
            data = request.get_json()
            
            question = data.get('question', '')
            mode = data.get('mode', 'hybrid')
            
            if not question:
                return jsonify({
                    "success": False,
                    "error": "Missing parameter: question"
                }), 400
            
            try:
                answer = self.system.query(question, mode)
                
                return jsonify({
                    "success": True,
                    "data": {
                        "question": question,
                        "answer": answer,
                        "mode": mode
                    }
                })
            
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        # ============ 知识编辑接口 ============
        
        @self.app.route('/api/edit/entity', methods=['POST'])
        def edit_entity():
            """
            编辑实体属性
            
            请求体:
            {
                "entity_id": "typhoon_meihua",
                "property": "max_wind_speed",
                "value": 58,
                "method": "both"  // kg_only / model_ft / model_memit / both
            }
            """
            data = request.get_json()
            
            entity_id = data.get('entity_id')
            property_name = data.get('property')
            value = data.get('value')
            method = data.get('method', 'kg_only')
            
            if not all([entity_id, property_name, value is not None]):
                return jsonify({
                    "success": False,
                    "error": "Missing required parameters"
                }), 400
            
            try:
                success = self.system.editor.edit_entity(
                    entity_id,
                    property_name,
                    value,
                    method
                )
                
                return jsonify({
                    "success": success,
                    "message": f"Entity {entity_id} updated"
                })
            
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        @self.app.route('/api/edit/relation', methods=['POST'])
        def add_relation_both():
            """
            添加关系到KG和模型
            
            请求体:
            {
                "head": "typhoon_meihua",
                "relation": "影响",
                "tail": "region_jiangsu"
            }
            """
            data = request.get_json()
            
            head = data.get('head')
            relation = data.get('relation')
            tail = data.get('tail')
            
            if not all([head, relation, tail]):
                return jsonify({
                    "success": False,
                    "error": "Missing required parameters"
                }), 400
            
            try:
                success = self.system.editor.add_relation_to_both(
                    head, relation, tail
                )
                
                return jsonify({
                    "success": success,
                    "message": "Relation added to both KG and model"
                })
            
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        @self.app.route('/api/edit/from_text', methods=['POST'])
        def add_from_text():
            """
            从文本提取并添加知识
            
            请求体:
            {
                "text": "台风烟花于2021年7月登陆浙江..."
            }
            """
            data = request.get_json()
            
            text = data.get('text', '')
            
            if not text:
                return jsonify({
                    "success": False,
                    "error": "Missing parameter: text"
                }), 400
            
            try:
                extracted = self.system.add_knowledge_from_text(text)
                
                return jsonify({
                    "success": True,
                    "data": extracted,
                    "summary": {
                        "entities_added": len(extracted.get('entities', [])),
                        "relations_added": len(extracted.get('relations', []))
                    }
                })
            
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        # ============ 编辑历史接口 ============
        
        @self.app.route('/api/edit/history', methods=['GET'])
        def get_edit_history():
            """获取编辑历史"""
            limit = int(request.args.get('limit', 20))
            
            cursor = self.kg.conn.cursor()
            cursor.execute("""
                SELECT id, edit_type, entity_or_relation_id, 
                       old_value, new_value, method, timestamp, success
                FROM edit_history
                ORDER BY id DESC
                LIMIT ?
            """, (limit,))
            
            history = []
            for row in cursor.fetchall():
                history.append({
                    "id": row[0],
                    "type": row[1],
                    "target": row[2],
                    "old_value": row[3],
                    "new_value": row[4],
                    "method": row[5],
                    "timestamp": row[6],
                    "success": bool(row[7])
                })
            
            return jsonify({
                "success": True,
                "data": history,
                "count": len(history)
            })
        
        # ============ 批量操作接口 ============
        
        @self.app.route('/api/batch/edit', methods=['POST'])
        def batch_edit():
            """
            批量编辑知识
            
            请求体:
            {
                "edits": [
                    {
                        "type": "edit_entity",
                        "entity_id": "...",
                        "property": "...",
                        "value": ...
                    },
                    {
                        "type": "add_relation",
                        "head": "...",
                        "relation": "...",
                        "tail": "..."
                    }
                ],
                "method": "both"
            }
            """
            data = request.get_json()
            
            edits = data.get('edits', [])
            method = data.get('method', 'kg_only')
            
            results = []
            
            for edit in edits:
                try:
                    if edit['type'] == 'edit_entity':
                        success = self.system.editor.edit_entity(
                            edit['entity_id'],
                            edit['property'],
                            edit['value'],
                            method
                        )
                    
                    elif edit['type'] == 'add_relation':
                        success = self.system.editor.add_relation_to_both(
                            edit['head'],
                            edit['relation'],
                            edit['tail']
                        )
                    
                    else:
                        success = False
                    
                    results.append({
                        "edit": edit,
                        "success": success
                    })
                
                except Exception as e:
                    results.append({
                        "edit": edit,
                        "success": False,
                        "error": str(e)
                    })
            
            success_count = sum(1 for r in results if r['success'])
            
            return jsonify({
                "success": True,
                "results": results,
                "summary": {
                    "total": len(edits),
                    "success": success_count,
                    "failed": len(edits) - success_count
                }
            })
    
    def run(self, debug: bool = False):
        """启动服务"""
        logger.info("="*70)
        logger.info("🚀 启动知识图谱编辑API服务")
        logger.info("="*70)
        logger.info(f"   地址: http://{self.host}:{self.port}")
        logger.info(f"   模型: {'已加载' if self.model.load_model else '未加载（规则模式）'}")
        logger.info("="*70)
        
        print("\n📚 API端点列表:")
        print("   GET  /api/health           - 健康检查")
        print("   GET  /api/stats            - 统计信息")
        print("   GET  /api/entity/<id>      - 获取实体")
        print("   GET  /api/search?q=...     - 搜索实体")
        print("   POST /api/entity           - 添加实体")
        print("   POST /api/relation         - 添加关系")
        print("   POST /api/query            - 智能查询")
        print("   POST /api/edit/entity      - 编辑实体")
        print("   POST /api/edit/relation    - 编辑关系")
        print("   POST /api/edit/from_text   - 从文本添加")
        print("   GET  /api/edit/history     - 编辑历史")
        print("   POST /api/batch/edit       - 批量编辑")
        
        print("\n✨ 服务启动中...\n")
        
        self.app.run(
            host=self.host,
            port=self.port,
            debug=debug,
            threaded=True
        )


# ============ 客户端SDK ============

class KGEditingClient:
    """Python客户端SDK"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url.rstrip('/')
        import requests
        self.session = requests.Session()
    
    def query(self, question: str, mode: str = "hybrid") -> str:
        """查询"""
        response = self.session.post(
            f"{self.base_url}/api/query",
            json={"question": question, "mode": mode}
        )
        result = response.json()
        
        if result['success']:
            return result['data']['answer']
        else:
            raise Exception(result.get('error'))
    
    def edit_entity(self, entity_id: str, property_name: str, 
                   value: any, method: str = "both"):
        """编辑实体"""
        response = self.session.post(
            f"{self.base_url}/api/edit/entity",
            json={
                "entity_id": entity_id,
                "property": property_name,
                "value": value,
                "method": method
            }
        )
        return response.json()
    
    def add_knowledge_from_text(self, text: str):
        """从文本添加知识"""
        response = self.session.post(
            f"{self.base_url}/api/edit/from_text",
            json={"text": text}
        )
        return response.json()
    
    def get_edit_history(self, limit: int = 20):
        """获取编辑历史"""
        response = self.session.get(
            f"{self.base_url}/api/edit/history",
            params={"limit": limit}
        )
        result = response.json()
        
        if result['success']:
            return result['data']
        else:
            raise Exception(result.get('error'))


# ============ 使用示例 ============

def start_server():
    """启动服务器"""
    api = KGEditingAPI(
        db_path="typhoon_kg_with_editing.db",
        load_model=False,  # 设为True加载真实Qwen模型
        host="0.0.0.0",
        port=5000
    )
    
    api.run(debug=False)


def test_client():
    """测试客户端"""
    from datetime import datetime
    
    print("\n🧪 测试知识图谱编辑API")
    print("="*70)
    
    client = KGEditingClient("http://localhost:5000")
    
    # 1. 查询
    print("\n1️⃣  测试查询:")
    try:
        answer = client.query("台风梅花在哪里登陆？", mode="hybrid")
        print(f"   问题: 台风梅花在哪里登陆？")
        print(f"   回答: {answer}")
    except Exception as e:
        print(f"   错误: {e}")
    
    # 2. 编辑实体
    print("\n2️⃣  测试编辑实体:")
    try:
        result = client.edit_entity(
            "typhoon_meihua",
            "max_wind_speed",
            60,
            method="both"
        )
        print(f"   结果: {result}")
    except Exception as e:
        print(f"   错误: {e}")
    
    # 3. 从文本添加知识
    print("\n3️⃣  测试从文本添加知识:")
    try:
        result = client.add_knowledge_from_text(
            "台风利奇马2019年8月登陆浙江温岭，最大风速52米/秒。"
        )
        print(f"   提取实体: {result['summary']['entities_added']}个")
        print(f"   提取关系: {result['summary']['relations_added']}个")
    except Exception as e:
        print(f"   错误: {e}")
    
    # 4. 查看编辑历史
    print("\n4️⃣  编辑历史:")
    try:
        history = client.get_edit_history(limit=5)
        for item in history:
            print(f"   [{item['timestamp'][:19]}] {item['type']}")
            print(f"     {item['target']}: {item['old_value']} → {item['new_value']}")
    except Exception as e:
        print(f"   错误: {e}")
    
    print("\n✅ 测试完成")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_client()
    else:
        start_server()