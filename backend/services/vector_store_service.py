import os
from datetime import datetime
import json
from typing import List, Dict, Any
import logging
from pathlib import Path
from pymilvus import connections, utility
from pymilvus import Collection, DataType, FieldSchema, CollectionSchema
from utils.config import VectorDBProvider, MILVUS_CONFIG, CHROMA_CONFIG  # Updated import
import re
from pypinyin import pinyin, Style
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

class VectorDBConfig:
    """
    向量数据库配置类，用于存储和管理向量数据库的配置信息
    """
    def __init__(self, provider: str, index_mode: str):
        """
        初始化向量数据库配置
        
        参数:
            provider: 向量数据库提供商名称
            index_mode: 索引模式
        """
        self.provider = provider
        self.index_mode = index_mode
        self.milvus_uri = MILVUS_CONFIG["uri"]
        self.chroma_persist_directory = CHROMA_CONFIG["persist_directory"]

    def _get_milvus_index_type(self, index_mode: str) -> str:
        """
        根据索引模式获取Milvus索引类型
        
        参数:
            index_mode: 索引模式
            
        返回:
            对应的Milvus索引类型
        """
        return MILVUS_CONFIG["index_types"].get(index_mode, "FLAT")
    
    def _get_milvus_index_params(self, index_mode: str) -> Dict[str, Any]:
        """
        根据索引模式获取Milvus索引参数
        
        参数:
            index_mode: 索引模式
            
        返回:
            对应的Milvus索引参数字典
        """
        return MILVUS_CONFIG["index_params"].get(index_mode, {})

    def _get_chroma_index_type(self, index_mode: str) -> str:
        """
        根据索引模式获取Chroma索引类型
        
        参数:
            index_mode: 索引模式
            
        返回:
            对应的Chroma索引类型
        """
        return CHROMA_CONFIG["index_types"].get(index_mode, "hnsw")

class VectorStoreService:
    """
    向量存储服务类，提供向量数据的索引、查询和管理功能
    """
    _instance = None
    _embedding_model = None
    _chroma_client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStoreService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """
        初始化向量存储服务
        """
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.initialized_dbs = {}
            # 确保存储目录存在
            os.makedirs("03-vector-store", exist_ok=True)
            os.makedirs(CHROMA_CONFIG["persist_directory"], exist_ok=True)
            # 初始化 Chroma 客户端
            self._init_chroma_client()
    
    def _init_chroma_client(self):
        """
        初始化 Chroma 客户端
        """
        if self._chroma_client is None:
            from chromadb import PersistentClient
            from chromadb.config import Settings
            
            self._chroma_client = PersistentClient(
                path=CHROMA_CONFIG["persist_directory"],
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

    def _get_chroma_client(self):
        """
        获取 Chroma 客户端实例
        """
        if self._chroma_client is None:
            self._init_chroma_client()
        return self._chroma_client

    def _get_embedding_model(self):
        """
        获取或创建 embedding 模型实例
        """
        if self._embedding_model is None:
            self._embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"
            )
        return self._embedding_model

    def _get_milvus_index_type(self, config: VectorDBConfig) -> str:
        """
        从配置对象获取Milvus索引类型
        
        参数:
            config: 向量数据库配置对象
            
        返回:
            Milvus索引类型
        """
        return config._get_milvus_index_type(config.index_mode)
    
    def _get_milvus_index_params(self, config: VectorDBConfig) -> Dict[str, Any]:
        """
        从配置对象获取Milvus索引参数
        
        参数:
            config: 向量数据库配置对象
            
        返回:
            Milvus索引参数字典
        """
        return config._get_milvus_index_params(config.index_mode)

    def _get_chroma_index_type(self, config: VectorDBConfig) -> str:
        """
        从配置对象获取Chroma索引类型
        
        参数:
            config: 向量数据库配置对象
            
        返回:
            Chroma索引类型
        """
        return config._get_chroma_index_type(config.index_mode)
    
    def index_embeddings(self, embedding_file: str, config: VectorDBConfig) -> Dict[str, Any]:
        """
        将嵌入向量索引到向量数据库
        
        参数:
            embedding_file: 嵌入向量文件路径
            config: 向量数据库配置对象
            
        返回:
            索引结果信息字典
        """
        start_time = datetime.now()
        
        # 读取embedding文件
        embeddings_data = self._load_embeddings(embedding_file)
        
        # 根据不同的数据库进行索引
        if config.provider == VectorDBProvider.MILVUS:
            result = self._index_to_milvus(embeddings_data, config)
        elif config.provider == VectorDBProvider.CHROMA:
            result = self._index_to_chroma(embeddings_data, config)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        return {
            "database": config.provider,
            "index_mode": config.index_mode,
            "total_vectors": len(embeddings_data["embeddings"]),
            "index_size": result.get("index_size", "N/A"),
            "processing_time": processing_time,
            "collection_name": result.get("collection_name", "N/A")
        }
    
    def _load_embeddings(self, file_path: str) -> Dict[str, Any]:
        """
        加载embedding文件，返回配置信息和embeddings
        
        参数:
            file_path: 嵌入向量文件路径
            
        返回:
            包含嵌入向量和元数据的字典
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Loading embeddings from {file_path}")
                
                if not isinstance(data, dict) or "embeddings" not in data:
                    raise ValueError("Invalid embedding file format: missing 'embeddings' key")
                    
                # 返回完整的数据，包括顶层配置
                logger.info(f"Found {len(data['embeddings'])} embeddings")
                return data
                
        except Exception as e:
            logger.error(f"Error loading embeddings from {file_path}: {str(e)}")
            raise
    
    def _sanitize_collection_name(self, name: str) -> str:
        """
        清理集合名称，将中文转换为拼音，确保只包含有效的ASCII字符
        
        参数:
            name: 原始名称
            
        返回:
            清理后的名称
        """
        # 将中文转换为拼音，不使用声调
        py_list = pinyin(name, style=Style.NORMAL, strict=False)
        # 将拼音列表转换为字符串，直接连接（不使用下划线）
        py_str = ''.join([''.join(item) for item in py_list])
        # 将非ASCII字符替换为下划线
        sanitized = re.sub(r'[^\x00-\x7F]+', '_', py_str)
        # 确保以字母或下划线开头
        if not (sanitized[0].isalpha() or sanitized[0] == '_'):
            sanitized = f"_{sanitized}"
        return sanitized

    def _index_to_milvus(self, embeddings_data: Dict[str, Any], config: VectorDBConfig) -> Dict[str, Any]:
        """
        将嵌入向量索引到Milvus数据库
        
        参数:
            embeddings_data: 嵌入向量数据
            config: 向量数据库配置对象
            
        返回:
            索引结果信息字典
        """
        try:
            # 使用 filename 作为 collection 名称前缀
            filename = embeddings_data.get("filename", "")
            # 如果有 .pdf 后缀，移除它
            base_name = filename.replace('.pdf', '') if filename else "doc"
            
            # 清理基础名称
            base_name = self._sanitize_collection_name(base_name)
            
            # Get embedding provider
            embedding_provider = embeddings_data.get("embedding_provider", "unknown")
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            collection_name = f"{base_name}_{embedding_provider}_{timestamp}"
            
            # 连接到Milvus
            connections.connect(
                alias="default", 
                uri=config.milvus_uri
            )
            
            # 从顶层配置获取向量维度
            vector_dim = int(embeddings_data.get("vector_dimension"))
            if not vector_dim:
                raise ValueError("Missing vector_dimension in embedding file")
            
            logger.info(f"Creating collection with dimension: {vector_dim}")
            
            # 定义字段
            fields = [
                {"name": "id", "dtype": "INT64", "is_primary": True, "auto_id": True},
                {"name": "content", "dtype": "VARCHAR", "max_length": 5000},
                {"name": "document_name", "dtype": "VARCHAR", "max_length": 255},
                {"name": "chunk_id", "dtype": "INT64"},
                {"name": "total_chunks", "dtype": "INT64"},
                {"name": "word_count", "dtype": "INT64"},
                {"name": "page_number", "dtype": "VARCHAR", "max_length": 10},
                {"name": "page_range", "dtype": "VARCHAR", "max_length": 10},
                # {"name": "chunking_method", "dtype": "VARCHAR", "max_length": 50},
                {"name": "embedding_provider", "dtype": "VARCHAR", "max_length": 50},
                {"name": "embedding_model", "dtype": "VARCHAR", "max_length": 50},
                {"name": "embedding_timestamp", "dtype": "VARCHAR", "max_length": 50},
                {
                    "name": "vector",
                    "dtype": "FLOAT_VECTOR",
                    "dim": vector_dim,
                    "params": self._get_milvus_index_params(config)
                }
            ]
            
            # 准备数据为列表格式
            entities = []
            for emb in embeddings_data["embeddings"]:
                entity = {
                    "content": str(emb["metadata"].get("content", "")),
                    "document_name": embeddings_data.get("filename", ""),  # 使用 filename 而不是 document_name
                    "chunk_id": int(emb["metadata"].get("chunk_id", 0)),
                    "total_chunks": int(emb["metadata"].get("total_chunks", 0)),
                    "word_count": int(emb["metadata"].get("word_count", 0)),
                    "page_number": str(emb["metadata"].get("page_number", 0)),
                    "page_range": str(emb["metadata"].get("page_range", "")),
                    # "chunking_method": str(emb["metadata"].get("chunking_method", "")),
                    "embedding_provider": embeddings_data.get("embedding_provider", ""),  # 从顶层配置获取
                    "embedding_model": embeddings_data.get("embedding_model", ""),  # 从顶层配置获取
                    "embedding_timestamp": str(emb["metadata"].get("embedding_timestamp", "")),
                    "vector": [float(x) for x in emb.get("embedding", [])]
                }
                entities.append(entity)
            
            logger.info(f"Creating Milvus collection: {collection_name}")
            
            # 创建collection
            # field_schemas = [
            #     FieldSchema(name=field["name"], 
            #                dtype=getattr(DataType, field["dtype"]),
            #                is_primary="is_primary" in field and field["is_primary"],
            #                auto_id="auto_id" in field and field["auto_id"],
            #                max_length=field.get("max_length"),
            #                dim=field.get("dim"),
            #                params=field.get("params"))
            #     for field in fields
            # ]

            field_schemas = []
            for field in fields:
                extra_params = {}
                if field.get('max_length') is not None:
                    extra_params['max_length'] = field['max_length']
                if field.get('dim') is not None:
                    extra_params['dim'] = field['dim']
                if field.get('params') is not None:
                    extra_params['params'] = field['params']
                field_schema = FieldSchema(
                    name=field["name"], 
                    dtype=getattr(DataType, field["dtype"]),
                    is_primary=field.get("is_primary", False),
                    auto_id=field.get("auto_id", False),
                    **extra_params
                )
                field_schemas.append(field_schema)

            schema = CollectionSchema(fields=field_schemas, description=f"Collection for {collection_name}")
            collection = Collection(name=collection_name, schema=schema)
            
            # 插入数据
            logger.info(f"Inserting {len(entities)} vectors")
            insert_result = collection.insert(entities)
            
            # 创建索引
            logger.info(f"Creating index with type: {config.index_mode}")
            index_params = {
                "metric_type": "L2",
                "index_type": self._get_milvus_index_type(config),
                "params": self._get_milvus_index_params(config)
            }
            collection.create_index(field_name="vector", index_params=index_params)
            collection.load()
            
            return {
                "index_size": len(insert_result.primary_keys),
                "collection_name": collection_name
            }
            
        except Exception as e:
            logger.error(f"Error indexing to Milvus: {str(e)}")
            raise
        
        finally:
            connections.disconnect("default")

    def _index_to_chroma(self, embeddings_data: Dict[str, Any], config: VectorDBConfig) -> Dict[str, Any]:
        """
        将嵌入向量索引到Chroma数据库
        
        参数:
            embeddings_data: 嵌入向量数据
            config: 向量数据库配置对象
            
        返回:
            索引结果信息字典
        """
        try:
            # 使用 filename 作为 collection 名称前缀
            filename = embeddings_data.get("filename", "")
            # 如果有 .pdf 后缀，移除它
            base_name = filename.replace('.pdf', '') if filename else "doc"
            
            # 清理基础名称
            base_name = self._sanitize_collection_name(base_name)
            
            # Get embedding provider
            embedding_provider = embeddings_data.get("embedding_provider", "unknown")
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            collection_name = f"{base_name}_{embedding_provider}_{timestamp}"
            
            logger.info(f"Creating Chroma collection: {collection_name}")

            # 准备数据
            texts = []
            metadatas = []
            embeddings = []
            
            for emb in embeddings_data["embeddings"]:
                texts.append(str(emb["metadata"].get("content", "")))
                metadata = {
                    "document_name": embeddings_data.get("filename", ""),
                    "chunk_id": int(emb["metadata"].get("chunk_id", 0)),
                    "total_chunks": int(emb["metadata"].get("total_chunks", 0)),
                    "word_count": int(emb["metadata"].get("word_count", 0)),
                    "page_number": str(emb["metadata"].get("page_number", 0)),
                    "page_range": str(emb["metadata"].get("page_range", "")),
                    "embedding_provider": embeddings_data.get("embedding_provider", ""),
                    "embedding_model": embeddings_data.get("embedding_model", ""),
                    "embedding_timestamp": str(emb["metadata"].get("embedding_timestamp", ""))
                }
                metadatas.append(metadata)
                embeddings.append([float(x) for x in emb.get("embedding", [])])

            logger.info(f"Prepared {len(texts)} documents for indexing")

            # 获取 Chroma 客户端并创建集合
            client = self._get_chroma_client()
            collection = client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )

            # 添加数据
            logger.info("Adding documents to Chroma...")
            collection.add(
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings,
                ids=[f"doc_{i}" for i in range(len(texts))]
            )

            logger.info(f"Successfully indexed {len(texts)} documents to Chroma collection: {collection_name}")

            return {
                "collection_name": collection_name,
                "index_size": len(texts)
            }

        except Exception as e:
            logger.error(f"Error indexing to Chroma: {str(e)}")
            raise

    def list_collections(self, provider: VectorDBProvider) -> List[Dict[str, Any]]:
        """
        列出指定提供商的所有集合
        
        参数:
            provider: 向量数据库提供商
            
        返回:
            集合列表，每个集合包含 id、name 和 count 字段
        """
        try:
            if provider == VectorDBProvider.MILVUS:
                # 连接到Milvus
                connections.connect(
                    alias="default",
                    uri=MILVUS_CONFIG["uri"]
                )
                
                # 获取所有集合名称
                collection_names = utility.list_collections()
                
                collections = []
                for name in collection_names:
                    try:
                        collection = Collection(name)
                        collection.load()
                        collections.append({
                            "id": name,
                            "name": name,
                            "count": collection.num_entities
                        })
                    except Exception as e:
                        logger.error(f"Error getting collection info for {name}: {str(e)}")
                        continue
                
                return collections
                
            elif provider == VectorDBProvider.CHROMA:
                client = self._get_chroma_client()
                collections = []
                
                # 获取所有集合
                for collection in client.list_collections():
                    try:
                        collections.append({
                            "id": collection.name,
                            "name": collection.name,
                            "count": collection.count()
                        })
                    except Exception as e:
                        logger.error(f"Error getting collection info for {collection.name}: {str(e)}")
                        continue
                
                return collections
            
            return []
            
        except Exception as e:
            logger.error(f"Error listing collections for {provider}: {str(e)}")
            return []

    def delete_collection(self, provider: str, collection_name: str) -> bool:
        """
        删除指定的集合
        
        参数:
            provider: 向量数据库提供商名称
            collection_name: 集合名称
            
        返回:
            是否删除成功
        """
        try:
            if provider == VectorDBProvider.MILVUS:
                connections.connect(alias="default", uri=MILVUS_CONFIG["uri"])
                utility.drop_collection(collection_name)
                connections.disconnect("default")
                return True
            elif provider == VectorDBProvider.CHROMA:
                client = self._get_chroma_client()
                try:
                    # 使用 Chroma 客户端的 API 删除集合
                    client.delete_collection(collection_name)
                    logger.info(f"Successfully deleted Chroma collection: {collection_name}")
                    return True
                except Exception as e:
                    logger.error(f"Error deleting Chroma collection {collection_name}: {str(e)}")
                    return False
            else:
                raise ValueError(f"Unsupported vector database provider: {provider}")
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            return False

    def get_collection_info(self, provider: VectorDBProvider, collection_name: str) -> Dict[str, Any]:
        """
        获取指定集合的详细信息
        
        参数:
            provider: 向量数据库提供商
            collection_name: 集合名称
            
        返回:
            集合详细信息字典
        """
        try:
            if provider == VectorDBProvider.MILVUS:
                # 连接到Milvus
                connections.connect(
                    alias="default",
                    uri=MILVUS_CONFIG["uri"]
                )
                
                collection = Collection(collection_name)
                collection.load()
                
                # 获取集合信息
                schema = collection.schema
                fields = [field.to_dict() for field in schema.fields]
                
                return {
                    "name": collection_name,
                    "num_entities": collection.num_entities,
                    "schema": {
                        "fields": fields
                    }
                }
                
            elif provider == VectorDBProvider.CHROMA:
                client = self._get_chroma_client()
                try:
                    collection = client.get_collection(collection_name)
                    return {
                        "name": collection_name,
                        "num_entities": collection.count(),
                        "schema": {
                            "fields": [
                                {
                                    "name": "embedding",
                                    "index_type": "hnsw",
                                    "params": {"space_type": "cosine"}
                                }
                            ]
                        }
                    }
                except Exception as e:
                    logger.error(f"Error getting Chroma collection info: {str(e)}")
                    raise ValueError(f"Collection {collection_name} not found")
            
            raise ValueError(f"Unsupported vector database provider: {provider}")
            
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            raise