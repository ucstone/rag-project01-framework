from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from pymilvus import connections, Collection, utility
from services.embedding_service import EmbeddingService
from utils.config import VectorDBProvider, MILVUS_CONFIG, CHROMA_CONFIG
from chromadb import PersistentClient
import os
import json

logger = logging.getLogger(__name__)

class SearchService:
    """
    搜索服务类，负责向量数据库的连接和向量搜索功能
    提供集合列表查询、向量相似度搜索和搜索结果保存等功能
    """
    def __init__(self):
        """
        初始化搜索服务
        创建嵌入服务实例，设置数据库连接URI，初始化搜索结果保存目录
        """
        self.embedding_service = EmbeddingService()
        self.milvus_uri = MILVUS_CONFIG["uri"]
        self.chroma_persist_directory = CHROMA_CONFIG["persist_directory"]
        self.search_results_dir = "04-search-results"
        os.makedirs(self.search_results_dir, exist_ok=True)
        self._init_chroma_client()

    def _init_chroma_client(self):
        """
        初始化 Chroma 客户端
        """
        from chromadb.config import Settings
        self.chroma_client = PersistentClient(
            path=self.chroma_persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

    def get_providers(self) -> List[Dict[str, str]]:
        """
        获取支持的向量数据库列表
        
        Returns:
            List[Dict[str, str]]: 支持的向量数据库提供商列表
        """
        return [
            {"id": VectorDBProvider.MILVUS.value, "name": "Milvus"},
            {"id": VectorDBProvider.CHROMA.value, "name": "Chroma"}
        ]

    def list_collections(self, provider: str = VectorDBProvider.MILVUS.value) -> List[Dict[str, Any]]:
        """
        获取指定向量数据库中的所有集合
        
        Args:
            provider (str): 向量数据库提供商，默认为Milvus
            
        Returns:
            List[Dict[str, Any]]: 集合信息列表，包含id、名称和实体数量
        """
        try:
            if provider == VectorDBProvider.MILVUS:
                connections.connect(
                    alias="default",
                    uri=self.milvus_uri
                )
                
                collections = []
                collection_names = utility.list_collections()
                
                for name in collection_names:
                    try:
                        collection = Collection(name)
                        collections.append({
                            "id": name,
                            "name": name,
                            "count": collection.num_entities
                        })
                    except Exception as e:
                        logger.error(f"Error getting info for collection {name}: {str(e)}")
                
                return collections
                
            elif provider == VectorDBProvider.CHROMA:
                collections = []
                for collection in self.chroma_client.list_collections():
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
            logger.error(f"Error listing collections: {str(e)}")
            raise
        finally:
            if provider == VectorDBProvider.MILVUS:
                connections.disconnect("default")

    def save_search_results(self, query: str, collection_id: str, results: List[Dict[str, Any]]) -> str:
        """
        保存搜索结果到JSON文件
        
        Args:
            query (str): 搜索查询文本
            collection_id (str): 集合ID
            results (List[Dict[str, Any]]): 搜索结果列表
            
        Returns:
            str: 保存文件的路径
            
        Raises:
            Exception: 保存文件时发生错误
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            # 使用集合ID的基础名称（去掉路径相关字符）
            collection_base = os.path.basename(collection_id)
            filename = f"search_{collection_base}_{timestamp}.json"
            filepath = os.path.join(self.search_results_dir, filename)
            
            search_data = {
                "query": query,
                "collection_id": collection_id,
                "timestamp": datetime.now().isoformat(),
                "results": results
            }
            
            logger.info(f"Saving search results to: {filepath}")
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(search_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Successfully saved search results to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving search results: {str(e)}")
            raise

    async def search(self, 
                    query: str, 
                    collection_id: str, 
                    top_k: int = 3, 
                    threshold: float = 0.7,
                    word_count_threshold: int = 20,
                    save_results: bool = False) -> Dict[str, Any]:
        """
        执行向量搜索
        """
        try:
            # 添加参数日志
            logger.info(f"Search parameters:")
            logger.info(f"- Query: {query}")
            logger.info(f"- Collection ID: {collection_id}")
            logger.info(f"- Top K: {top_k}")
            logger.info(f"- Threshold: {threshold}")
            logger.info(f"- Word Count Threshold: {word_count_threshold}")
            logger.info(f"- Save Results: {save_results} (type: {type(save_results)})")

            logger.info(f"Starting search with parameters - Collection: {collection_id}, Query: {query}, Top K: {top_k}")
            
            # 确定数据库提供商
            provider = self._determine_provider(collection_id)
            logger.info(f"Determined provider: {provider}")
            
            if provider == VectorDBProvider.MILVUS:
                return await self._search_milvus(
                    query, collection_id, top_k, threshold, word_count_threshold, save_results
                )
            elif provider == VectorDBProvider.CHROMA:
                return await self._search_chroma(
                    query, collection_id, top_k, threshold, word_count_threshold, save_results
                )
            else:
                raise ValueError(f"Unsupported vector database provider: {provider}")
            
        except Exception as e:
            logger.error(f"Error performing search: {str(e)}")
            raise

    def _determine_provider(self, collection_id: str) -> str:
        """
        根据集合ID确定数据库提供商
        """
        try:
            # 尝试从 Milvus 获取集合
            connections.connect(alias="default", uri=self.milvus_uri)
            collection = Collection(collection_id)
            collection.load()
            connections.disconnect("default")
            return VectorDBProvider.MILVUS
        except:
            try:
                # 尝试从 Chroma 获取集合
                collection = self.chroma_client.get_collection(collection_id)
                return VectorDBProvider.CHROMA
            except:
                raise ValueError(f"Collection {collection_id} not found in any supported database")

    async def _search_milvus(self, 
                           query: str, 
                           collection_id: str, 
                           top_k: int, 
                           threshold: float,
                           word_count_threshold: int,
                           save_results: bool) -> Dict[str, Any]:
        """
        在 Milvus 中执行搜索
        """
        try:
            connections.connect(alias="default", uri=self.milvus_uri)
            collection = Collection(collection_id)
            collection.load()
            
            # 获取集合的索引配置
            index_info = collection.index()
            if not index_info:
                raise ValueError(f"Collection {collection_id} has no index")
            
            # 获取索引的度量类型
            metric_type = index_info.params.get("metric_type", "L2")
            
            # 获取 embedding 配置
            sample_entity = collection.query(
                expr="id >= 0", 
                output_fields=["embedding_provider", "embedding_model"],
                limit=1
            )
            if not sample_entity:
                raise ValueError(f"Collection {collection_id} is empty")
            
            # 创建查询向量
            query_embedding = self.embedding_service.create_single_embedding(
                query,
                provider=sample_entity[0]["embedding_provider"],
                model=sample_entity[0]["embedding_model"]
            )
            
            # 根据度量类型设置搜索参数
            search_params = {
                "metric_type": metric_type,
                "params": {"nprobe": 10}
            }
            
            # 执行搜索
            results = collection.search(
                data=[query_embedding],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                expr=f"word_count >= {word_count_threshold}",
                output_fields=[
                    "content", "document_name", "chunk_id", "total_chunks",
                    "word_count", "page_number", "page_range",
                    "embedding_provider", "embedding_model", "embedding_timestamp"
                ]
            )
            
            return self._process_search_results(results, threshold, save_results, query, collection_id)
            
        finally:
            connections.disconnect("default")

    async def _search_chroma(self, 
                           query: str, 
                           collection_id: str, 
                           top_k: int, 
                           threshold: float,
                           word_count_threshold: int,
                           save_results: bool) -> Dict[str, Any]:
        """
        在 Chroma 中执行搜索
        """
        try:
            collection = self.chroma_client.get_collection(collection_id)
            
            # 获取集合的元数据以确定 embedding 配置
            collection_metadata = collection.metadata
            embedding_provider = collection_metadata.get("embedding_provider", "huggingface")
            embedding_model = collection_metadata.get("embedding_model", "BAAI/bge-base-zh")
            
            # 创建查询向量
            query_embedding = self.embedding_service.create_single_embedding(
                query,
                provider=embedding_provider,
                model=embedding_model
            )
            
            # 获取集合的索引配置
            index_type = collection_metadata.get("index_type", "hnsw")
            metric_type = collection_metadata.get("metric_type", "l2")
            
            # 执行搜索
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where={"word_count": {"$gte": word_count_threshold}},
                include=["metadatas", "documents", "distances"]
            )
            
            # 处理结果
            processed_results = []
            for i in range(len(results['documents'][0])):
                # 根据度量类型转换分数
                if metric_type.lower() == "l2":
                    score = 1 - results['distances'][0][i]  # 将 L2 距离转换为相似度
                elif metric_type.lower() == "cosine":
                    score = results['distances'][0][i]  # 余弦相似度直接使用
                else:
                    score = results['distances'][0][i]  # 其他度量类型直接使用
                
                if score >= threshold:
                    processed_results.append({
                        "text": results['documents'][0][i],
                        "score": float(score),
                        "metadata": results['metadatas'][0][i]
                    })
            
            response_data = {"results": processed_results}
            
            if save_results and processed_results:
                try:
                    filepath = self.save_search_results(query, collection_id, processed_results)
                    response_data["saved_filepath"] = filepath
                except Exception as e:
                    logger.error(f"Error saving results: {str(e)}")
                    response_data["save_error"] = str(e)
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error searching in Chroma: {str(e)}")
            raise

    def _process_search_results(self, results, threshold, save_results, query, collection_id):
        """
        处理搜索结果
        """
        processed_results = []
        for hits in results:
            for hit in hits:
                if hit.score >= threshold:
                    processed_results.append({
                        "text": hit.entity.content,
                        "score": float(hit.score),
                        "metadata": {
                            "source": hit.entity.document_name,
                            "page": hit.entity.page_number,
                            "chunk": hit.entity.chunk_id,
                            "total_chunks": hit.entity.total_chunks,
                            "page_range": hit.entity.page_range,
                            "embedding_provider": hit.entity.embedding_provider,
                            "embedding_model": hit.entity.embedding_model,
                            "embedding_timestamp": hit.entity.embedding_timestamp
                        }
                    })

        response_data = {"results": processed_results}
        
        if save_results and processed_results:
            try:
                filepath = self.save_search_results(query, collection_id, processed_results)
                response_data["saved_filepath"] = filepath
            except Exception as e:
                logger.error(f"Error saving results: {str(e)}")
                response_data["save_error"] = str(e)
        
        return response_data 