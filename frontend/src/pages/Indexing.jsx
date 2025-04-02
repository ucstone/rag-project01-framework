// src/pages/Indexing.jsx
import React, { useState, useEffect } from 'react';
import RandomImage from '../components/RandomImage';
import { apiBaseUrl } from '../config/config';

const Indexing = () => {
  const [embeddingFile, setEmbeddingFile] = useState('');
  const [vectorDb, setVectorDb] = useState('milvus');
  const [indexMode, setIndexMode] = useState('standard');
  const [status, setStatus] = useState('');
  const [embeddedFiles, setEmbeddedFiles] = useState([]);
  const [indexingResult, setIndexingResult] = useState(null);
  const [collections, setCollections] = useState([]);
  const [selectedCollection, setSelectedCollection] = useState('');
  const [collectionDetails, setCollectionDetails] = useState(null);
  const [providers, setProviders] = useState([]);
  const [selectedProvider, setSelectedProvider] = useState('milvus');

  // 数据库和索引模式的配置
  const dbConfigs = {
    pinecone: {
      modes: ['standard', 'hybrid']
    },
    milvus: {
      modes: ['flat', 'hnsw', 'auto']
    },
    qdrant: {
      modes: ['hnsw', 'custom']
    },
    weaviate: {
      modes: ['hnsw', 'flat']
    },
    chroma: {
      modes: ['hnsw', 'flat', 'cosine', 'l2']
    },
    faiss: {
      modes: ['flat', 'ivf', 'hnsw']
    }
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        // 获取providers列表
        const providersResponse = await fetch(`${apiBaseUrl}/providers`);
        const providersData = await providersResponse.json();
        if (providersData.providers && providersData.providers.length > 0) {
          setProviders(providersData.providers);
          // 设置默认选中的数据库
          const defaultProvider = providersData.providers[0].id;
          setSelectedProvider(defaultProvider);
          setVectorDb(defaultProvider);
          setIndexMode(dbConfigs[defaultProvider].modes[0]);
        }
      } catch (error) {
        console.error('Error fetching providers:', error);
        setStatus('Error loading providers');
      }
    };

    fetchData();
    fetchEmbeddedFiles();
  }, []); // 组件加载时只执行一次

  // 当选择的数据库改变时，更新相关状态
  useEffect(() => {
    if (selectedProvider) {
      setVectorDb(selectedProvider);
      setIndexMode(dbConfigs[selectedProvider].modes[0]);
      setCollections([]); // 清空当前集合列表
      fetchCollections(); // 获取新的集合列表
    }
  }, [selectedProvider]);

  const fetchEmbeddedFiles = async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/list-embedded`);
      const data = await response.json();
      if (data.documents) {
        setEmbeddedFiles(data.documents.map(doc => ({
          ...doc,
          id: doc.name,
          displayName: doc.name
        })));
      }
    } catch (error) {
      console.error('Error fetching embedded files:', error);
      setStatus('Error loading embedding files');
    }
  };

  const fetchCollections = async () => {
    try {
      setStatus('Loading collections...');
      const response = await fetch(`${apiBaseUrl}/collections?provider=${selectedProvider}`);
      const data = await response.json();
      if (data.collections) {
        setCollections(data.collections);
        setStatus('');
      } else {
        setCollections([]);
        setStatus('No collections found');
      }
    } catch (error) {
      console.error('Error fetching collections:', error);
      setStatus('Error loading collections');
      setCollections([]);
    }
  };

  const handleIndex = async () => {
    if (!embeddingFile) {
      setStatus('Please select an embedding file');
      return;
    }

    setStatus('Indexing...');
    try {
      const response = await fetch(`${apiBaseUrl}/index`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          fileId: embeddingFile,
          vectorDb,
          indexMode
        }),
      });

      const data = await response.json();
      setIndexingResult(data);
      setStatus('Indexing completed successfully');

      // 自动刷新集合列表
      await fetchCollections();

      // 如果索引成功，自动选中新创建的集合
      if (data.collection_name) {
        setSelectedCollection(data.collection_name);
        handleDisplay(data.collection_name);
      }
    } catch (error) {
      console.error('Error indexing:', error);
      setStatus('Error during indexing: ' + error.message);
    }
  };

  const handleDisplay = async (collectionName) => {
    if (!collectionName) return;

    try {
      setStatus('Loading collection details...');
      const response = await fetch(`${apiBaseUrl}/collections/${selectedProvider}/${collectionName}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch collection details: ${response.statusText}`);
      }

      const data = await response.json();

      // 构建结果对象
      const result = {
        database: selectedProvider,
        collection_name: data.name,
        total_vectors: data.num_entities,
        index_size: data.num_entities
      };

      // 获取索引类型
      if (data.schema?.fields) {
        const embeddingField = data.schema.fields.find(f => f.name === 'embedding' || f.name === 'vector');
        if (embeddingField) {
          result.index_mode = embeddingField.index_type || 'default';
          if (embeddingField.params?.space_type) {
            result.space_type = embeddingField.params.space_type;
          }
        }
      }

      setIndexingResult(result);
      setStatus('');
    } catch (error) {
      console.error('Error displaying collection:', error);
      setStatus(`Error loading collection details: ${error.message}`);
      setIndexingResult(null);
    }
  };

  const handleDelete = async (collectionName) => {
    if (!collectionName) return;

    if (window.confirm(`Are you sure you want to delete collection "${collectionName}"?`)) {
      try {
        await fetch(`${apiBaseUrl}/collections/${selectedProvider}/${collectionName}`, {
          method: 'DELETE',
        });
        setSelectedCollection('');
        // 重新获取collections列表
        const response = await fetch(`${apiBaseUrl}/collections?provider=${selectedProvider}`);
        const data = await response.json();
        setCollections(data.collections);
      } catch (error) {
        console.error('Error deleting collection:', error);
      }
    }
  };

  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-6">Vector Database Indexing</h2>

      <div className="grid grid-cols-12 gap-6">
        {/* Left Panel - Controls */}
        <div className="col-span-3">
          <div className="p-4 border rounded-lg bg-white shadow-sm space-y-4">
            {/* Embedding File Selection */}
            <div>
              <label className="block text-sm font-medium mb-1">Embedding File</label>
              <select
                value={embeddingFile}
                onChange={(e) => setEmbeddingFile(e.target.value)}
                className="block w-full p-2 border rounded"
              >
                <option value="">Choose a file...</option>
                {embeddedFiles.map(file => (
                  <option key={file.name} value={file.name}>
                    {file.displayName}
                  </option>
                ))}
              </select>
            </div>

            {/* Vector Database Selection */}
            <div>
              <label className="block text-sm font-medium mb-1">Vector Database</label>
              <select
                value={selectedProvider}
                onChange={(e) => setSelectedProvider(e.target.value)}
                className="block w-full p-2 border rounded"
              >
                {providers.map(provider => (
                  <option key={provider.id} value={provider.id}>
                    {provider.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Index Mode Selection */}
            <div>
              <label className="block text-sm font-medium mb-1">Index Mode</label>
              <select
                value={indexMode}
                onChange={(e) => setIndexMode(e.target.value)}
                className="block w-full p-2 border rounded"
              >
                {dbConfigs[selectedProvider].modes.map(mode => (
                  <option key={mode} value={mode}>
                    {mode.toUpperCase()}
                  </option>
                ))}
              </select>
            </div>

            {/* Action Buttons and Collection Management */}
            <div className="space-y-2">
              {/* Index Data Button */}
              <button
                onClick={handleIndex}
                className="w-full px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-blue-300"
                disabled={!embeddingFile}
              >
                Index Data
              </button>

              {/* Collection Selection */}
              <div>
                <label className="block text-sm font-medium mb-1">Collection</label>
                <select
                  value={selectedCollection}
                  onChange={(e) => {
                    setSelectedCollection(e.target.value);
                    handleDisplay(e.target.value);
                  }}
                  className="block w-full p-2 border rounded"
                >
                  <option value="">Choose a collection...</option>
                  {collections.map(coll => (
                    <option key={coll.id} value={coll.name}>
                      {coll.name} ({coll.count} documents)
                    </option>
                  ))}
                </select>
              </div>

              {/* Display Collection Button */}
              <button
                onClick={() => handleDisplay(selectedCollection)}
                disabled={!selectedCollection}
                className="w-full px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-blue-300"
              >
                Display Collection
              </button>

              {/* Delete Collection Button */}
              <button
                onClick={() => handleDelete(selectedCollection)}
                disabled={!selectedCollection}
                className="w-full px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600 disabled:bg-red-300"
              >
                Delete Collection
              </button>
            </div>

            {status && (
              <div className="mt-4 p-3 rounded border bg-gray-50">
                <p className="text-sm">{status}</p>
              </div>
            )}
          </div>
        </div>

        {/* Right Panel - Results */}
        <div className="col-span-9 border rounded-lg bg-white shadow-sm">
          {indexingResult ? (
            <div className="p-4">
              <h3 className="text-xl font-semibold mb-4">Indexing Results</h3>
              <div className="space-y-3">
                <div className="p-3 border rounded bg-gray-50">
                  <div className="text-sm text-gray-600">
                    <p>Database: {indexingResult.database}</p>
                    {indexingResult.index_mode && (
                      <p>Index Mode: {indexingResult.index_mode}</p>
                    )}
                    <p>Total Vectors: {indexingResult.total_vectors}</p>
                    <p>Index Size: {indexingResult.index_size}</p>
                    {indexingResult.space_type && (
                      <p>Space Type: {indexingResult.space_type}</p>
                    )}
                    <p>Collection Name: {indexingResult.collection_name}</p>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <RandomImage message="Indexing results will appear here" />
          )}
        </div>
      </div>
    </div>
  );
};

export default Indexing;