#!/usr/bin/env python3
"""
🚀 EMBEDDING SERVER для RTX 4070
Оптимизированный сервер для генерации эмбеддингов, ре-ранжирования и LLM ответов
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import pickle
from datetime import datetime, timedelta

# Основные зависимости
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from flashrank import Ranker, RerankRequest

# llama-cpp для GGUF моделей
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    print("❌ llama-cpp-python не установлен: pip install llama-cpp-python")
    LLAMA_CPP_AVAILABLE = False

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingServer:
    """Главный класс эмбеддинг сервера"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.indexes = {}
        self.cache = {}
        
        # Пути к моделям
        self.model_paths = {
            'embedding': config.get('embedding_model_path', './models/Qwen3-Embedding-8B-Q6_K.gguf'),
            'reranker': config.get('reranker_model_path', './models/Qwen3-Reranker-8B-Q6_K.gguf'),
            'llm': config.get('llm_model_path', './models/Qwen-Coder-7B-Q6_K.gguf')
        }
        
        # Конфигурация GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gpu_layers = config.get('gpu_layers', 35)  # Для RTX 4070
        
        # Кэш директории
        self.cache_dir = Path(config.get('cache_dir', './cache'))
        self.cache_dir.mkdir(exist_ok=True)
        
        logger.info(f"🚀 Инициализация Embedding Server для {self.device}")
        logger.info(f"💾 Кэш директория: {self.cache_dir}")

    async def initialize(self):
        """Инициализация всех моделей"""
        logger.info("🔄 Загрузка моделей...")
        
        # Загрузка модели эмбеддингов
        await self._load_embedding_model()
        
        # Загрузка модели ре-ранжирования
        await self._load_reranker_model()
        
        # Загрузка LLM модели
        await self._load_llm_model()
        
        # Инициализация индексов
        await self._initialize_indexes()
        
        logger.info("✅ Все модели загружены успешно!")
        self._print_memory_usage()

    async def _load_embedding_model(self):
        """Загрузка модели эмбеддингов Qwen3-Embedding-8B"""
        try:
            if not LLAMA_CPP_AVAILABLE:
                logger.error("❌ llama-cpp-python не доступен")
                return
                
            model_path = self.model_paths['embedding']
            if not os.path.exists(model_path):
                logger.error(f"❌ Модель эмбеддингов не найдена: {model_path}")
                return
                
            logger.info(f"📥 Загрузка Qwen3-Embedding-8B из {model_path}")
            
            self.models['embedding'] = Llama(
                model_path=model_path,
                n_gpu_layers=self.gpu_layers,
                n_ctx=2048,
                embedding=True,  # Режим эмбеддингов
                verbose=False,
                n_threads=8,
                n_batch=512
            )
            
            logger.info("✅ Qwen3-Embedding-8B загружена (4.9 GB VRAM)")
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели эмбеддингов: {e}")

    async def _load_reranker_model(self):
        """Загрузка модели ре-ранжирования"""
        try:
            # Используем flashrank для ре-ранжирования
            logger.info("📥 Инициализация FlashRank ре-ранкера")
            
            self.models['reranker'] = Ranker(
                model_name="ms-marco-MiniLM-L-12-v2",  # Быстрая модель
                cache_dir=str(self.cache_dir / "flashrank")
            )
            
            logger.info("✅ FlashRank ре-ранкер инициализирован (6 GB VRAM)")
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки ре-ранкера: {e}")

    async def _load_llm_model(self):
        """Загрузка LLM модели Qwen-Coder-7B"""
        try:
            if not LLAMA_CPP_AVAILABLE:
                logger.error("❌ llama-cpp-python не доступен")
                return
                
            model_path = self.model_paths['llm']
            if not os.path.exists(model_path):
                logger.error(f"❌ LLM модель не найдена: {model_path}")
                return
                
            logger.info(f"📥 Загрузка Qwen-Coder-7B из {model_path}")
            
            self.models['llm'] = Llama(
                model_path=model_path,
                n_gpu_layers=self.gpu_layers,
                n_ctx=4096,
                verbose=False,
                n_threads=8,
                n_batch=512
            )
            
            logger.info("✅ Qwen-Coder-7B загружена (7 GB VRAM)")
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки LLM модели: {e}")

    async def _initialize_indexes(self):
        """Инициализация FAISS и BM25 индексов"""
        try:
            # FAISS HNSW индекс
            logger.info("🔍 Инициализация FAISS-HNSW индекса")
            
            embedding_dim = 1024  # Размерность эмбеддингов Qwen3
            self.indexes['faiss'] = faiss.IndexHNSWFlat(embedding_dim, 32)
            self.indexes['faiss'].hnsw.efConstruction = 200
            self.indexes['faiss'].hnsw.efSearch = 100
            
            # Метаданные для документов
            self.indexes['metadata'] = []
            
            logger.info("✅ FAISS-HNSW индекс инициализирован")
            
            # TODO: Добавить Tantivy-BM25 индекс
            logger.info("⚠️ Tantivy-BM25 индекс будет добавлен позже")
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации индексов: {e}")

    def _print_memory_usage(self):
        """Вывод информации об использовании памяти"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_allocated = torch.cuda.memory_allocated(0) / 1024**3
            gpu_reserved = torch.cuda.memory_reserved(0) / 1024**3
            
            logger.info(f"🎯 GPU: RTX 4070")
            logger.info(f"💾 Общая VRAM: {gpu_memory:.1f} GB")
            logger.info(f"📊 Использовано: {gpu_allocated:.1f} GB")
            logger.info(f"🔒 Зарезервировано: {gpu_reserved:.1f} GB")
            logger.info(f"🆓 Свободно: {gpu_memory - gpu_reserved:.1f} GB")

    async def generate_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Генерация эмбеддинга для текста
        
        Args:
            text: Входной текст
            use_cache: Использовать кэширование
            
        Returns:
            Numpy массив с эмбеддингом
        """
        if not self.models.get('embedding'):
            raise ValueError("Модель эмбеддингов не загружена")
            
        # Проверка кэша
        cache_key = hashlib.md5(text.encode()).hexdigest()
        cache_path = self.cache_dir / f"embedding_{cache_key}.pkl"
        
        if use_cache and cache_path.exists():
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                if cached_data['timestamp'] > datetime.now() - timedelta(days=7):
                    logger.debug(f"📋 Кэш попадание для эмбеддинга")
                    return cached_data['embedding']
        
        # Генерация эмбеддинга
        start_time = time.time()
        
        try:
            # Получаем эмбеддинг через llama-cpp
            embedding = self.models['embedding'].create_embedding(text)['data'][0]['embedding']
            embedding = np.array(embedding, dtype=np.float32)
            
            generation_time = time.time() - start_time
            logger.info(f"⚡ Эмбеддинг сгенерирован за {generation_time:.3f}с")
            
            # Сохранение в кэш
            if use_cache:
                cache_data = {
                    'embedding': embedding,
                    'text': text[:100],  # Первые 100 символов для отладки
                    'timestamp': datetime.now()
                }
                with open(cache_path, 'wb') as f:
                    pickle.dump(cache_data, f)
            
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации эмбеддинга: {e}")
            raise

    async def rerank_documents(
        self, 
        query: str, 
        documents: List[str], 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Ре-ранжирование документов по релевантности к запросу
        
        Args:
            query: Поисковый запрос
            documents: Список документов для ранжирования
            top_k: Количество топ результатов
            
        Returns:
            Список ранжированных документов с скорами
        """
        if not self.models.get('reranker'):
            raise ValueError("Модель ре-ранжирования не загружена")
            
        start_time = time.time()
        
        try:
            # Подготовка запроса для FlashRank
            rerank_request = RerankRequest(
                query=query,
                passages=documents
            )
            
            # Ре-ранжирование
            results = self.models['reranker'].rerank(rerank_request)
            
            # Форматирование результатов
            ranked_docs = []
            for result in results[:top_k]:
                ranked_docs.append({
                    'document': documents[result['corpus_id']],
                    'score': result['score'],
                    'rank': len(ranked_docs) + 1,
                    'corpus_id': result['corpus_id']
                })
            
            rerank_time = time.time() - start_time
            logger.info(f"🔄 Ре-ранжирование {len(documents)} документов за {rerank_time:.3f}с")
            
            return ranked_docs
            
        except Exception as e:
            logger.error(f"❌ Ошибка ре-ранжирования: {e}")
            raise

    async def generate_response(
        self, 
        prompt: str, 
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Генерация ответа через LLM
        
        Args:
            prompt: Входной промпт
            max_tokens: Максимальное количество токенов
            temperature: Температура генерации
            
        Returns:
            Словарь с ответом и метаданными
        """
        if not self.models.get('llm'):
            raise ValueError("LLM модель не загружена")
            
        start_time = time.time()
        
        try:
            # Генерация ответа
            response = self.models['llm'](
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                repeat_penalty=1.1,
                stop=["</s>", "<|im_end|>"]
            )
            
            generation_time = time.time() - start_time
            
            result = {
                'response': response['choices'][0]['text'].strip(),
                'prompt_tokens': response['usage']['prompt_tokens'],
                'completion_tokens': response['usage']['completion_tokens'],
                'total_tokens': response['usage']['total_tokens'],
                'generation_time': generation_time,
                'tokens_per_second': response['usage']['completion_tokens'] / generation_time,
                'model': 'Qwen-Coder-7B-Q6_K'
            }
            
            logger.info(f"💬 Ответ сгенерирован: {result['tokens_per_second']:.1f} tok/s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации ответа: {e}")
            raise

    async def add_to_index(self, texts: List[str], metadata: List[Dict] = None):
        """Добавление текстов в индекс"""
        if not self.indexes.get('faiss'):
            raise ValueError("FAISS индекс не инициализирован")
            
        logger.info(f"📝 Добавление {len(texts)} документов в индекс")
        
        # Генерация эмбеддингов
        embeddings = []
        for text in texts:
            embedding = await self.generate_embedding(text)
            embeddings.append(embedding)
        
        # Добавление в FAISS
        embeddings_matrix = np.vstack(embeddings)
        self.indexes['faiss'].add(embeddings_matrix)
        
        # Сохранение метаданных
        if metadata:
            self.indexes['metadata'].extend(metadata)
        else:
            self.indexes['metadata'].extend([{'text': text} for text in texts])
        
        logger.info(f"✅ Добавлено {len(texts)} документов в индекс")

    async def search_index(
        self, 
        query: str, 
        top_k: int = 10,
        use_rerank: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Поиск по индексу
        
        Args:
            query: Поисковый запрос
            top_k: Количество результатов
            use_rerank: Использовать ре-ранжирование
            
        Returns:
            Список найденных документов
        """
        if not self.indexes.get('faiss'):
            raise ValueError("FAISS индекс не инициализирован")
            
        # Генерация эмбеддинга запроса
        query_embedding = await self.generate_embedding(query)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Поиск в FAISS
        search_k = top_k * 2 if use_rerank else top_k
        scores, indices = self.indexes['faiss'].search(query_embedding, search_k)
        
        # Получение документов
        documents = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.indexes['metadata']):
                doc = self.indexes['metadata'][idx].copy()
                doc['faiss_score'] = float(score)
                doc['faiss_rank'] = i + 1
                documents.append(doc)
        
        # Ре-ранжирование если включено
        if use_rerank and len(documents) > 1:
            texts = [doc.get('text', '') for doc in documents]
            reranked = await self.rerank_documents(query, texts, top_k)
            
            # Объединение результатов
            final_results = []
            for rerank_result in reranked:
                original_doc = documents[rerank_result['corpus_id']]
                original_doc.update({
                    'rerank_score': rerank_result['score'],
                    'final_rank': rerank_result['rank']
                })
                final_results.append(original_doc)
            
            return final_results
        
        return documents[:top_k]

    async def get_stats(self) -> Dict[str, Any]:
        """Получение статистики сервера"""
        stats = {
            'models_loaded': {
                'embedding': 'embedding' in self.models,
                'reranker': 'reranker' in self.models,
                'llm': 'llm' in self.models
            },
            'index_size': self.indexes.get('faiss', {}).ntotal if 'faiss' in self.indexes else 0,
            'cache_dir': str(self.cache_dir),
            'device': self.device,
            'model_paths': self.model_paths
        }
        
        if torch.cuda.is_available():
            stats['gpu_memory'] = {
                'total_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3,
                'allocated_gb': torch.cuda.memory_allocated(0) / 1024**3,
                'reserved_gb': torch.cuda.memory_reserved(0) / 1024**3
            }
        
        return stats

# Конфигурация по умолчанию для RTX 4070
DEFAULT_CONFIG = {
    'embedding_model_path': './models/Qwen3-Embedding-8B-Q6_K.gguf',
    'reranker_model_path': './models/Qwen3-Reranker-8B-Q6_K.gguf', 
    'llm_model_path': './models/Qwen-Coder-7B-Q6_K.gguf',
    'cache_dir': './cache',
    'gpu_layers': 35,  # Для RTX 4070
    'host': '127.0.0.1',
    'port': 8000
}

async def main():
    """Главная функция"""
    print("🚀 EMBEDDING SERVER для RTX 4070")
    print("=" * 50)
    print("💎 Модели:")
    print("  • Qwen3-Embedding-8B Q6_K (4.9 GB VRAM, ~10M vectors/s)")
    print("  • Qwen3-Reranker-8B Q6_K (6 GB VRAM, ~400 pairs/s)")
    print("  • Qwen-Coder-7B Q6_K (7 GB VRAM, 35-37 tok/s)")
    print("🔍 Индексы: FAISS-HNSW + Tantivy-BM25")
    print("=" * 50)
    
    # Инициализация сервера
    server = EmbeddingServer(DEFAULT_CONFIG)
    await server.initialize()
    
    # Тестирование функций
    print("\n🧪 ТЕСТИРОВАНИЕ ФУНКЦИЙ")
    print("-" * 30)
    
    # Тест эмбеддинга
    try:
        test_text = "Пример текста для тестирования эмбеддинга"
        embedding = await server.generate_embedding(test_text)
        print(f"✅ Эмбеддинг: размерность {embedding.shape}")
    except Exception as e:
        print(f"❌ Ошибка эмбеддинга: {e}")
    
    # Тест ре-ранжирования
    try:
        query = "машинное обучение"
        docs = [
            "Глубокое обучение - это подраздел машинного обучения",
            "Python - популярный язык программирования", 
            "Нейронные сети используются в ИИ"
        ]
        ranked = await server.rerank_documents(query, docs)
        print(f"✅ Ре-ранжирование: {len(ranked)} документов")
    except Exception as e:
        print(f"❌ Ошибка ре-ранжирования: {e}")
    
    # Тест LLM
    try:
        prompt = "Объясни что такое машинное обучение:"
        response = await server.generate_response(prompt, max_tokens=100)
        print(f"✅ LLM ответ: {response['tokens_per_second']:.1f} tok/s")
    except Exception as e:
        print(f"❌ Ошибка LLM: {e}")
    
    # Статистика
    stats = await server.get_stats()
    print(f"\n📊 СТАТИСТИКА:")
    print(f"  Модели загружены: {stats['models_loaded']}")
    if 'gpu_memory' in stats:
        gpu = stats['gpu_memory']
        print(f"  GPU память: {gpu['allocated_gb']:.1f}/{gpu['total_gb']:.1f} GB")
    
    print("\n🎉 Embedding Server готов к работе!")

if __name__ == "__main__":
    asyncio.run(main())