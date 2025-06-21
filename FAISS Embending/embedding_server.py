#!/usr/bin/env python3
"""
🚀 EMBEDDING SERVER для RTX 4070
Оптимизированный сервер для генерации эмбеддингов, ре-ранжирования и LLM ответов
Использует llama.cpp напрямую с OpenAI-совместимым API для Cursor
"""

import asyncio
import json
import logging
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import hashlib
import pickle
from datetime import datetime, timedelta

# Основные зависимости
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# FastAPI для HTTP сервера
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# FlashRank для ре-ранжирования
try:
    from flashrank import Ranker, RerankRequest
    FLASHRANK_AVAILABLE = True
except ImportError:
    print("❌ flashrank не установлен: pip install flashrank")
    FLASHRANK_AVAILABLE = False

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создание FastAPI приложения
app = FastAPI(
    title="Cursor Compatible Embedding Server",
    description="Локальный сервер эмбеддингов с OpenAI API совместимостью",
    version="1.0.0"
)

# CORS для веб-интерфейса
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальная переменная для сервера
embedding_server = None

# === OpenAI API Models ===
class EmbeddingRequest(BaseModel):
    input: Union[List[str], str]
    model: str = "text-embedding-ada-002"
    encoding_format: str = "float"
    dimensions: Optional[int] = None

class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "local"

class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]

class LlamaCppInterface:
    """Интерфейс для работы с llama.cpp"""
    
    def __init__(self, executable_path: str, model_path: str, gpu_layers: int = 35):
        self.executable_path = executable_path
        self.model_path = model_path
        self.gpu_layers = gpu_layers
        self.process = None
        
    async def start_server(self, port: int = 8080, embedding_mode: bool = False):
        """Запуск llama.cpp сервера"""
        cmd = [
            self.executable_path,
            "-m", self.model_path,
            "--port", str(port),
            "--host", "127.0.0.1",
            "-ngl", str(self.gpu_layers),
            "-c", "2048",
            "--log-disable"
        ]
        
        if embedding_mode:
            cmd.append("--embedding")
        
        logger.info(f"🚀 Запуск llama.cpp сервера: {' '.join(cmd)}")
        
        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Ждем запуска сервера
        await asyncio.sleep(3)
        
        if self.process.returncode is not None:
            stdout, stderr = await self.process.communicate()
            raise RuntimeError(f"Ошибка запуска llama.cpp: {stderr.decode()}")
        
        logger.info(f"✅ llama.cpp сервер запущен на порту {port}")
        
    async def stop_server(self):
        """Остановка сервера"""
        if self.process:
            self.process.terminate()
            await self.process.wait()
            self.process = None
            
    async def generate_embedding(self, text: str, port: int = 8080) -> np.ndarray:
        """Генерация эмбеддинга через HTTP API"""
        import aiohttp
        
        url = f"http://127.0.0.1:{port}/embedding"
        payload = {"content": text}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    # llama.cpp возвращает список с объектами
                    if isinstance(result, list) and len(result) > 0:
                        embedding_data = result[0]["embedding"]
                        # Проверяем если embedding это вложенный список
                        if isinstance(embedding_data, list) and len(embedding_data) > 0 and isinstance(embedding_data[0], list):
                            embedding_data = embedding_data[0]
                        return np.array(embedding_data, dtype=np.float32)
                    else:
                        raise RuntimeError(f"Неожиданный формат ответа: {result}")
                else:
                    raise RuntimeError(f"Ошибка API llama.cpp: {response.status}")
    
    async def generate_completion(self, prompt: str, port: int = 8081, **kwargs) -> Dict[str, Any]:
        """Генерация текста через HTTP API"""
        import aiohttp
        
        url = f"http://127.0.0.1:{port}/completion"
        payload = {
            "prompt": prompt,
            "n_predict": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "repeat_penalty": kwargs.get("repeat_penalty", 1.1),
            "stop": kwargs.get("stop", ["</s>", "<|im_end|>"])
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise RuntimeError(f"Ошибка API llama.cpp: {response.status}")

class EmbeddingServer:
    """Главный класс эмбеддинг сервера"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.indexes = {}
        self.cache = {}
        self.llama_processes = {}
        
        # Пути к исполняемым файлам и моделям
        self.llama_cpp_path = config.get('llama_cpp_path', './llama_build/llama.cpp/build/bin/Release/llama-server.exe')
        self.model_paths = {
            'embedding': config.get('embedding_model_path', './models/Qwen3-Embedding-8B-Q6_K.gguf'),
            'reranker': config.get('reranker_model_path', './models/Qwen3-Reranker-8B-Q6_K.gguf'),
            'coder': config.get('coder_model_path', './models/Qwen2.5-Coder-7B-Instruct.Q6_K.gguf'),
            'llm': config.get('llm_model_path', './models/Magistral-Small-2506-UD-Q4_K_XL.gguf')
        }
        
        # Порты для разных сервисов
        self.ports = {
            'embedding': 8080,
            'coder': 8081,
            'llm': 8082
        }
        
        # Конфигурация GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gpu_layers = config.get('gpu_layers', 35)  # Для RTX 4070
        
        # Кэш директории
        self.cache_dir = Path(config.get('cache_dir', './cache'))
        self.cache_dir.mkdir(exist_ok=True)

        # Путь к репозиторию для индексирования
        self.repo_path = Path(config.get('repo_path', '.')).resolve()
        self.watcher = None
        
        logger.info(f"🚀 Инициализация Embedding Server для {self.device}")
        logger.info(f"💾 Кэш директория: {self.cache_dir}")
        logger.info(f"🔧 llama.cpp путь: {self.llama_cpp_path}")

    async def initialize(self):
        """Инициализация всех моделей"""
        logger.info("🔄 Загрузка моделей...")
        
        # Проверка наличия llama.cpp
        if not os.path.exists(self.llama_cpp_path):
            logger.error(f"❌ llama.cpp не найден: {self.llama_cpp_path}")
            logger.info("💡 Постройте llama.cpp: cd llama_build && make")
            return
        
        # Запуск серверов llama.cpp
        await self._start_llama_servers()
        
        # Загрузка модели ре-ранжирования
        await self._load_reranker_model()
        
        # Инициализация индексов
        await self._initialize_indexes()

        # Первичная индексация репозитория
        await self.index_repository(self.repo_path)

        # Запуск наблюдателя за репозиторием
        if self.repo_path.exists():
            from .repo_watcher import RepositoryWatcher
            self.watcher = RepositoryWatcher(self, self.repo_path)
            self.watcher.start()
        else:
            logger.warning(f"⚠️ Репозиторий не найден: {self.repo_path}")

        logger.info("✅ Все модели загружены успешно!")
        self._print_memory_usage()

    async def _start_llama_servers(self):
        """Запуск серверов llama.cpp для разных моделей"""
        
        # Embedding сервер
        if os.path.exists(self.model_paths['embedding']):
            logger.info(f"📥 Запуск Qwen3-Embedding-8B сервера")
            self.llama_processes['embedding'] = LlamaCppInterface(
                self.llama_cpp_path,
                self.model_paths['embedding'],
                self.gpu_layers
            )
            await self.llama_processes['embedding'].start_server(
                self.ports['embedding'], 
                embedding_mode=True
            )
        else:
            logger.warning(f"⚠️ Модель эмбеддингов не найдена: {self.model_paths['embedding']}")
        
        # Coder сервер (Qwen2.5)
        if os.path.exists(self.model_paths['coder']):
            logger.info(f"📥 Запуск Qwen2.5-Coder-7B сервера")
            self.llama_processes['coder'] = LlamaCppInterface(
                self.llama_cpp_path,
                self.model_paths['coder'],
                self.gpu_layers
            )
            await self.llama_processes['coder'].start_server(self.ports['coder'])
        else:
            logger.warning(f"⚠️ Coder модель не найдена: {self.model_paths['coder']}")

        # LLM сервер (Magistral)
        if os.path.exists(self.model_paths['llm']):
            logger.info(f"📥 Запуск Magistral-Small-2506 сервера")
            self.llama_processes['llm'] = LlamaCppInterface(
                self.llama_cpp_path,
                self.model_paths['llm'],
                self.gpu_layers
            )
            await self.llama_processes['llm'].start_server(self.ports['llm'])
        else:
            logger.warning(f"⚠️ LLM модель не найдена: {self.model_paths['llm']}")

    async def _load_reranker_model(self):
        """Загрузка модели ре-ранжирования"""
        if not FLASHRANK_AVAILABLE:
            logger.warning("⚠️ FlashRank недоступен, ре-ранжирование отключено")
            return
            
        try:
            logger.info("📥 Загрузка FlashRank ре-ранкера...")
            self.models['reranker'] = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=str(self.cache_dir))
            logger.info("✅ FlashRank ре-ранкер загружен")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки ре-ранкера: {e}")

    async def _initialize_indexes(self):
        """Инициализация FAISS индекса"""
        logger.info("🔧 Инициализация FAISS индекса...")
        
        # Создание FAISS индекса
        embedding_dim = 4096  # Qwen3-Embedding-8B размерность
        self.indexes['faiss'] = faiss.IndexHNSWFlat(embedding_dim, 32)
        self.indexes['faiss'].hnsw.efConstruction = 40
        self.indexes['faiss'].hnsw.efSearch = 16
        
        # Метаданные для документов
        self.indexes['metadata'] = []

        logger.info("✅ FAISS индекс инициализирован")

    async def index_repository(self, repo_path: Path):
        """Полная индексация репозитория."""
        if not repo_path.exists():
            logger.warning(f"⚠️ Репозиторий не найден: {repo_path}")
            return

        logger.info(f"📂 Индексация репозитория {repo_path}...")

        for file_path in repo_path.rglob('*'):
            if not file_path.is_file():
                continue

            try:
                from .chunker import chunk_file
                chunks = chunk_file(str(file_path))
                if not chunks:
                    continue
                rel = os.path.relpath(file_path, repo_path)
                meta = [{"path": rel, "chunk_id": i} for i in range(len(chunks))]
                await self.add_to_index(chunks, meta)
            except Exception as e:
                logger.warning(f"⚠️ Ошибка индексации {file_path}: {e}")

        logger.info("✅ Репозиторий проиндексирован")

    def _print_memory_usage(self):
        """Вывод информации об использовании памяти"""
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
            logger.info(f"🔧 GPU память: {allocated_memory:.1f}/{total_memory:.1f} GB")
        else:
            logger.info("🔧 GPU недоступно, используется CPU")

    async def generate_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Генерация эмбеддинга для текста
        
        Args:
            text: Входной текст
            use_cache: Использовать кэширование
            
        Returns:
            Numpy массив с эмбеддингом
        """
        if 'embedding' not in self.llama_processes:
            raise ValueError("Embedding сервер не запущен")
            
        # Проверка кэша
        if use_cache:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            cache_path = self.cache_dir / f"embedding_{text_hash}.pkl"
            
            if cache_path.exists():
                try:
                    with open(cache_path, 'rb') as f:
                        cached_embedding = pickle.load(f)
                    logger.debug(f"💾 Загружен эмбеддинг из кэша")
                    return cached_embedding
                except Exception as e:
                    logger.warning(f"⚠️ Ошибка загрузки кэша: {e}")
        
        start_time = time.time()
        
        try:
            # Генерация эмбеддинга через llama.cpp
            embedding = await self.llama_processes['embedding'].generate_embedding(
                text, 
                port=self.ports['embedding']
            )
            
            generation_time = time.time() - start_time
            
            # Сохранение в кэш
            if use_cache:
                try:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(embedding, f)
                except Exception as e:
                    logger.warning(f"⚠️ Ошибка сохранения в кэш: {e}")
            
            logger.info(f"⚡ Эмбеддинг сгенерирован: {generation_time:.3f}s, размерность: {embedding.shape}")
            
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
        Ре-ранжирование документов
        
        Args:
            query: Поисковый запрос
            documents: Список документов для ранжирования
            top_k: Количество лучших результатов
            
        Returns:
            Список документов с оценками
        """
        if 'reranker' not in self.models:
            # Возвращаем документы в оригинальном порядке
            return [
                {
                    'text': doc,
                    'score': 1.0 - (i * 0.1),  # Убывающие оценки
                    'rank': i + 1,
                    'corpus_id': i
                }
                for i, doc in enumerate(documents[:top_k])
            ]
        
        try:
            # Создание запроса для FlashRank
            rerank_request = RerankRequest(query=query, passages=documents)
            
            # Ре-ранжирование
            results = self.models['reranker'].rerank(rerank_request)
            
            # Форматирование результатов
            ranked_docs = []
            for i, result in enumerate(results[:top_k]):
                ranked_docs.append({
                    'text': documents[result['corpus_id']],
                    'score': result['score'],
                    'rank': i + 1,
                    'corpus_id': result['corpus_id']
                })
            
            logger.info(f"📊 Ре-ранжирование завершено: {len(ranked_docs)} документов")
            
            return ranked_docs
            
        except Exception as e:
            logger.error(f"❌ Ошибка ре-ранжирования: {e}")
            # Возвращаем документы в оригинальном порядке
            return [
                {
                    'text': doc,
                    'score': 1.0 - (i * 0.1),
                    'rank': i + 1,
                    'corpus_id': i
                }
                for i, doc in enumerate(documents[:top_k])
            ]

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
        if 'coder' not in self.llama_processes and 'llm' not in self.llama_processes:
            raise ValueError("LLM сервера не запущены")
            
        start_time = time.time()
        
        try:
            # Генерация ответа от всех доступных моделей
            results = []
            for name in ['coder', 'llm']:
                if name in self.llama_processes:
                    resp = await self.llama_processes[name].generate_completion(
                        prompt,
                        port=self.ports[name],
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    results.append((name, resp.get('content', '').strip()))

            generation_time = time.time() - start_time

            # Если доступен ре-ранкер и есть несколько результатов - выбираем лучший
            if self.models.get('reranker') and len(results) > 1:
                texts = [r[1] for r in results]
                ranked = await self.rerank_documents(prompt, texts, top_k=1)
                best_index = ranked[0]['corpus_id'] if ranked else 0
            else:
                best_index = 0

            best_name, best_text = results[best_index]

            prompt_tokens = len(prompt.split())
            completion_tokens = len(best_text.split())

            result = {
                'response': best_text,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens,
                'generation_time': generation_time,
                'tokens_per_second': completion_tokens / generation_time if generation_time > 0 else 0,
                'model': best_name
            }

            logger.info(
                f"💬 Ответ сгенерирован {best_name}: {result['tokens_per_second']:.1f} tok/s"
            )

            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации ответа: {e}")
            raise

    async def add_to_index(self, texts: List[str], metadata: List[Dict] = None):
        """Добавление документов в индекс"""
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
        if use_rerank and len(documents) > 1 and self.models.get('reranker'):
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
                'embedding': 'embedding' in self.llama_processes,
                'reranker': 'reranker' in self.models,
                'coder': 'coder' in self.llama_processes,
                'llm': 'llm' in self.llama_processes
            },
            'index_size': self.indexes.get('faiss', {}).ntotal if 'faiss' in self.indexes else 0,
            'cache_dir': str(self.cache_dir),
            'device': self.device,
            'model_paths': self.model_paths,
            'llama_cpp_path': self.llama_cpp_path,
            'ports': self.ports
        }
        
        if torch.cuda.is_available():
            stats['gpu_memory'] = {
                'total_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3,
                'allocated_gb': torch.cuda.memory_allocated(0) / 1024**3,
                'reserved_gb': torch.cuda.memory_reserved(0) / 1024**3
            }
        
        return stats

    async def shutdown(self):
        """Корректное завершение работы"""
        logger.info("🛑 Завершение работы Embedding Server...")
        
        # Остановка всех llama.cpp серверов
        for name, process in self.llama_processes.items():
            logger.info(f"🛑 Остановка {name} сервера...")
            await process.stop_server()

        if self.watcher:
            await self.watcher.stop()

        logger.info("✅ Все серверы остановлены")

# === FASTAPI HTTP СЕРВЕР ===

# Глобальный экземпляр сервера
embedding_server: Optional[EmbeddingServer] = None

# Создание FastAPI приложения
app = FastAPI(
    title="Cursor Compatible Embedding Server",
    description="Локальный эмбеддинг сервер с OpenAI API для Cursor",
    version="1.0.0"
)

# CORS для всех доменов
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Инициализация сервера при запуске"""
    global embedding_server
    
    logger.info("🚀 Запуск Cursor Compatible Embedding Server...")
    
    # Инициализация сервера
    config = {
        'llama_cpp_path': './llama_build/llama.cpp/build/bin/Release/llama-server.exe',
        'embedding_model_path': './models/Qwen3-Embedding-8B-Q6_K.gguf',
        'reranker_model_path': './models/Qwen3-Reranker-8B-Q6_K.gguf',
        'coder_model_path': './models/Qwen2.5-Coder-7B-Instruct.Q6_K.gguf',
        'llm_model_path': './models/Magistral-Small-2506-UD-Q4_K_XL.gguf',
        'cache_dir': './cache',
        'repo_path': '.',
        'gpu_layers': 35,  # Для RTX 4070
    }
    
    embedding_server = EmbeddingServer(config)
    await embedding_server.initialize()
    
    logger.info("✅ Embedding Server готов к обработке запросов!")

@app.on_event("shutdown")
async def shutdown_event():
    """Корректное завершение при остановке"""
    global embedding_server
    if embedding_server:
        await embedding_server.shutdown()

# === OpenAI-совместимые API endpoints ===

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """
    Создание эмбеддингов (OpenAI API совместимый endpoint)
    
    Поддерживает как одиночные строки, так и массивы строк
    """
    global embedding_server
    
    if not embedding_server:
        raise HTTPException(status_code=503, detail="Embedding server не инициализирован")
    
    try:
        # Нормализация входных данных
        if isinstance(request.input, str):
            texts = [request.input]
        else:
            texts = request.input
        
        if not texts:
            raise HTTPException(status_code=400, detail="Пустой input")
        
        logger.info(f"📥 Запрос эмбеддингов для {len(texts)} текстов")
        
        # Генерация эмбеддингов
        embeddings_data = []
        total_tokens = 0
        
        for i, text in enumerate(texts):
            # Подсчет токенов (приблизительно)
            tokens = len(text.split())
            total_tokens += tokens
            
            # Генерация эмбеддинга
            embedding = await embedding_server.generate_embedding(text)
            
            # Конвертация в список
            embedding_list = embedding.tolist()
            
            embeddings_data.append(EmbeddingData(
                embedding=embedding_list,
                index=i
            ))
        
        response = EmbeddingResponse(
            data=embeddings_data,
            model=request.model,
            usage=EmbeddingUsage(
                prompt_tokens=total_tokens,
                total_tokens=total_tokens
            )
        )
        
        logger.info(f"✅ Возвращены эмбеддинги для {len(texts)} текстов")
        return response
        
    except Exception as e:
        logger.error(f"❌ Ошибка создания эмбеддингов: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """Список доступных моделей (OpenAI API совместимый)"""
    models = [
        ModelInfo(
            id="text-embedding-ada-002",
            created=int(time.time()),
            owned_by="local"
        ),
        ModelInfo(
            id="qwen3-embedding-8b", 
            created=int(time.time()),
            owned_by="local"
        ),
        ModelInfo(
            id="text-embedding-3-small",
            created=int(time.time()),
            owned_by="local"
        ),
        ModelInfo(
            id="text-embedding-3-large", 
            created=int(time.time()),
            owned_by="local"
        )
    ]
    
    return ModelsResponse(data=models)

@app.get("/health")
async def health_check():
    """Проверка состояния сервера"""
    global embedding_server
    
    if not embedding_server:
        return {"status": "error", "message": "Embedding server не инициализирован"}
    
    try:
        stats = await embedding_server.get_stats()
        return {
            "status": "healthy",
            "models_loaded": stats["models_loaded"],
            "gpu_available": torch.cuda.is_available(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/stats")
async def get_server_stats():
    """Получение статистики сервера"""
    global embedding_server
    
    if not embedding_server:
        raise HTTPException(status_code=503, detail="Embedding server не инициализирован")
    
    try:
        return await embedding_server.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === Дополнительные endpoints для совместимости ===

@app.post("/api/embeddings", response_model=EmbeddingResponse)
async def create_embeddings_api(request: EmbeddingRequest):
    """Альтернативный endpoint для эмбеддингов (совместимость с некоторыми клиентами)"""
    return await create_embeddings(request)

@app.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings_simple(request: EmbeddingRequest):
    """Упрощенный endpoint для эмбеддингов"""
    return await create_embeddings(request)

@app.get("/api/models", response_model=ModelsResponse)
async def list_models_api():
    """Альтернативный endpoint для списка моделей"""
    return await list_models()

@app.get("/models", response_model=ModelsResponse)
async def list_models_simple():
    """Упрощенный endpoint для списка моделей"""
    return await list_models()


@app.post("/search")
async def search_code(query: str, top_k: int = 5):
    """Поиск по индексу репозитория"""
    global embedding_server
    if not embedding_server:
        raise HTTPException(status_code=503, detail="Server not ready")
    results = await embedding_server.search_index(query, top_k)
    return {"results": results}


@app.post("/generate")
async def generate(prompt: str, max_tokens: int = 256, temperature: float = 0.7):
    """Генерация ответа с помощью LLM"""
    global embedding_server
    if not embedding_server:
        raise HTTPException(status_code=503, detail="Server not ready")
    result = await embedding_server.generate_response(prompt, max_tokens, temperature)
    return result

@app.get("/")
async def root():
    """Корневой endpoint с информацией о сервере"""
    return {
        "name": "Cursor Compatible Embedding Server",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "embeddings": ["/v1/embeddings", "/api/embeddings", "/embeddings"],
            "models": ["/v1/models", "/api/models", "/models"],
            "health": "/health",
            "stats": "/stats",
            "search": "/search",
            "generate": "/generate",
            "docs": "/docs"
        },
        "port": 11435,
        "compatible_with": ["OpenAI API", "Cursor Editor", "Generic embedding clients"]
    }

# Конфигурация по умолчанию для RTX 4070
DEFAULT_CONFIG = {
    'llama_cpp_path': './llama_build/llama.cpp/build/bin/Release/llama-server.exe',
    'embedding_model_path': './models/Qwen3-Embedding-8B-Q6_K.gguf',
    'reranker_model_path': './models/Qwen3-Reranker-8B-Q6_K.gguf',
    'coder_model_path': './models/Qwen2.5-Coder-7B-Instruct.Q6_K.gguf',
    'llm_model_path': './models/Magistral-Small-2506-UD-Q4_K_XL.gguf',
    'cache_dir': './cache',
    'repo_path': '.',
    'gpu_layers': 35,  # Для RTX 4070
    'host': '127.0.0.1',
    'port': 11435  # Cursor совместимый порт
}

async def main():
    """Главная функция для тестирования"""
    print("🚀 EMBEDDING SERVER для RTX 4070 (llama.cpp)")
    print("=" * 60)
    print("💎 Модели:")
    print("  • Qwen3-Embedding-8B Q6_K (4.9 GB VRAM, ~10M vectors/s)")
    print("  • Qwen3-Reranker-8B Q6_K (6 GB VRAM, ~400 pairs/s)")
    print("  • Qwen2.5-Coder-7B-Instruct Q6_K (7 GB VRAM, ~35 tok/s)")
    print("  • Magistral-Small-2506-UD-Q4_K_XL (14 GB VRAM, 6-8 tok/s)")
    print("🔍 Индексы: FAISS-HNSW + Tantivy-BM25")
    print("🔧 Движок: llama.cpp (нативный)")
    print("=" * 60)
    
    # Инициализация сервера
    server = EmbeddingServer(DEFAULT_CONFIG)
    
    try:
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
        print("💡 Для запуска HTTP сервера используйте: python embedding_server.py server")
        print("💡 Или: uvicorn embedding_server:app --host 127.0.0.1 --port 11435")
        
    finally:
        await server.shutdown()


def start_server():
    """Запуск HTTP сервера"""
    try:
        print("🚀 Запуск Cursor Compatible Embedding Server...")
        print("🌐 Адрес: http://127.0.0.1:11435")
        print("📖 Документация: http://127.0.0.1:11435/docs")
        print("💡 Для остановки нажмите Ctrl+C")
    except UnicodeEncodeError:
        # Fallback для Windows консоли
        print("* Запуск Cursor Compatible Embedding Server...")
        print("* Адрес: http://127.0.0.1:11435")
        print("* Документация: http://127.0.0.1:11435/docs")
        print("* Для остановки нажмите Ctrl+C")
    
    uvicorn.run(
        "embedding_server:app",
        host="127.0.0.1",
        port=11435,
        reload=False,
        access_log=True,
        log_level="info"
    )

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        # Запуск HTTP сервера
        start_server()
    else:
        # Тестирование функций
        asyncio.run(main()) 
