# 🔌 API Endpoints - Cursor Compatible Embedding Server

## 📋 Полный список endpoints

### ✅ Основные OpenAI-совместимые endpoints

**🎯 Все endpoints для максимальной совместимости:**
- **Эмбеддинги**: `/v1/embeddings`, `/api/embeddings`, `/embeddings`
- **Модели**: `/v1/models`, `/api/models`, `/models`
- **Статус**: `/health`, `/stats`
- **Документация**: `/docs`, `/openapi.json`
- **Информация**: `/` (корневой endpoint)

#### 1. `POST /v1/embeddings`
**Назначение**: Генерация эмбеддингов (основной endpoint для Cursor)

**Запрос**:
```json
{
  "input": ["текст для эмбеддинга"] | "одиночный текст",
  "model": "text-embedding-ada-002",
  "encoding_format": "float"
}
```

**Ответ**:
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.1, 0.2, ...],
      "index": 0
    }
  ],
  "model": "text-embedding-ada-002",
  "usage": {
    "prompt_tokens": 10,
    "total_tokens": 10
  }
}
```

#### 2. `GET /v1/models`
**Назначение**: Список доступных моделей

**Ответ**:
```json
{
  "object": "list",
  "data": [
    {
      "id": "text-embedding-ada-002",
      "object": "model",
      "created": 1735142223,
      "owned_by": "local"
    },
    {
      "id": "qwen3-embedding-8b",
      "object": "model", 
      "created": 1735142223,
      "owned_by": "local"
    }
  ]
}
```

### ✅ Мониторинг и диагностика

#### 3. `GET /health`
**Назначение**: Проверка состояния сервера

**Ответ**:
```json
{
  "status": "healthy",
  "models_loaded": {"embedding": true, "reranker": false},
  "gpu_available": true,
  "timestamp": "2024-12-26T10:30:00"
}
```

#### 4. `GET /stats`
**Назначение**: Подробная статистика сервера

**Ответ**:
```json
{
  "models_loaded": {
    "embedding": true,
    "reranker": false,
    "llm": false
  },
  "index_size": 0,
  "cache_dir": "./cache",
  "device": "cuda",
  "gpu_memory": {
    "total_gb": 12.0,
    "allocated_gb": 2.1,
    "reserved_gb": 2.5
  }
}
```

### ✅ Автоматически генерируемые endpoints (FastAPI)

#### 5. `GET /docs`
**Назначение**: Swagger UI документация

#### 6. `GET /openapi.json`
**Назначение**: OpenAPI схема

#### 7. `GET /redoc`
**Назначение**: ReDoc документация

## 🎯 Совместимость с Cursor

### Конфигурация Cursor:
```
Custom OpenAI Endpoint: http://127.0.0.1:11435
API Key: local-key (любая строка)
```

### Поддерживаемые модели в Cursor:
- `text-embedding-ada-002` (основная)
- `text-embedding-3-small`
- `text-embedding-3-large`
- `qwen3-embedding-8b`

## 🧪 Тестирование endpoints

### Быстрый тест через curl:
```bash
# Проверка здоровья
curl http://127.0.0.1:11435/health

# Список моделей
curl http://127.0.0.1:11435/v1/models

# Генерация эмбеддинга
curl -X POST http://127.0.0.1:11435/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "model": "text-embedding-ada-002"}'
```

### Автоматическое тестирование:
```bash
# Python тест
python test_endpoints.py

# Batch тест
test_curl.bat
```

## ⚡ Производительность endpoints

| Endpoint | Время ответа | Нагрузка |
|----------|--------------|----------|
| `/health` | ~1ms | Минимальная |
| `/v1/models` | ~1ms | Минимальная |
| `/stats` | ~5ms | Низкая |
| `/v1/embeddings` | ~100-200ms | Высокая (GPU) |

## 🔧 Устранение неполадок

### Endpoint не отвечает:
1. Проверьте запуск сервера: `start_server.bat`
2. Проверьте порт: `netstat -an | findstr 11435`
3. Проверьте логи в консоли сервера

### Ошибка 500 в `/v1/embeddings`:
1. Проверьте наличие моделей в папке `models/`
2. Проверьте GPU память: `nvidia-smi`
3. Проверьте логи для деталей ошибки

### Cursor не видит сервер:
1. Убедитесь что URL точно `http://127.0.0.1:11435`
2. Перезапустите Cursor после настройки
3. Проверьте что сервер отвечает: `curl http://127.0.0.1:11435/health`

## 🎉 Готовность к production

✅ **Все endpoints реализованы**  
✅ **OpenAI API совместимость**  
✅ **Автоматическая документация**  
✅ **Мониторинг и диагностика**  
✅ **Обработка ошибок**  
✅ **CORS поддержка**  
✅ **Тестовые скрипты**

Сервер готов к использованию с Cursor! 