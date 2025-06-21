#!/usr/bin/env python3
"""
🚀 BM25 INDEX BUILDER
Создание BM25 индекса с помощью Tantivy для текстового поиска
"""

import json
from pathlib import Path
from typing import List, Dict
import re
import math
from collections import defaultdict, Counter

# Опциональный импорт tantivy
try:
    import tantivy
    TANTIVY_AVAILABLE = True
except ImportError:
    TANTIVY_AVAILABLE = False
    print("⚠️ tantivy не установлен, будет использоваться простая TF-IDF имплементация")

def clean_text_for_bm25(text: str) -> str:
    """Очистка текста для BM25 индексирования"""
    # Удаляем лишние символы, но сохраняем важные для кода
    text = re.sub(r'[^\w\s\.\(\)\[\]{}=+\-*/<>!&|^%]', ' ', text)
    # Убираем множественные пробелы
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

class SimpleBM25:
    """Простая реализация BM25 без Tantivy"""
    
    def __init__(self, documents: List[Dict], k1: float = 1.2, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        
        # Предобрабатываем документы
        self.processed_docs = []
        self.doc_freqs = defaultdict(int)
        self.idf = {}
        
        # Обрабатываем документы
        total_len = 0
        for doc in documents:
            cleaned_text = clean_text_for_bm25(doc["text"])
            tokens = cleaned_text.lower().split()
            self.processed_docs.append({
                "id": doc["id"],
                "tokens": tokens,
                "doc": doc,
                "token_count": Counter(tokens)
            })
            total_len += len(tokens)
            
            # Считаем частоту документов для каждого термина
            for token in set(tokens):
                self.doc_freqs[token] += 1
        
        self.avg_doc_len = total_len / len(documents) if documents else 0
        
        # Вычисляем IDF
        N = len(documents)
        for term, df in self.doc_freqs.items():
            self.idf[term] = math.log((N - df + 0.5) / (df + 0.5))
    
    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """Поиск по BM25"""
        query_tokens = clean_text_for_bm25(query).lower().split()
        scores = []
        
        for proc_doc in self.processed_docs:
            score = 0.0
            doc_len = len(proc_doc["tokens"])
            
            for term in query_tokens:
                if term in proc_doc["token_count"]:
                    tf = proc_doc["token_count"][term]
                    idf = self.idf.get(term, 0)
                    
                    # BM25 формула
                    numerator = idf * tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                    score += numerator / denominator
            
            if score > 0:
                result = proc_doc["doc"].copy()
                result["bm25_score"] = score
                scores.append(result)
        
        # Сортируем по убыванию скора
        scores.sort(key=lambda x: x["bm25_score"], reverse=True)
        return scores[:limit]

def build_bm25_index(documents_path: str, output_dir: str = "rag_data") -> bool:
    """Построение BM25 индекса из документов"""
    
    documents_path_obj = Path(documents_path)
    output_dir_obj = Path(output_dir)
    output_dir_obj.mkdir(exist_ok=True)
    
    print(f"🔍 Построение BM25 индекса из: {documents_path_obj}")
    
    # Загружаем документы
    if not documents_path_obj.exists():
        print(f"❌ Файл документов не найден: {documents_path_obj}")
        return False
    
    with open(documents_path_obj, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    print(f"📄 Загружено {len(documents)} документов")
    
    if TANTIVY_AVAILABLE:
        return _build_tantivy_index(documents, output_dir_obj)
    else:
        return _build_simple_index(documents, output_dir_obj)

def _build_tantivy_index(documents: List[Dict], output_dir: Path) -> bool:
    """Построение индекса с помощью Tantivy"""
    
    # Создаем схему индекса
    schema_builder = tantivy.SchemaBuilder()
    
    # Поля индекса
    schema_builder.add_text_field("id", stored=True, tokenizer_name="raw")
    schema_builder.add_text_field("file_path", stored=True, tokenizer_name="raw") 
    schema_builder.add_text_field("chunk_type", stored=True, tokenizer_name="raw")
    schema_builder.add_text_field("text", stored=True, tokenizer_name="default")
    schema_builder.add_text_field("text_cleaned", tokenizer_name="default")  # Для поиска
    schema_builder.add_integer_field("chunk_size", stored=True)
    
    schema = schema_builder.build()
    
    # Создаем индекс
    index_path = output_dir / "bm25_index"
    if index_path.exists():
        import shutil
        shutil.rmtree(index_path)
    
    index = tantivy.Index(schema, path=str(index_path))
    writer = index.writer(heap_size=50_000_000)  # 50MB heap
    
    print("⚙️ Индексирование документов...")
    
    # Индексируем документы
    for doc in documents:
        # Очищаем текст для лучшего поиска
        cleaned_text = clean_text_for_bm25(doc["text"])
        
        # Добавляем документ в индекс
        writer.add_document(tantivy.Document(
            id=str(doc["id"]),
            file_path=doc["file_path"],
            chunk_type=doc["chunk_type"],
            text=doc["text"],
            text_cleaned=cleaned_text,
            chunk_size=doc["chunk_size"]
        ))
    
    # Коммитим изменения
    writer.commit()
    print(f"✅ BM25 индекс создан: {index_path}")
    
    # Тестируем индекс
    print("🧪 Тестирование индекса...")
    reader = index.reader()
    searcher = reader.searcher()
    
    # Простой тестовый запрос
    query_parser = tantivy.QueryParser.for_index(index, ["text_cleaned"])
    test_query = query_parser.parse_query("function")
    
    top_docs = searcher.search(test_query, 5)
    print(f"   Найдено {len(top_docs)} документов по запросу 'function'")
    
    # Сохраняем метаданные индекса
    bm25_metadata = {
        "index_type": "tantivy",
        "index_path": str(index_path),
        "total_documents": len(documents),
        "fields": ["id", "file_path", "chunk_type", "text", "text_cleaned", "chunk_size"],
        "tokenizer": "default",
        "test_query_results": len(top_docs)
    }
    
    with open(output_dir / "bm25_metadata.json", "w", encoding="utf-8") as f:
        json.dump(bm25_metadata, f, ensure_ascii=False, indent=2)
    
    print("💾 BM25 индекс готов к использованию!")
    return True

def _build_simple_index(documents: List[Dict], output_dir: Path) -> bool:
    """Построение простого BM25 индекса"""
    
    print("⚙️ Создание простого BM25 индекса...")
    
    # Создаем BM25 объект
    bm25 = SimpleBM25(documents)
    
    # Сохраняем индекс
    index_data = {
        "documents": documents,
        "processed_docs": bm25.processed_docs,
        "doc_freqs": dict(bm25.doc_freqs),
        "idf": bm25.idf,
        "avg_doc_len": bm25.avg_doc_len,
        "k1": bm25.k1,
        "b": bm25.b
    }
    
    with open(output_dir / "simple_bm25_index.json", "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)
    
    # Тестируем
    test_results = bm25.search("function", 5)
    
    # Метаданные
    bm25_metadata = {
        "index_type": "simple_bm25",
        "index_path": str(output_dir / "simple_bm25_index.json"),
        "total_documents": len(documents),
        "avg_doc_len": bm25.avg_doc_len,
        "vocab_size": len(bm25.idf),
        "test_query_results": len(test_results)
    }
    
    with open(output_dir / "bm25_metadata.json", "w", encoding="utf-8") as f:
        json.dump(bm25_metadata, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Простой BM25 индекс создан: {output_dir / 'simple_bm25_index.json'}")
    print("💾 BM25 индекс готов к использованию!")
    return True

def search_bm25(query: str, index_path: str, limit: int = 10) -> List[Dict]:
    """Поиск в BM25 индексе"""
    index_path_obj = Path(index_path)
    
    if TANTIVY_AVAILABLE and (index_path_obj / "meta.json").exists():
        return _search_tantivy(query, index_path_obj, limit)
    else:
        # Ищем простой индекс
        simple_index_path = index_path_obj / "simple_bm25_index.json"
        if simple_index_path.exists():
            return _search_simple(query, simple_index_path, limit)
        else:
            print(f"❌ BM25 индекс не найден: {index_path_obj}")
            return []

def _search_tantivy(query: str, index_path: Path, limit: int) -> List[Dict]:
    """Поиск в Tantivy индексе"""
    try:
        # Загружаем индекс
        schema_builder = tantivy.SchemaBuilder()
        schema_builder.add_text_field("id", stored=True, tokenizer_name="raw")
        schema_builder.add_text_field("file_path", stored=True, tokenizer_name="raw")
        schema_builder.add_text_field("chunk_type", stored=True, tokenizer_name="raw") 
        schema_builder.add_text_field("text", stored=True, tokenizer_name="default")
        schema_builder.add_text_field("text_cleaned", tokenizer_name="default")
        schema_builder.add_integer_field("chunk_size", stored=True)
        schema = schema_builder.build()
        
        index = tantivy.Index(schema, path=str(index_path))
        reader = index.reader()
        searcher = reader.searcher()
        
        # Создаем запрос
        query_parser = tantivy.QueryParser.for_index(index, ["text_cleaned"])
        parsed_query = query_parser.parse_query(query)
        
        # Выполняем поиск
        top_docs = searcher.search(parsed_query, limit)
        
        results = []
        for score, doc_address in top_docs:
            doc = searcher.doc(doc_address)
            results.append({
                "id": doc["id"][0],
                "file_path": doc["file_path"][0],
                "chunk_type": doc["chunk_type"][0],
                "text": doc["text"][0],
                "chunk_size": doc["chunk_size"][0],
                "bm25_score": score
            })
        
        return results
        
    except Exception as e:
        print(f"❌ Ошибка поиска в Tantivy: {e}")
        return []

def _search_simple(query: str, index_path: Path, limit: int) -> List[Dict]:
    """Поиск в простом BM25 индексе"""
    try:
        with open(index_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        
        # Восстанавливаем BM25 объект
        bm25 = SimpleBM25(index_data["documents"])
        bm25.processed_docs = index_data["processed_docs"]
        bm25.doc_freqs = defaultdict(int, index_data["doc_freqs"])
        bm25.idf = index_data["idf"]
        bm25.avg_doc_len = index_data["avg_doc_len"]
        bm25.k1 = index_data["k1"]
        bm25.b = index_data["b"]
        
        # Восстанавливаем Counter объекты
        for proc_doc in bm25.processed_docs:
            proc_doc["token_count"] = Counter(proc_doc["token_count"])
        
        return bm25.search(query, limit)
        
    except Exception as e:
        print(f"❌ Ошибка поиска в простом BM25: {e}")
        return []

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Использование:")
        print("  Построение: python build_bm25.py build <documents.json> [output_dir]")
        print("  Поиск: python build_bm25.py search <query> <index_path> [limit]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "build":
        if len(sys.argv) < 3:
            print("❌ Укажите путь к файлу documents.json")
            sys.exit(1)
        
        documents_path = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else "rag_data"
        
        success = build_bm25_index(documents_path, output_dir)
        if success:
            print("🎉 BM25 индекс построен успешно!")
        else:
            print("❌ Ошибка построения BM25 индекса")
            sys.exit(1)
    
    elif command == "search":
        if len(sys.argv) < 4:
            print("❌ Укажите запрос и путь к индексу")
            sys.exit(1)
        
        query = sys.argv[2]
        index_path = sys.argv[3]
        limit = int(sys.argv[4]) if len(sys.argv) > 4 else 10
        
        results = search_bm25(query, index_path, limit)
        
        print(f"🔍 Результаты поиска '{query}':")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['file_path']} (score: {result['bm25_score']:.3f})")
            print(f"   Type: {result['chunk_type']}")
            print(f"   Text: {result['text'][:100]}...")
            print()
    
    else:
        print("❌ Неизвестная команда. Используйте 'build' или 'search'")
        sys.exit(1) 