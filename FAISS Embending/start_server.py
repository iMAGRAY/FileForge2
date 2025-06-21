import subprocess
import sys
import os
import time

def start_embedding_server():
    print("�� ЗАПУСК EMBEDDING СЕРВЕРА...")
    
    # Проверяем существование файлов
    script_path = "embedding_server.py"
    if not os.path.exists(script_path):
        print("❌ embedding_server.py не найден")
        return False
    
    try:
        # Запускаем сервер
        process = subprocess.Popen([
            sys.executable, script_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print("✅ Embedding сервер запущен")
        print("🔗 http://localhost:8000")
        
        # Ждем немного и проверяем что процесс запустился
        time.sleep(2)
        if process.poll() is None:
            print("🎉 Сервер работает успешно")
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Ошибка запуска: {stderr.decode()}")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

if __name__ == "__main__":
    start_embedding_server()
