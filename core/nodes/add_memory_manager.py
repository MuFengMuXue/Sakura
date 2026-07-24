import asyncio
import logging

logger = logging.getLogger(__name__)

# 事件通知队列
notification_queue = asyncio.Queue()
# 发送处理锁，防止同时执行多个发送任务（虽然只有一个后台任务，但锁有助于防御）
processing_lock = asyncio.Lock()

# 文件路径常量
MEMOS_FILE = "memories_01.jsonl"
TEMP_FILE = MEMOS_FILE + ".processing"
BATCH_SIZE = 5
MEMOS_BASE_URL = "http://127.0.0.1:8003"