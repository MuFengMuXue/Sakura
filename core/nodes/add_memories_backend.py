import os
import json
import aiohttp
import aiofiles
from core.nodes.add_memory_manager import (
    notification_queue, processing_lock, MEMOS_FILE, TEMP_FILE,
    BATCH_SIZE, MEMOS_BASE_URL, logger
)

async def process_temp_file(temp_path: str) -> bool:
    """读取临时文件，发送数据，成功返回True，失败返回False"""
    try:
        async with aiofiles.open(temp_path, "r", encoding="utf-8") as f:
            lines = await f.readlines()
        if not lines:
            os.remove(temp_path)
            logger.info("临时文件为空，已删除")
            return True

        rounds = []
        for line in lines:
            line = line.strip()
            if line:
                try:
                    rounds.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if not rounds:
            os.remove(temp_path)
            return True

        messages_to_send = []
        for r in rounds:
            messages_to_send.append({"role": "user", "content": r["user"]})
            messages_to_send.append({"role": "assistant", "content": r["assistant"]})

        payload = {"messages": messages_to_send, "user_id": "01"}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{MEMOS_BASE_URL}/add",
                json=payload,
                timeout=aiohttp.ClientTimeout(connect=1)
            ) as resp:
                if resp.status == 200:
                    logger.info(f"成功发送 {len(rounds)} 轮记忆")
                    os.remove(temp_path)
                    return True
                else:
                    error_text = await resp.text()
                    logger.error(f"发送失败，状态码 {resp.status}，错误: {error_text}")
                    return False
    except Exception as e:
        logger.exception("处理临时文件时发生异常")
        return False


async def flush_memories_task():
    """事件驱动的后台任务：等待通知，检查文件，按需发送"""
    while True:
        await notification_queue.get()   # 阻塞直到有通知
        async with processing_lock:
            # 1. 优先处理遗留的临时文件（重试）
            if os.path.exists(TEMP_FILE):
                logger.info("发现遗留临时文件，尝试重发...")
                success = await process_temp_file(TEMP_FILE)
                if not success:
                    logger.warning("临时文件发送失败，保留等待下次通知重试")
                    continue   # 本次不再检查原文件

            # 2. 检查原文件
            if not os.path.exists(MEMOS_FILE):
                continue

            async with aiofiles.open(MEMOS_FILE, "r", encoding="utf-8") as f:
                lines = await f.readlines()
            if len(lines) < BATCH_SIZE:
                logger.debug("当前行数不足，不发送")
                continue

            # 3. 原子重命名原文件 -> 临时文件
            try:
                os.rename(MEMOS_FILE, TEMP_FILE)
            except Exception as e:
                logger.error(f"重命名失败: {e}")
                continue

            # 4. 创建新的空原文件
            try:
                open(MEMOS_FILE, "w").close()
            except Exception as e:
                logger.error(f"创建新文件失败: {e}")
                os.rename(TEMP_FILE, MEMOS_FILE)   # 回退
                continue

            # 5. 处理临时文件（发送）
            success = await process_temp_file(TEMP_FILE)
            if not success:
                logger.warning("HTTP发送失败，临时文件保留，等待下次通知重试")
            # 如果成功，临时文件已被删除