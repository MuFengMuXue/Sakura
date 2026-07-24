from langchain_core.tools import tool
import aiohttp
import asyncio
from datetime import datetime
import logging
logger = logging.getLogger(__name__)


MEMOS_BASE_URL = "http://127.0.0.1:8003"
DEFAULT_USER_ID = "01"

@tool
async def memos_add_memory(content: str) -> str:
    """
    记住重要信息。
    当用户明确说“记住”、“别忘了”，或者透露个人偏好时使用。
    或者是你觉得重要的内容，也可以调用。
    参数：
        content: 要记住的内容（简洁明了）
    """
    payload = {
        "messages": [{"role": "user", "content": content}],
        "user_id": DEFAULT_USER_ID
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{MEMOS_BASE_URL}/add",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10, connect=1)
            ) as resp:
                if resp.status == 200:
                    logger.info(f"已成功记住: {content}")
                    return f"已成功记住: {content}"
                else:
                    error_text = await resp.text()
                    return f"保存失败 (HTTP {resp.status}): {error_text}"
    except Exception as e:
        return f"保存失败: {e}"

@tool
async def memos_search_memory(query: str) -> str:
    """
    从记忆中深度搜索相关的历史信息和对话。
    当用户询问'你还记得吗'、'之前说过'、'上次'、'以前'、'有没有'、'记不记得'等涉及过去事件的问题时必须使用此工具！
    也可用于主动搜索用户的偏好、经历、约定等
    参数：
        query: 需要查询的内容
    """
    payload = {
        "query": query,
        "user_id": DEFAULT_USER_ID,
        "top_k": 3,
        "similarity_threshold": 0.5
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{MEMOS_BASE_URL}/search",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=3, connect=1)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    memories = data.get("memories", [])
                    if not memories:
                        return f"没有与'{query}'相关的记忆"
                    lines = []
                    for mem in memories:
                        mem_id = mem.get("id", "")
                        content = mem.get("content", "")
                        created_at = mem.get("created_at", "")
                        time_str = ""
                        if created_at:
                            try:
                                dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                                time_str = dt.strftime("%Y年%m月%d日")
                            except:
                                time_str = created_at[:10]
                        updated_at = mem.get("updated_at")
                        update_mark = ""
                        if updated_at and updated_at != created_at:
                            update_mark = "(已更新)"
                        # 构建一行，包含 ID
                        line = f"{content}"
                        if time_str:
                            line += f" [{time_str}]"
                        if update_mark:
                            line += f" {update_mark}"
                        if mem_id:
                            line += f" [ID: {mem_id}]"
                        lines.append(line)
                    result = f"相关记忆:\n" + "\n".join(lines)
                    return result
                else:
                    error_text = await response.text()
                    return f"搜索失败 (HTTP {response.status}): {error_text}"
    except Exception as e:
        return f"搜索失败: {e}"
    
@tool
async def memos_correct_memory(
    memory_id: str,
    action: str,
    new_content: str = "",
    reason: str = ""
    ) -> str:
    """
    修正、补充或删除已有的记忆。
    需要先用 memos_search_memory 获取记忆 ID。
    参数：
        memory_id: 要操作的记忆 ID（通过搜索获取）
        action: 操作类型，可选值：'correct'（修正）、'supplement'（补充）、'delete'（删除）
        new_content: 修正或补充的新内容（当 action 为 correct 或 supplement 时必填）
        reason: 操作原因（可选）
    """
    #元素检查
    if not memory_id:
        return "错误，未提供记忆id。请先通过memos_search_memory搜索获取并获取记忆id"
    
    if action not in ['correct','supplement','delete']:
        return "错误：未指定操作类型。可选：correct、supplement、delete"
    
    if action in ['correct','supplement'] and not new_content:
        return f"错误，{action}必须提供new_content"
   
     
    payload = {
        "memory_id":memory_id,
        "feedback_type": action,
        "reason": reason or "",
        "user_id": DEFAULT_USER_ID,
        
    }
    if action in ['correct', 'supplement']:
        payload["correction"] = new_content
    try:
         async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{MEMOS_BASE_URL}/memory/feedback",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10, connect=1)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("status") == "success":
                        if action == "delete":
                            return f"记忆成功删除，id:{memory_id}"
                        else:
                            action_name = {'correct':'修正','supplement':'补充'}[action]
                            return f"已成功{action_name}\n ID:{memory_id}\n新内容: {data.get('new_content', new_content)}"
                    else:
                        return f"操作失败{data.get('message'),'未知错误'}"
                elif resp.status == 404:
                    return f"记忆ID{memory_id}不存在，请确认ID是否正确"
                else:
                    error_text = await resp.text()
                    return f"操作失败：{resp.status}，错误信息：{error_text}"
    except Exception as e:
        return f"操作失败:{e}"
                  


                
                

