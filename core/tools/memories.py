from langchain_core.tools import tool
import aiohttp
import asyncio
from datetime import datetime

MEMOS_BASE_URL = "http://127.0.0.1:8004"
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
                    print(f"已成功记住: {content}")
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
                json = payload,
                timeout = aiohttp.ClientTimeout(total = 3,connect = 1)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    memories = data.get("memories", [])
                    if not memories:
                        return f"没有与{query}相关的记忆"
                    lines = []
                    for mem in memories:
                        content = mem.get("content", "")
                        created_at = mem.get("created_at", "")
                        time_str = ""
                        if created_at:
                            try:
                                dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                                time_str = dt.strftime("%Y年%m月%d日")
                            except:
                                time_str = created_at[:10]
                            
                        #检查是否更新过
                        updated_at = mem.get("updated_at")
                        update_mark = ""
                        if updated_at and updated_at != created_at:
                            update_mark = "(已更新)"
                        
                        #组装最终回复
                        line = f"{content}"
                        if time_str:
                            line += f" [{time_str}]"
                        if update_mark:
                            line += f" {update_mark}"
                        lines.append(line)
                    result = f"相关记忆\n" + "\n".join(lines)
                    return result

                else:
                    error_text = await response.text()
                    return f"搜索失败 (HTTP {response.status}): {error_text}"
    except Exception as e:
        return f"搜索失败: {e}"