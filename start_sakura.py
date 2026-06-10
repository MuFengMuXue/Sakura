from core.graph import app
from langchain_core.messages import HumanMessage
def main():
    print("机器人已启动，输入 quit 退出")
    config = {"configurable": {"thread_id": "1"}}
    while True:
        user = input("\n你: ")
        if user.lower() == "quit":
            break
        print("AI: ", end="", flush=True)
        for msg_chunk, _ in app.stream(
         {"messages": [HumanMessage(content=user)]},
         config,
         stream_mode="messages",
        ):
          if msg_chunk.content:
            print(msg_chunk.content, end="", flush=True)
        print()  # 换行
      

if __name__ == "__main__":
    main()