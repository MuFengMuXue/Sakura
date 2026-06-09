import sys
import os

from core.graph import app

def main():
    print("机器人已启动，输入 quit 退出")
    config = {"configurable": {"thread_id": "1"}}
    while True:
        user = input("\n你: ")
        if user.lower() == "quit":
            break
        result = app.invoke({"messages": [("user", user)]}, config)
        print(f"AI: {result['messages'][-1].content}")

if __name__ == "__main__":
    main()