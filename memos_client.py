'''
懒加载，避免重复加载造成延迟
'''

import os 
from dotenv import load_dotenv
load_dotenv() #加载环境变量

from memos.api.handlers import init_server
from memos.api.handlers.base_handler import HandlerDependencies
from memos.api.handlers.search_handler import SearchHandler

_search_handler = None

def get_search_handler():
    """懒加载获取 SearchHandler 单例"""
    global _search_handler
    if _search_handler is None:
        components = init_server()
        dependencies = HandlerDependencies.from_init_server(components)
        _search_handler = SearchHandler(dependencies)
    return _search_handler