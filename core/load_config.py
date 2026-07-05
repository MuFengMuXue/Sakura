#加载配置文件
import os
from pathlib import Path
from pydantic import BaseModel
import yaml

class LLMConfig(BaseModel):
    model: str
    api_key: str
    base_url: str
    temperature: float

class AppConfig(BaseModel):
    llm: LLMConfig


_config: AppConfig = None
_persona: str = None

def _get_project_root():
    return Path(__file__).parent.parent

def load_config()->AppConfig:
    config_path = _get_project_root() / "config" / "settings.yaml"
    with open(config_path, 'r' , encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return AppConfig(**data)

def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config

def get_persona() -> str:
    global _persona
    if _persona is None:
        persona_path = _get_project_root() / "config"/ "persona.yaml"
        with open(persona_path,'r',encoding='utf-8') as f:
            data = yaml.safe_load(f)
            _persona = data.get("persona","")
    return _persona

