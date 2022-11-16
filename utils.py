from enum import Enum


class LogType(Enum):
    info = "INFO"
    debug = "DEBUG"

class LogMode(Enum):
    off = "OFF"
    info = "INFO"
    debug = "DEBUG"

def logger(message: str, mode: LogMode = LogMode.info, type: LogType = LogType.info):
    if mode == LogMode.off:
        return
    
    if mode == LogMode.info and type == LogType.debug:
        return
    
    print(f"[{type.value}] {message}")