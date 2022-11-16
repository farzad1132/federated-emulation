from enum import Enum


class LogType(Enum):
    info = "INFO"
    debug = "DEBUG"

class LogMode(Enum):
    off = "OFF"
    info = "INFO"
    debug = "DEBUG"

class Logger:
    def __init__(self, title: str, rank: int, mode: LogMode) -> None:
        self.title = title
        self.rank = rank
        self.mode = mode
        
    def log(self, message: str, type: LogType = LogType.info):
        if self.mode == LogMode.off:
            return
        
        if self.mode == LogMode.info and type == LogType.debug:
            return
        
        print(f"[{type.value}][{self.title}, rank={self.rank}] {message}")