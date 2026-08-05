import aiohttp
import asyncio

from dataclasses import dataclass
from typing import TypedDict, Optional

class BenchmarkConfig(TypedDict):
    # benchmark 的数据集路径
    dataset_path: str
    # 运行的名称,用于区别 baselines
    dataset_name: str

    # 只给 search 和 visit 工具
    search_and_visit: bool
    # 只给explore
    explore: bool
    # 不给工具
    no_tools: bool
    
    # Explore-specific flags
    with_explore_context: bool
    tools_response: bool

    # Execution control
    sample_size: Optional[int]
    max_tokens: int
    max_concurrency: int
    pool_no_progress_timeout: int
    tool_count_max: int
    reasoning_effort: str
    
    # Restart logic
    restart: Optional[str]
    restart_num: int
    
    # Logging and output
    monitoring: bool
    print_stream: bool
    debug: bool

    # System prompt 
    system_prompt: str

@dataclass
class TaskContext:
    # 共享网络会话
    session: aiohttp.ClientSession
    
    # 并发控制
    semaphore: asyncio.Semaphore
    
    # 共享文件/状态锁
    state_lock: asyncio.Lock
    file_write_lock: asyncio.Lock
    
    # 运行配置
    config: BenchmarkConfig
    
    # 输出路径
    run_temp_root: str
    logs_dir: str
    state_file: str
    result_path: str
    log_path: str