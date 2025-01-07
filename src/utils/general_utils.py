import time
from functools import wraps  # 添加这行导入
from src.utils.log_handler import debug_logger, embed_logger, rerank_logger

# 异步执行环境下的耗时统计装饰器
def get_time_async(func):
    @wraps(func)
    async def get_time_async_inner(*args, **kwargs):
        s_time = time.perf_counter()
        res = await func(*args, **kwargs)  # 注意这里使用 await 来调用异步函数
        e_time = time.perf_counter()
        if 'embed' in func.__name__:
            embed_logger.info('函数 {} 执行耗时: {:.2f} 秒'.format(func.__name__, e_time - s_time))
        elif 'rerank' in func.__name__:
            rerank_logger.info('函数 {} 执行耗时: {:.2f} 秒'.format(func.__name__, e_time - s_time))
        else:
            debug_logger.info('函数 {} 执行耗时: {:.2f} 毫秒'.format(func.__name__, (e_time - s_time) * 1000))
        return res

    return get_time_async_inner

# 同步执行环境下的耗时统计装饰器
def get_time(func):
    def get_time_inner(*arg, **kwargs):
        s_time = time.time()
        res = func(*arg, **kwargs)
        e_time = time.time()
        if 'embed' in func.__name__:
            embed_logger.info('函数 {} 执行耗时: {:.2f} 秒'.format(func.__name__, e_time - s_time))
        elif 'rerank' in func.__name__:
            rerank_logger.info('函数 {} 执行耗时: {:.2f} 秒'.format(func.__name__, e_time - s_time))
        else:
            debug_logger.info('函数 {} 执行耗时: {:.2f} 毫秒'.format(func.__name__, (e_time - s_time) * 1000))
        return res

    return get_time_inner