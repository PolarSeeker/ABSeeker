import queue as _queue

from megatron.core.dist_checkpointing.strategies.filesystem_async import FileSystemWriterAsync


class ROCmFileSystemWriterAsync(FileSystemWriterAsync):

    @staticmethod
    def preload_tensors(write_buckets, non_blocking=False):
        return FileSystemWriterAsync.preload_tensors(write_buckets, non_blocking=False)

    @staticmethod
    def write_preloaded_data_multiproc(
        transform_list, use_msc, rank, write_buckets, global_results_queue
    ):
        rq = _queue.SimpleQueue()
        cq = type("_CQ", (), {"get": lambda s: None, "task_done": lambda s: None})()
        results = {}
        for i, bucket in enumerate(write_buckets):
            FileSystemWriterAsync.write_preloaded_data(
                transform_list, local_proc_idx=i, write_bucket=bucket,
                results_queue=rq, count_queue=cq, use_fsync=True, use_msc=use_msc,
            )
            _, res = rq.get()
            results[i] = res
        global_results_queue.put(results)
