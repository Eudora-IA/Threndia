"""
Thread Manager for concurrent analysis operations
"""
import asyncio
from typing import Dict, List, Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from ..models.data_models import ThreadConfig


class ThreadManager:
    """Manages concurrent thread execution for analysis tasks"""
    
    def __init__(self, max_workers: int = 5):
        """
        Initialize ThreadManager
        
        Args:
            max_workers: Maximum number of concurrent workers
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_threads: Dict[str, Any] = {}
        self.thread_results: Dict[str, Any] = {}
    
    def submit_task(
        self,
        thread_id: str,
        task: Callable,
        *args,
        **kwargs
    ) -> None:
        """
        Submit a task for execution
        
        Args:
            thread_id: Unique identifier for the thread
            task: Callable task to execute
            *args: Positional arguments for the task
            **kwargs: Keyword arguments for the task
        """
        future = self.executor.submit(task, *args, **kwargs)
        self.active_threads[thread_id] = {
            "future": future,
            "status": "running",
            "started_at": datetime.now(),
        }
    
    def get_thread_status(self, thread_id: str) -> Optional[str]:
        """
        Get the status of a thread
        
        Args:
            thread_id: Thread identifier
            
        Returns:
            Status string or None if thread doesn't exist
        """
        if thread_id not in self.active_threads:
            return None
        return self.active_threads[thread_id]["status"]
    
    def get_result(self, thread_id: str, timeout: Optional[float] = None) -> Any:
        """
        Get the result of a completed thread
        
        Args:
            thread_id: Thread identifier
            timeout: Optional timeout in seconds
            
        Returns:
            Result of the thread execution
            
        Raises:
            KeyError: If thread doesn't exist
            TimeoutError: If timeout is reached
        """
        if thread_id not in self.active_threads:
            raise KeyError(f"Thread {thread_id} not found")
        
        future = self.active_threads[thread_id]["future"]
        result = future.result(timeout=timeout)
        
        self.active_threads[thread_id]["status"] = "completed"
        self.thread_results[thread_id] = result
        
        return result
    
    def wait_for_all(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Wait for all active threads to complete
        
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            Dictionary mapping thread_id to results
        """
        results = {}
        for thread_id in list(self.active_threads.keys()):
            try:
                result = self.get_result(thread_id, timeout=timeout)
                results[thread_id] = result
            except Exception as e:
                results[thread_id] = {"error": str(e)}
                self.active_threads[thread_id]["status"] = "failed"
        
        return results
    
    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the thread executor
        
        Args:
            wait: Whether to wait for active threads to complete
        """
        self.executor.shutdown(wait=wait)
    
    async def submit_async_task(
        self,
        thread_id: str,
        task: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Submit and await an async task
        
        Args:
            thread_id: Unique identifier for the thread
            task: Async callable task to execute
            *args: Positional arguments for the task
            **kwargs: Keyword arguments for the task
            
        Returns:
            Result of the async task
        """
        self.active_threads[thread_id] = {
            "status": "running",
            "started_at": datetime.now(),
        }
        
        try:
            result = await task(*args, **kwargs)
            self.active_threads[thread_id]["status"] = "completed"
            self.thread_results[thread_id] = result
            return result
        except Exception as e:
            self.active_threads[thread_id]["status"] = "failed"
            self.thread_results[thread_id] = {"error": str(e)}
            raise
    
    async def run_concurrent_analyses(
        self,
        tasks: List[tuple],
        config: Optional[ThreadConfig] = None
    ) -> List[Any]:
        """
        Run multiple analysis tasks concurrently
        
        Args:
            tasks: List of (task_function, args, kwargs) tuples
            config: Optional ThreadConfig for execution parameters
            
        Returns:
            List of results from all tasks
        """
        if config:
            max_concurrent = config.max_concurrent_analyses
        else:
            max_concurrent = self.max_workers
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_with_semaphore(task_func, args, kwargs):
            async with semaphore:
                if asyncio.iscoroutinefunction(task_func):
                    return await task_func(*args, **kwargs)
                else:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        self.executor, task_func, *args, **kwargs
                    )
        
        coroutines = [
            run_with_semaphore(task_func, args or (), kwargs or {})
            for task_func, args, kwargs in tasks
        ]
        
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        return results
    
    def get_all_statuses(self) -> Dict[str, str]:
        """
        Get status of all threads
        
        Returns:
            Dictionary mapping thread_id to status
        """
        return {
            thread_id: info["status"]
            for thread_id, info in self.active_threads.items()
        }
