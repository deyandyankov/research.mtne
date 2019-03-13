import threading
from queue import Queue
from multiprocessing.pool import ApplyResult

import tabular_logger as tlogger

class AsyncWorker(object):
    @property
    def concurrent_tasks(self):
        raise NotImplementedError()

    def run_async(self, task_id, task, callback):
        raise NotImplementedError()


class WorkerHub(object):
    def __init__(self, workers, input_queue, done_queue):
        print("=== Initializing WorkerHub()")
        self.done_buffer = Queue()
        self.workers = workers
#        self.available_workers = Queue()
        self.done_queue = done_queue
        self._cache = {}
        self.input_queue = input_queue
        self.next_game_type_must_be = 0

#        for w in workers:
#            for t in w.concurrent_tasks:
#                self.available_workers.put((w, t))

        self.available_workers_0 = Queue()
        for w in workers:
            for t in w.concurrent_tasks:
                if w.game_index == 0:
                    self.available_workers_0.put((w, t))

        self.available_workers_1 = Queue()
        for w in workers:
            for t in w.concurrent_tasks:
                if w.game_index == 1:
                    self.available_workers_1.put((w, t))

        self.__initialize_handlers()

    def __initialize_handlers(self):
        self._input_handler = threading.Thread(
            target=WorkerHub._handle_input,
            args=(self,)
            )
        self._input_handler._state = 0

        self._output_handler = threading.Thread(
            target=WorkerHub._handle_output,
            args=(self,)
            )
        self._output_handler._state = 0

    def worker_callback(self, worker, subworker, result):
        worker_task = (worker, subworker)
#        self.available_workers.put(worker_task)
        if worker.game_index == 0:
            self.available_workers_0.put(worker_task)
        if worker.game_index == 1:
            self.available_workers_1.put(worker_task)

        task_id = self._cache[worker_task]
        del self._cache[worker_task]
        print("== putting into done buffer: worker.game_index={}, task_id={}, result={}".format(worker.game_index, task_id, result))
        self.done_buffer.put((worker.game_index, task_id, result))

    @staticmethod
    def _handle_input(self):
        try:
            while True:
#                worker_task = self.available_workers.get()
#                if worker_task is None:
#                    tlogger.info('WorkerHub._handle_input done')
#                    break
#                worker, subworker = worker_task
                if self.next_game_type_must_be == 0:
#                    print("Getting queue 0")
                    current_q = self.available_workers_0
                if self.next_game_type_must_be == 1:
#                    print("Getting queue 1")
                    current_q = self.available_workers_1

                worker_task = current_q.get()
                if worker_task is None:
                    tlogger.info('WorkerHub._handle_input done')
                    break
                worker, subworker = worker_task
                if self.next_game_type_must_be == 0:
                    self.next_game_type_must_be = 1
                else:
                    self.next_game_type_must_be = 0

                task = self.input_queue.get()
                if task is None:
                    tlogger.info('WorkerHub._handle_input done')
                    break
                task_id, task = task
                self._cache[worker_task] = task_id

                worker.run_async(subworker, task, self.worker_callback)
        except:
            tlogger.exception('WorkerHub._handle_input exception thrown')
            raise

    @staticmethod
    def _handle_output(self):
        try:
            while True:
                result = self.done_buffer.get()
                if result is None:
                    tlogger.info('RESULT IS NONE: WorkerHub._handle_output done')
                    continue
                    #break
                self.done_queue.put(result)
        except:
            tlogger.exception('WorkerHub._handle_output exception thrown')
            raise

    def initialize(self):
        self._input_handler.start()
        self._output_handler.start()

    def close(self):
#        self.available_workers.put(None)
        self.available_workers_0.put(None)
        self.available_workers_1.put(None)

        self.input_queue.put(None)
        self.done_buffer.put(None)

class AsyncTaskHub(object):
    def __init__(self, input_queue=None, results_queue=None):
        if input_queue is None:
            input_queue = Queue(64)
        self.input_queue = input_queue
        self._cache = {}
        self.results_queue = None
        if results_queue is not None:
            self.results_queue = results_queue

            self._output_handler = threading.Thread(
                target=AsyncTaskHub._handle_output,
                args=(self,)
                )
            self._output_handler.daemon = True
            self._output_handler._state = 0
            self._output_handler.start()

    @staticmethod
    def _handle_output(self):
        try:
            while True:
                result = self.results_queue.get()
                if result is None:
                    tlogger.info('AsyncTaskHub._handle_output done')
                    break
                self.put(result)
        except:
            tlogger.exception('AsyncTaskHub._handle_output exception thrown')
            raise

    def run_async(self, task, callback=None, error_callback=None):
        result = ApplyResult(self._cache, callback, error_callback)
        self.input_queue.put((result._job, task))
        return result

    def put(self, result):
        game_index, job, result = result
        return_result = [game_index]
        for e in result:
            return_result.append(e)
        return_result = tuple(return_result)
        self._cache[job]._set(0, (True, return_result))

class MTWorkerHub(WorkerHub):
    pass

class MTAsyncTaskHub(AsyncTaskHub):
    pass

class MTAsyncTaskHub2(AsyncTaskHub):
    def __init__(self, input_queue=None, results_queue=None):
        print("== Instantiating MTAsyncTaskHub")

        self.next_game_type_must_be_0 = 0
        if input_queue is None:
            input_queue = Queue(64)

        self.input_queue = input_queue
        self._cache = {}
        self.results_queue_0 = None
        self.results_queue_1 = None
        self._output_handler = threading.Thread(
            target=MTAsyncTaskHub._handle_output,
            args=(self,)
        )
        self._output_handler.daemon = True
        self._output_handler._state = 0
        self._output_handler.start()

    @staticmethod
    def _handle_output(self):
        try:
            while True:
                result_0 = self.results_queue_0.get()
                result_1 = self.results_queue_1.get()
                if result_0 is None and result_1 is None:
                    tlogger.info('AsyncTaskHub._handle_output done')
                    break
                if result_0 is not None:
                    self.put(result_0)
                if result_1 is not None:
                    self.put(result_1)
        except:
            tlogger.exception('AsyncTaskHub._handle_output exception thrown')
            raise

    def run_async(self, game_index, task, callback=None, error_callback=None):
        print(task)
        result = ApplyResult(self._cache, callback, error_callback)
        if game_index == 0:
            self.input_queue_0.put((result._job, task))
        if game_index == 1:
            self.input_queue_1.put((result._job, task))
        return result

    def put(self, result):
        game_index, job, result = result
        return_result = [game_index]
        for e in result:
            return_result.append(e)
        return_result = tuple(return_result)
        self._cache[job]._set(0, (True, return_result))
