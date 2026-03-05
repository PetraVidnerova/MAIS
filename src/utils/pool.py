"""Multiprocessing worker pool for parallel model evaluation in the MAIS simulation.

This module provides a simple process-pool abstraction built on top of
:mod:`multiprocessing`. Each worker process runs an evaluation function in an
infinite loop, consuming queries from a shared input queue and posting results
to a shared output queue. The main process enqueues tasks and dequeues results
through the :class:`Pool` interface.
"""

from multiprocessing import Process, Queue


def worker(name, evalfunc, querries, answers, model):
    """Entry point for a pool worker process.

    Runs an infinite loop: retrieves one query at a time from the ``querries``
    queue, evaluates ``evalfunc(model, query)``, and places the result onto
    the ``answers`` queue. The loop continues until the process is terminated
    externally (e.g., via :meth:`Pool.close`).

    Args:
        name (int): Numeric identifier for this worker (used for logging or
            debugging).
        evalfunc (callable): Function to evaluate. Called as
            ``evalfunc(model, query)`` and must return a serialisable result.
        querries (multiprocessing.Queue): Input queue from which queries are
            consumed.
        answers (multiprocessing.Queue): Output queue to which computed
            answers are posted.
        model: Model object passed as the first argument to ``evalfunc``.
            Each worker receives its own model instance.
    """
    while True:
        querry = querries.get()
        answer = evalfunc(model, querry)
        answers.put(answer)


class Pool:
    """Process pool that parallelises model evaluation across multiple workers.

    Workers consume tasks from a shared queries queue and post results to a
    shared answers queue. The pool is non-blocking from the caller's
    perspective: tasks are submitted via :meth:`putQuerry` and results are
    retrieved (blocking) via :meth:`getAnswer`.

    Args:
        processors (int): Number of worker processes to spawn.
        evalfunc (callable): Evaluation function passed to each worker.
            Signature: ``evalfunc(model, query) -> answer``.
        models (list): List of model instances, one per worker. Element ``i``
            is passed to worker ``i``.
    """

    def __init__(self, processors, evalfunc, models):

        self.querries = Queue()
        self.answers = Queue()

        self.workers = []
        for i in range(processors):
             worker_i = Process(target=worker, args=(i, evalfunc, self.querries, self.answers, models[i]))
             self.workers.append(worker_i)
             worker_i.start()        # Launch worker() as a separate python process

    def putQuerry(self, querry):
        """Enqueue a query for processing by one of the worker processes.

        Args:
            querry: The query object to evaluate. Must be picklable.
        """
        self.querries.put(querry)

    def getAnswer(self):
        """Block until an answer is available and return it.

        Returns:
            The result produced by ``evalfunc`` for the oldest unread query.
        """
        return self.answers.get()

    def close(self):
        """Terminate all worker processes and shut down the pool.

        Sends a ``SIGTERM`` signal to each worker process. After calling this
        method the pool should not be used further.
        """
        for w in self.workers:
            w.terminate()
        #print("pool killed")
        
