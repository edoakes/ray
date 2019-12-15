from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from multiprocessing import TimeoutError
import os
import random

import ray


class AsyncResult(object):
    def __init__(self,
                 chunk_object_ids,
                 callback=None,
                 error_callback=None,
                 single_result=False):
        if callback is not None or error_callback is not None:
            raise NotImplementedError(
                "Callbacks are not currently implemented.")
        self._chunk_object_ids = chunk_object_ids
        self._callback = callback
        self._error_callback = error_callback
        self._single_result = single_result

    def get(self, timeout=None):
        if timeout is not None:
            timeout = float(timeout)
        try:
            results = []
            for chunk_results in ray.get(
                    self._chunk_object_ids, timeout=timeout):
                results.extend(chunk_results)
            if self._single_result:
                results = results[0]
            return results
        except ray.exceptions.RayTimeoutError:
            raise TimeoutError

    def wait(self, timeout=None):
        if timeout is not None:
            timeout = float(timeout)
        ray.wait(
            self._chunk_object_ids,
            num_returns=len(self._chunk_object_ids),
            timeout=timeout)

    def ready(self):
        ready_ids, _ = ray.wait(
            self._chunk_object_ids,
            num_returns=len(self._chunk_object_ids),
            timeout=0.0)
        return len(ready_ids) == len(self._chunk_object_ids)

    def successful(self):
        if not self.ready():
            raise ValueError("{0!r} not ready".format(self))
        try:
            ray.get(self._chunk_object_ids)
        except Exception:
            return False
        return True


class IMapIterator(object):
    def __init__(self, chunk_object_ids):
        self._chunk_object_ids = chunk_object_ids
        self._ready_objects = []
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()


class OrderedIMapIterator(IMapIterator):
    def next(self, timeout=None):
        if timeout is not None:
            timeout = float(timeout)

        if len(self._ready_objects) != 0:
            return self._ready_objects.pop(0)

        if self._index == len(self._chunk_object_ids):
            raise StopIteration

        try:
            chunk_result = ray.get(
                self._chunk_object_ids[self._index], timeout=timeout)
            result = chunk_result[0]
            self._ready_objects = chunk_result[1:]
        except ray.exceptions.RayTimeoutError:
            raise TimeoutError

        self._index += 1
        return result


class UnorderedIMapIterator(IMapIterator):
    def next(self, timeout=None):
        if timeout is not None:
            timeout = float(timeout)

        if len(self._ready_objects) != 0:
            return self._ready_objects.pop(0)

        if len(self._chunk_object_ids) == 0:
            raise StopIteration

        ready_ids, self._chunk_object_ids = ray.wait(
            self._chunk_object_ids, num_returns=1, timeout=timeout)
        if len(ready_ids) == 0:
            raise TimeoutError

        for ready_id in ready_ids:
            # TODO(edoakes): can we safely set timeout=0 here?
            self._ready_objects.extend(ray.get(ready_id, timeout=timeout))

        return self._ready_objects.pop(0)


@ray.remote
class PoolActor(object):
    def __init__(self, initializer=None, initargs=None):
        if initializer:
            if initargs is None:
                initargs = ()
            initializer(*initargs)

    def ping(self):
        pass

    def run_batch(self, func, batch):
        results = []
        for args, kwargs in batch:
            if args is None:
                args = tuple()
            if kwargs is None:
                kwargs = {}
            results.append(func(*args, **kwargs))
        return results


# https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing.pool
class Pool(object):
    def __init__(self,
                 processes=None,
                 initializer=None,
                 initargs=None,
                 maxtasksperchild=None):
        self._closed = False
        self._initializer = initializer
        self._initargs = initargs
        self._maxtasksperchild = maxtasksperchild if maxtasksperchild else -1
        self._actor_deletion_ids = []

        processes = self._init_ray(processes)
        self._start_actor_pool(processes)

    def _init_ray(self, processes=None):
        if not ray.is_initialized():
            # Cluster mode.
            if "RAY_ADDRESS" in os.environ:
                address = os.environ["RAY_ADDRESS"]
                print("Connecting to ray cluster at address='{}'".format(
                    address))
                ray.init(address=address)
            # Local mode.
            else:
                print("Starting local ray cluster")
                ray.init(num_cpus=processes)

        ray_cpus = int(ray.state.cluster_resources()["CPU"])
        if processes is None:
            processes = ray_cpus
        elif ray_cpus < processes:
            raise ValueError("Tried to start a pool with {} processes on an "
                             "existing ray cluster, but there are only {} "
                             "CPUs in the ray cluster.".format(
                                 processes, ray_cpus))

        return processes

    def _start_actor_pool(self, processes):
        self._actor_pool = [self._new_actor_entry() for _ in range(processes)]
        ray.get([actor.ping.remote() for actor, _ in self._actor_pool])

    def _wait_for_stopping_actors(self, timeout=None):
        if len(self._actor_deletion_ids) == 0:
            return
        if timeout is not None:
            timeout = float(timeout)

        _, deleting = ray.wait(
            self._actor_deletion_ids,
            num_returns=len(self._actor_deletion_ids),
            timeout=timeout)
        self._actor_deletion_ids = deleting

    def _stop_actor(self, actor):
        # Check and clean up any outstanding IDs corresponding to deletions.
        self._wait_for_stopping_actors(timeout=0.0)
        # The deletion task will block until the actor has finished executing
        # all pending tasks.
        self._actor_deletion_ids.append(actor.__ray_terminate__.remote())

    def _new_actor_entry(self):
        # TODO(edoakes): The initializer function can't currently be used to
        # modify the global namespace (e.g., import packages or set globals)
        # due to a limitation in cloudpickle.
        return (PoolActor.remote(self._initializer, self._initargs), 0)

    # Batch should be a list of tuples: (args, kwargs).
    def _run_batch(self, actor_index, func, batch):
        actor, count = self._actor_pool[actor_index]
        object_id = actor.run_batch.remote(func, batch)
        count += 1
        if count == self._maxtasksperchild:
            self._stop_actor(actor)
            actor, count = self._new_actor_entry()
        self._actor_pool[actor_index] = (actor, count)
        return object_id

    def apply(self, func, args=None, kwargs=None):
        return self.apply_async(func, args, kwargs).get()

    def apply_async(self,
                    func,
                    args=None,
                    kwargs=None,
                    callback=None,
                    error_callback=None):
        self._check_running()
        random_actor_index = random.randrange(len(self._actor_pool))
        object_id = self._run_batch(random_actor_index, func, [(args, kwargs)])
        return AsyncResult(
            [object_id], callback, error_callback, single_result=True)

    def map(self, func, iterable, chunksize=None):
        self._check_running()
        return self._map_async(
            func, iterable, chunksize=chunksize, unpack_args=False).get()

    def map_async(self,
                  func,
                  iterable,
                  chunksize=None,
                  callback=None,
                  error_callback=None):
        self._check_running()
        return self._map_async(
            func,
            iterable,
            chunksize=chunksize,
            unpack_args=False,
            callback=callback,
            error_callback=error_callback)

    def starmap(self, func, iterable, chunksize=None):
        self._check_running()
        return self._map_async(
            func, iterable, chunksize=chunksize, unpack_args=True).get()

    def starmap_async(self, func, iterable, callback=None,
                      error_callback=None):
        self._check_running()
        return self._map_async(
            func,
            iterable,
            unpack_args=True,
            callback=callback,
            error_callback=error_callback)

    def _map_async(self,
                   func,
                   iterable,
                   chunksize=None,
                   unpack_args=False,
                   callback=None,
                   error_callback=None):
        object_ids = self._chunk_and_run(func, iterable, chunksize=chunksize)
        return AsyncResult(object_ids, callback, error_callback)

    def _chunk_and_run(self, func, iterable, chunksize=None,
                       unpack_args=False):
        if not hasattr(iterable, "__len__"):
            iterable = [iterable]

        if chunksize is None:
            chunksize, extra = divmod(len(iterable), len(self._actor_pool) * 4)
            if extra:
                chunksize += 1

        chunk_object_ids = []
        chunk = []
        for i, args in enumerate(iterable):
            if not unpack_args:
                args = (args, )
            chunk.append((args, {}))
            if len(chunk) == chunksize or i == len(iterable) - 1:
                actor_index = len(chunk_object_ids) % len(self._actor_pool)
                chunk_object_ids.append(
                    self._run_batch(actor_index, func, chunk))
                chunk = []

        return chunk_object_ids

    # TODO(edoakes): imap and imap_unordered shouldn't submit the full iterable
    # at once, but rather submit a new batch as each one finishes. This is
    # important for very long (or even infinite) iterables.
    def imap(self, func, iterable, chunksize=1):
        self._check_running()
        return OrderedIMapIterator(
            self._chunk_and_run(func, iterable, chunksize=chunksize))

    def imap_unordered(self, func, iterable, chunksize=1):
        self._check_running()
        return UnorderedIMapIterator(
            self._chunk_and_run(func, iterable, chunksize=chunksize))

    def _check_running(self):
        if self._closed:
            raise ValueError("Pool not running")

    def __enter__(self):
        self._check_running()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminate()

    def close(self):
        for actor, _ in self._actor_pool:
            self._stop_actor(actor)
        self._closed = True

    # TODO(edoakes): Terminating a multiprocessing pool doesn't complete
    # oustanding work, but there isn't currently any clean way to do this
    # using the ray API.
    def terminate(self):
        return self.close()

    def join(self):
        if not self._closed:
            raise ValueError("Pool is still running")
        self._wait_for_stopping_actors()
