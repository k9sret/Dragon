# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import pprint
from multiprocessing import Queue

import dragon.core.mpi as mpi

from .data_reader import DataReader
from .data_transformer import DataTransformer
from .blob_fetcher import BlobFetcher


class DataBatch(object):
    """DataBatch aims to prefetch data by ``Triple-Buffering``.

    It takes full advantages of the Process/Thread of Python,
    which provides remarkable I/O speed up for scalable distributed training.

    """
    def __init__(self, **kwargs):
        """Construct a ``DataBatch``.

        Parameters
        ----------
        source : str
            The path of database.
        multiple_nodes: boolean
            Whether to split data for multiple parallel nodes. Default is ``False``.
        shuffle : boolean
            Whether to shuffle the data. Default is ``False``.
        num_chunks : int
            The number of chunks to split. Default is ``2048``.
        chunk_size : int
            The size(MB) of each chunk. Default is -1 (Refer ``num_chunks``).
        mean_values : list
            The mean value of each image channel.
        scale : float
            The scale performed after mean subtraction. Default is ``1.0``.
        padding : int
            The zero-padding size. Default is ``0`` (Disabled).
        fill_value : int
            The value to fill when padding is valid. Default is ``127``.
        crop_size : int
            The crop size. Default is ``0`` (Disabled).
        mirror : boolean
            Whether to flip(horizontally) images. Default is ``False``.
        color_augmentation : boolean
            Whether to distort colors. Default is ``False``.
        min_random_scale : float
            The min scale of the input images. Default is ``1.0``.
        max_random_scale : float
            The max scale of the input images. Default is ``1.0``.
        force_color : boolean
            Set to duplicate channels for gray. Default is ``False``.
        phase : str
            The phase of this operator, ``TRAIN`` or ``TEST``. Default is ``TRAIN``.
        batch_size : int
            The size of a training batch.
        dtype : str
            The data type of batch. Default is ``float32``.
        partition : boolean
            Whether to partition batch. Default is ``False``.
        prefetch : int
            The prefetch count. Default is ``5``.

        """
        super(DataBatch, self).__init__()
        # init mpi
        global_rank = 0; local_rank = 0; group_size = 1
        if mpi.Is_Init():
            idx, group = mpi.AllowParallel()
            if idx != -1:  # data parallel
                global_rank = mpi.Rank()
                group_size = len(group)
                for i, node in enumerate(group):
                    if global_rank == node: local_rank = i
        kwargs['group_size'] = group_size

        # configuration
        self._prefetch = kwargs.get('prefetch', 5)
        self._num_readers = kwargs.get('num_readers', 1)
        self._num_transformers = kwargs.get('num_transformers', -1)
        self._max_transformers = kwargs.get('max_transformers', 3)
        self._num_fetchers = kwargs.get('num_fetchers', 1)

        # io-aware policy
        if self._num_transformers == -1:
            self._num_transformers = 1
            # add 1 transformer for color augmentation
            if kwargs.get('color_augmentation', False):
                self._num_transformers += 1
            # add 1 transformer for random scale
            if kwargs.get('max_random_scale', 1.0) - \
                kwargs.get('min_random_scale', 1.0) != 0:
                    self._num_transformers += 1
            # add 1 transformer for random crop
            if kwargs.get('crop_size', 0) > 0 and \
                kwargs.get('phase', 'TEST') == 'TRAIN':
                    self._num_transformers += 1
        self._num_transformers = min(self._num_transformers, self._max_transformers)

        self._batch_size = kwargs.get('batch_size', 100)
        self._partition = kwargs.get('partition', False)
        if self._partition:
            self._batch_size = int(self._batch_size / kwargs['group_size'])

        # init queues
        self.Q_level_1 = Queue(self._prefetch * self._num_readers * self._batch_size)
        self.Q_level_2 = Queue(self._prefetch * self._num_readers * self._batch_size)
        self.Q_level_3 = Queue(self._prefetch * self._num_readers)

        # init readers
        self._readers = []
        for i in range(self._num_readers):
            self._readers.append(DataReader(**kwargs))
            self._readers[-1].Q_out = self.Q_level_1

        for i in range(self._num_readers):
            num_parts = self._num_readers
            part_idx = i

            if self._readers[i]._multiple_nodes or \
                    self._readers[i]._use_shuffle:
                num_parts *= group_size
                part_idx += local_rank * self._num_readers

            self._readers[i]._num_parts = num_parts
            self._readers[i]._part_idx = part_idx
            self._readers[i]._random_seed += part_idx
            self._readers[i].start()
            time.sleep(0.1)

        # init transformers
        self._transformers = []
        for i in range(self._num_transformers):
            transformer = DataTransformer(**kwargs)
            transformer._random_seed += (i + local_rank * self._num_transformers)
            transformer.Q_in = self.Q_level_1
            transformer.Q_out = self.Q_level_2
            transformer.start()
            self._transformers.append(transformer)
            time.sleep(0.1)

        # init blob fetchers
        self._fetchers = []
        for i in range(self._num_fetchers):
            fetcher = BlobFetcher(**kwargs)
            fetcher.Q_in = self.Q_level_2
            fetcher.Q_out = self.Q_level_3
            fetcher.start()
            self._fetchers.append(fetcher)
            time.sleep(0.1)

        # prevent to echo multiple nodes
        if local_rank == 0: self.echo()
        def cleanup():
            def terminate(processes):
                for process in processes:
                    process.terminate()
                    process.join()
            from dragon.config import logger
            terminate(self._fetchers)
            if local_rank == 0: logger.info('Terminating BlobFetcher ......')
            terminate(self._transformers)
            if local_rank == 0: logger.info('Terminating DataTransformer ......')
            terminate(self._readers)
            if local_rank == 0: logger.info('Terminating DataReader......')
        import atexit
        atexit.register(cleanup)

    def get(self):
        """Get a batch.

        Returns
        -------
        tuple
            The batch, representing data and labels respectively.

        """
        return self.Q_level_3.get()

    def echo(self):
        """Print I/O Information.

        Returns
        -------
        None

        """
        print('---------------------------------------------------------')
        print('BatchFetcher({} Threads), Using config:'.format(
            self._num_readers + self._num_transformers + self._num_fetchers))
        params = {'queue_size': self._prefetch,
                  'n_readers': self._num_readers,
                  'n_transformers': self._num_transformers,
                  'n_fetchers': self._num_fetchers}
        pprint.pprint(params)
        print('---------------------------------------------------------')
