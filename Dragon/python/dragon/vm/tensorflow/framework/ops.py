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

import re

from dragon.core.tensor import Tensor
from dragon.core.scope import TensorScope, DeviceScope

from dragon.vm.tensorflow.framework import dtypes

_default_graph = None


def convert_to_tensor(value, dtype=None, name=None, **kwargs):
    """Converts the given value to a Tensor.

    Parameters
    ----------
    value : number, list or numpy.ndarray
        The value to convert.
    dtype : Dtype or None
        The data type. If ``None``, inferred from the type of `value`.
    name : str or None
        The Optional name.

    Returns
    -------
    Tensor
        The output tensor.

    """
    if dtype is not None:
        if not isinstance(dtype, str):
            if isinstance(dtype, dtypes.DType):
                dtype = dtype.name
            else:
                raise ValueError('The dtype should be a str of a tf.Dtype.')
    tensor = Tensor(name=name, dtype=dtype)
    tensor.set_value(value)
    return tensor


class Graph(object):
    """
    A virtual graph.
    """
    def __init__(self):
        self._collections = {}

    def get_collection_ref(self, name):
        coll_list = self._collections.get(name, None)
        if coll_list is None:
            coll_list = []
            self._collections[name] = coll_list
        return coll_list

    def get_collection(self, name, scope=None):
        coll_list = self._collections.get(name, None)
        if coll_list is None:
            return []
        if scope is None:
            return list(coll_list)
        else:
            filter_coll_list = []
            regex = re.compile(scope)
            for item in coll_list:
                if hasattr(item, "name") and regex.match(item.name):
                    filter_coll_list.append(item)
            return filter_coll_list

    def add_to_collection(self, name, value):
        if name not in self._collections:
            self._collections[name] = [value]
        else:
            self._collections[name].append(value)

    def add_to_collections(self, names, value):
        for name in names:
            self.add_to_collection(name, value)

    def device(self, device_name_or_function):
        if not isinstance(device_name_or_function, str):
            raise TypeError('The device function should be a str.')
        device_and_id = device_name_or_function.split('/')[1]
        device, id = device_and_id.split(':')
        if device not in ['cpu', 'gpu']:
            raise ValueError('The device should either be cpu or gpu.')
        try:
            id = int(id)
        except Exception as e:
            raise ValueError('The device id should be a integer.')
        return DeviceScope(device, id=id, use_cudnn=True)


def device(device_name_or_function):
  return get_default_graph().device(device_name_or_function)


def get_default_graph():
    global _default_graph
    if _default_graph is None:
        _default_graph = Graph()
    return _default_graph


class GraphKeys(object):
  GLOBAL_VARIABLES = "variables"
  # Key to collect local variables that are local to the machine and are not
  # saved/restored.
  LOCAL_VARIABLES = "local_variables"
  # Key to collect model variables defined by layers.
  MODEL_VARIABLES = "model_variables"
  # Key to collect Variable objects that will be trained by the
  # optimizers.
  TRAINABLE_VARIABLES = "trainable_variables"
  # Key to collect summaries.
  SUMMARIES = "summaries"
  # Key to collect QueueRunners.
  QUEUE_RUNNERS = "queue_runners"
  # Key to collect table initializers.
  TABLE_INITIALIZERS = "table_initializer"
  # Key to collect asset filepaths. An asset represents an external resource
  # like a vocabulary file.
  ASSET_FILEPATHS = "asset_filepaths"
  # Key to collect Variable objects that keep moving averages.
  MOVING_AVERAGE_VARIABLES = "moving_average_variables"
  # Key to collect regularization losses at graph construction.
  REGULARIZATION_LOSSES = "regularization_losses"
  # Key to collect concatenated sharded variables.
  CONCATENATED_VARIABLES = "concatenated_variables"
  # Key to collect savers.
  SAVERS = "savers"
  # Key to collect weights
  WEIGHTS = "weights"
  # Key to collect biases
  BIASES = "biases"
  # Key to collect activations
  ACTIVATIONS = "activations"
  # Key to collect update_ops
  UPDATE_OPS = "update_ops"
  # Key to collect losses
  LOSSES = "losses"
  # Key to collect BaseSaverBuilder.SaveableObject instances for checkpointing.
  SAVEABLE_OBJECTS = "saveable_objects"
  # Key to collect all shared resources used by the graph which need to be
  # initialized once per cluster.
  RESOURCES = "resources"
  # Key to collect all shared resources used in this graph which need to be
  # initialized once per session.
  LOCAL_RESOURCES = "local_resources"
  # Trainable resource-style variables.
  TRAINABLE_RESOURCE_VARIABLES = "trainable_resource_variables"

  # Key to indicate various ops.
  INIT_OP = "init_op"
  LOCAL_INIT_OP = "local_init_op"
  READY_OP = "ready_op"
  READY_FOR_LOCAL_INIT_OP = "ready_for_local_init_op"
  SUMMARY_OP = "summary_op"
  GLOBAL_STEP = "global_step"

  # Used to count the number of evaluations performed during a single evaluation
  # run.
  EVAL_STEP = "eval_step"
  TRAIN_OP = "train_op"

  # Key for control flow context.
  COND_CONTEXT = "cond_context"
  WHILE_CONTEXT = "while_context"

  # Key for streaming model ports.
  # NOTE(yuanbyu): internal and experimental.
  _STREAMING_MODEL_PORTS = "streaming_model_ports"


def get_collection_ref(key):
  return get_default_graph().get_collection_ref(key)


def get_collection(key, scope=None):
    return get_default_graph().get_collection(key, scope)


def add_to_collection(name, value):
    get_default_graph().add_to_collection(name, value)


def add_to_collections(names, value):
    get_default_graph().add_to_collections(names, value)


def name_scope(name, default_name=None, values=None):
    n = default_name if name is None else name
    n = '' if n is None else n
    return TensorScope(prefix=n)
