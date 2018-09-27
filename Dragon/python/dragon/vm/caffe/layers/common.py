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

import dragon.ops as ops
from dragon.core.tensor import Tensor

from ..layer import Layer


class InnerProductLayer(Layer):
    """The implementation of ``InnerProductLayer``.

    Parameters
    ----------
    num_output : int
         The output dim. Refer `InnerProductParameter.num_output`_.
    bias_term : boolean
         Whether to use bias. Refer `InnerProductParameter.bias_term`_.
    weight_filler : caffe_pb2.FillerParameter
         The filler of weight. Refer `InnerProductParameter.weight_filler`_.
    bias_filler : caffe_pb2.FillerParameter
         The filler of bias. Refer `InnerProductParameter.bias_filler`_.
    axis : int
        The start axis to calculate. Refer `InnerProductParameter.axis`_.
    transpose : boolean
        Whether to transpose the weights. Refer `InnerProductParameter.transpose`_.

    """
    def __init__(self, LayerParameter):
        super(InnerProductLayer, self).__init__(LayerParameter)
        param = LayerParameter.inner_product_param
        self._param = {'axis': param.axis,
                       'num_output': param.num_output,
                       'TransW': not param.transpose}
        scope = LayerParameter.name
        weight = Tensor(scope + '/param:0')
        weight_diff = Tensor(scope + '/param:0_grad')
        self.Fill(weight, param, 'weight_filler')
        self._blobs.append({'data': weight, 'diff': weight_diff})

        if param.bias_term:
            bias = Tensor(scope + '/param:1')
            bias_diff = Tensor(scope + '/param:1_grad')
            self.Fill(bias, param, 'bias_filler')
            self._blobs.append({'data': bias, 'diff': bias_diff})

    def Setup(self, bottom):
        super(InnerProductLayer, self).Setup(bottom)
        return ops.InnerProduct(bottom + [blob['data'] for blob in self._blobs], **self._param)


class AccuracyLayer(Layer):
    """The implementation of ``AccuracyLayer``.

    Parameters
    ----------
    top_k : int
        The top-k accuracy. Refer `AccuracyParameter.top_k`_.
    axis : int
        The axis of classes. Refer `AccuracyParameter.axis`_.
    ignore_label : int
        The label to ignore. Refer `AccuracyParameter.ignore_label`_.

    """
    def __init__(self, LayerParameter):
        super(AccuracyLayer, self).__init__(LayerParameter)
        param = LayerParameter.accuracy_param
        self._param = {'top_k': param.top_k,
                       'ignore_labels': [param.ignore_label]
                            if param.HasField('ignore_label') else []}

    def Setup(self, bottom):
        super(AccuracyLayer, self).Setup(bottom)
        return ops.Accuracy(bottom, **self._param)


class PythonLayer(Layer):
    """The implementation of ``PythonLayer``.

    Parameters
    ----------
    module : str
        The module. Refer `PythonParameter.module`_.
    layer : str
        The class name of layer. Refer `PythonParameter.layer`_.
    param_str : str
        The str describing parameters. Refer `PythonParameter.param_str`_.

    """
    def __init__(self, LayerParameter):
        super(PythonLayer, self).__init__(LayerParameter)
        param = LayerParameter.python_param
        self._param = {'module': param.module,
                       'op': param.layer,
                       'param_str': param.param_str}

    def Setup(self, bottom):
        super(PythonLayer, self).Setup(bottom)
        return ops.Run(bottom, nout=len(self._top), **self._param)


class EltwiseLayer(Layer):
    """The implementation of ``EltwiseLayer``.

    Parameters
    ----------
    operation : EltwiseParameter.EltwiseOp
        The operation. Refer `EltwiseParameter.operation`_.
    coeff : list of float
        The coefficients. Refer `EltwiseParameter.coeff`_.

    """
    def __init__(self, LayerParameter):
        super(EltwiseLayer, self).__init__(LayerParameter)
        param = LayerParameter.eltwise_param
        self._param = {'operation': {0: 'PROD', 1: 'SUM', 2: 'MAX'}[param.operation],
                       'coeffs': [element for element in param.coeff]
                            if len(param.coeff) > 0 else None}

    def Setup(self, bottom):
        super(EltwiseLayer, self).Setup(bottom)
        return ops.Eltwise(bottom, **self._param)


class AddLayer(Layer):
    """
    The extended implementation of ``EltwiseLayer``.
    """
    def __init__(self, LayerParameter):
        super(AddLayer, self).__init__(LayerParameter)

    def Setup(self, bottom):
        super(AddLayer, self).Setup(bottom)
        return ops.Add(bottom, **self._param)


class ConcatLayer(Layer):
    """The implementation of ``ConcatLayer``.

    Parameters
    ----------
    axis : int
        The axis to concatenate. Refer `ConcatParameter.axis`_.

    """
    def __init__(self, LayerParameter):
        super(ConcatLayer, self).__init__(LayerParameter)
        param = LayerParameter.concat_param
        self._param = {'axis': param.axis}

    def Setup(self, bottom):
        super(ConcatLayer, self).Setup(bottom)
        return ops.Concat(bottom, **self._param)


class DenseConcatLayer(Layer):
    """The extended implementation for `DenseNet`_.

    Parameters
    ----------
    axis : int
        The axis to concatenate. Refer `ConcatParameter.axis`_.
    growth_rate : int
        The growth rate.

    """
    def __init__(self, LayerParameter):
        super(DenseConcatLayer, self).__init__(LayerParameter)
        param = LayerParameter.dense_concat_param
        self._param = {'axis': param.axis,
                       'growth_rate': param.growth_rate}

    def Setup(self, bottom):
        super(DenseConcatLayer, self).Setup(bottom)
        return ops.DenseConcat(bottom, **self._param)


class CropLayer(Layer):
    """The implementation of ``CropLayer``.

    Parameters
    ----------
    axis : int
        The start axis. Refer `CropParameter.axis`_.
    offset : list of int
        The offsets. Refer `CropParameter.offset`_.

    """
    def __init__(self, LayerParameter):
        super(CropLayer, self).__init__(LayerParameter)
        param = LayerParameter.crop_param
        self._param = {'start_axis': param.axis,
                       'offsets': [int(element) for element in param.offset]}

    def Setup(self, bottom):
        super(CropLayer, self).Setup(bottom)
        self._param['shape_like'] = bottom[1]
        self._param['starts'] = self._param['ends'] = None
        return ops.Crop(bottom[0], **self._param)


class ReshapeLayer(Layer):
    """The implementation of ``ReshapeLayer``.

    Parameters
    ----------
    shape : list of int
        The output shape. Refer `ReshapeParameter.shape`_.

    """
    def __init__(self, LayerParameter):
        super(ReshapeLayer, self).__init__(LayerParameter)
        param = LayerParameter.reshape_param
        shape = param.shape
        self._param = {'shape': [int(element) for element in shape.dim]}

    def Setup(self, bottom):
        super(ReshapeLayer, self).Setup(bottom)
        input = bottom[0] if isinstance(bottom, list) else bottom
        return ops.Reshape(input, **self._param)


class PermuteLayer(Layer):
    """The implementation of ``PermuteLayer``.

    Parameters
    ----------
    order : list of int
        The permutation. Refer `PermuteParameter.order`_.

    """
    def __init__(self, LayerParameter):
        super(PermuteLayer, self).__init__(LayerParameter)
        param = LayerParameter.permute_param
        self._param = {'perms': [int(element) for element in param.order]}

    def Setup(self, bottom):
        super(PermuteLayer, self).Setup(bottom)
        input = bottom[0] if isinstance(bottom, list) else bottom
        return ops.Transpose(input, **self._param)


class FlattenLayer(Layer):
    """The implementation of ``FlattenLayer``.

    Parameters
    ----------
    axis : int
        The start axis. Refer `FlattenParameter.axis`_.
    end_axis : int
        The end axis. Refer `FlattenParameter.end_axis`_.

    """
    def __init__(self, LayerParameter):
        super(FlattenLayer, self).__init__(LayerParameter)
        param = LayerParameter.flatten_param
        axis = param.axis; end_axis = param.end_axis
        num_axes =  end_axis - axis + 1 if end_axis != -1 else -1
        self._param = {'axis': axis, 'num_axes': num_axes}


    def Setup(self, bottom):
        super(FlattenLayer, self).Setup(bottom)
        input = bottom[0] if isinstance(bottom, list) else bottom
        return ops.Flatten(input, **self._param)


class GatherLayer(Layer):
    """The extended implementation of ``GatherOp``.

    Parameters
    ----------
    axis : int
        The axis for gathering. Refer ``GatherParameter.axis``.

    """
    def __init__(self, LayerParameter):
        super(GatherLayer, self).__init__(LayerParameter)
        param = LayerParameter.gather_param
        self._param = {'axis': param.axis}

    def Setup(self, bottom):
        super(GatherLayer, self).Setup(bottom)
        return ops.Gather(bottom[0], indices=bottom[1], **self._param)


class SoftmaxLayer(Layer):
    """The implementation of ``SoftmaxLayer``.

    Parameters
    ----------
    axis : int
        The axis to perform softmax. Refer `SoftmaxParameter.axis`_.

    """
    def __init__(self, LayerParameter):
        super(SoftmaxLayer, self).__init__(LayerParameter)
        param = LayerParameter.softmax_param
        self._param = {'axis': param.axis}

    def Setup(self, bottom):
        super(SoftmaxLayer, self).Setup(bottom)
        input = bottom[0] if isinstance(bottom, list) else bottom
        return ops.Softmax(input, **self._param)


class ArgMaxLayer(Layer):
    """The implementation of ``ArgMaxLayer``.

    Parameters
    ----------
    top_k : int
        The top k results to keep. Refer `ArgMaxParameter.top_k`_.
    axis : int
        The axis to perform argmax. Refer `ArgMaxParameter.axis`_.

    """
    def __init__(self, LayerParameter):
        super(ArgMaxLayer, self).__init__(LayerParameter)
        param = LayerParameter.argmax_param
        self._param = {'top_k': param.top_k,
                       'axis': param.axis,
                       'keep_dims': True}

    def Setup(self, bottom):
        super(ArgMaxLayer, self).Setup(bottom)
        input = bottom[0] if isinstance(bottom, list) else bottom
        return ops.Argmax(input, **self._param)


class BatchNormLayer(Layer):
    """The implementation of ``BatchNormLayer``.

    Parameters
    ----------
    use_global_stats : boolean
        Refer `BatchNormParameter.use_global_stats`_.
    moving_average_fraction : float
        Refer `BatchNormParameter.moving_average_fraction`_.
    eps : float
        Refer `BatchNormParameter.eps`_.

    """
    def __init__(self, LayerParameter):
        super(BatchNormLayer, self).__init__(LayerParameter)
        param = LayerParameter.batch_norm_param
        self._param = {'use_stats': int(param.use_global_stats)
                            if param.HasField('use_global_stats') else -1,
                       'momentum': param.moving_average_fraction,
                       'eps': param.eps,
                       'axis': 1,
                       'mode': 'CAFFE'}
        scope = LayerParameter.name
        # mean, var, factor are set to 0 in order to do statistics
        mean = Tensor(scope + '/param:0').Constant(value=0.0)
        var  = Tensor(scope + '/param:1').Constant(value=0.0)
        factor = Tensor(scope + '/param:2').Constant(value=0.0)
        # in dragon, set diff as None will ignore computing grad automatically
        # but in bvlc-caffe, you must set lr_mult = 0 manually
        self._blobs.append({'data': mean, 'diff': None})
        self._blobs.append({'data': var, 'diff': None})
        self._blobs.append({'data': factor, 'diff': None})

    def Setup(self, bottom):
        super(BatchNormLayer, self).Setup(bottom)
        return ops.BatchNorm(bottom + [blob['data'] for blob in self._blobs], **self._param)


class BatchRenormLayer(Layer):
    """The implementation of ``BatchRenormLayer``.

    Parameters
    ----------
    use_global_stats : boolean
        Refer ``BatchRenormParameter.use_global_stats``.
    moving_average_fraction : float
        Refer ``BatchRenormParameter.moving_average_fraction``.
    eps : float
        Refer ``BatchRenormParameter.eps``.
    r_max : float
        Refer ``BatchRenormParameter.r_max``.
    d_max : float
        Refer ``BatchRenormParameter.d_max``.
    t_delta : float
        Refer ``BatchRenormParameter.t_delta``.

    """
    def __init__(self, LayerParameter):
        super(BatchRenormLayer, self).__init__(LayerParameter)
        param = LayerParameter.batch_renorm_param
        self._param = {'use_stats': int(param.use_global_stats)
                            if param.HasField('use_global_stats') else -1,
                       'momentum': param.moving_average_fraction,
                       'eps': param.eps,
                       'r_max': float(param.r_max),
                       'd_max': float(param.d_max),
                       't_delta': float(param.t_delta),
                       'axis': 1,
                       'mode': 'CAFFE'}
        scope = LayerParameter.name
        mean = Tensor(scope + '/param:0').Constant(value=0.0)
        var  = Tensor(scope + '/param:1').Constant(value=0.0)
        factor = Tensor(scope + '/param:2').Constant(value=0.0)
        self._blobs.append({'data': mean, 'diff': None})
        self._blobs.append({'data': var, 'diff': None})
        self._blobs.append({'data': factor, 'diff': None})

    def Setup(self, bottom):
        super(BatchRenormLayer, self).Setup(bottom)
        return ops.BatchRenorm(bottom + [blob['data'] for blob in self._blobs], **self._param)


class GroupNormLayer(Layer):
    """The implementation of ``GroupNormLayer``.

    Parameters
    ----------
    group : int
        Refer ``GroupNormParameter.group``.
    eps : float
        Refer ``GroupNormParameter.eps``.

    """
    def __init__(self, LayerParameter):
        super(GroupNormLayer, self).__init__(LayerParameter)
        param = LayerParameter.group_norm_param
        self._param = {'group': int(param.group),
                       'eps': param.eps,
                       'axis': 1}

    def Setup(self, bottom):
        super(GroupNormLayer, self).Setup(bottom)
        return ops.GroupNorm(bottom[0], **self._param)


class InstanceNormLayer(Layer):
    """
    The implementation of ``InstanceNormLayer``.

    Introduced by `[Ulyanov et.al, 2016] <https://arxiv.org/abs/1607.08022>`_

    """
    def __init__(self, LayerParameter):
        super(InstanceNormLayer, self).__init__(LayerParameter)
        param = LayerParameter.instance_norm_param
        self._param = {'eps': param.eps,
                       'axis': 1}

    def Setup(self, bottom):
        super(InstanceNormLayer, self).Setup(bottom)
        return ops.InstanceNorm(bottom[0], **self._param)


class ScaleLayer(Layer):
    """The implementation of ``ScaleLayer``.

    Parameters
    ----------
    axis : int
        The start axis. Refer `ScaleParameter.axis`_.
    num_axes : int
        The number of axes. Refer `ScaleParameter.num_axes`_.
    filler : FillerParameter
        The filler of scale parameter. Refer `ScaleParameter.filler`_.
    bias_term : boolean
        Whether to use bias. Refer `ScaleParameter.bias_term`_.
    bias_filler : FillerParameter
        The filler of bias parameter. Refer `ScaleParameter.bias_filler`_.

    """
    def __init__(self, LayerParameter):
        super(ScaleLayer, self).__init__(LayerParameter)
        param = LayerParameter.scale_param
        self._param = {'axis': param.axis,
                       'num_axes': param.num_axes}
        scope = LayerParameter.name
        scale = Tensor(scope + '/param:0')
        scale_diff = Tensor(scope + '/param:0_grad')
        if param.HasField('filler'):
            self.Fill(scale, param, 'filler')
        else: scale.Constant(value=1.0)
        self._blobs.append({'data': scale, 'diff': scale_diff})
        if param.bias_term:
            bias = Tensor(scope + '/param:1')
            bias_diff = Tensor(scope + '/param:1_grad')
            # auto fill 0 if not specficed bias_filler
            self.Fill(bias, param, 'bias_filler')
            self._blobs.append({'data': bias, 'diff': bias_diff})

    def Setup(self, bottom):
        super(ScaleLayer, self).Setup(bottom)
        return ops.Affine(bottom + [blob['data'] for blob in self._blobs], **self._param)


class BNLayer(Layer):
    """The implementation of ``BNLayer``.

    Parameters
    ----------
    use_global_stats : boolean
        Refer `BatchNormParameter.use_global_stats`_.
    moving_average_fraction : float
        Refer `BatchNormParameter.moving_average_fraction`_.
    eps : float
        Refer `BatchNormParameter.eps`_.
    filler : FillerParameter
        The filler of scale parameter. Refer `ScaleParameter.filler`_.
    bias_filler : FillerParameter
        The filler of bias parameter. Refer `ScaleParameter.bias_filler`_.

    """
    def __init__(self, LayerParameter):
        super(BNLayer, self).__init__(LayerParameter)
        bn_param = LayerParameter.batch_norm_param
        scale_param = LayerParameter.scale_param
        self._param = {'use_stats': int(bn_param.use_global_stats)
                                        if bn_param.HasField('use_global_stats') else -1,
                       'momentum': bn_param.moving_average_fraction,
                       'eps': bn_param.eps,
                       'axis': 1}
        scope = LayerParameter.name
        mean = Tensor(scope + '/param:0').Constant(value=0.0)
        var = Tensor(scope + '/param:1').Constant(value=0.0)
        scale = Tensor(scope + '/param:2')
        scale_diff = Tensor(scope + '/param:2_grad')
        bias = Tensor(scope + '/param:3')
        bias_diff = Tensor(scope + '/param:3_grad')

        if scale_param.HasField('filler'):
            self.Fill(scale, scale_param, 'filler')
        else: scale.Constant(value=1.0)
        self.Fill(bias, scale_param, 'bias_filler')
        self.norm_blobs = [{'data': mean, 'diff': None},
                           {'data': var, 'diff': None}]
        self.scale_blobs = [{'data': scale, 'diff': scale_diff},
                            {'data': bias, 'diff': bias_diff}]
        self._blobs.extend(self.norm_blobs)
        self._blobs.extend(self.scale_blobs)

    def Setup(self, bottom):
        super(BNLayer, self).Setup(bottom)
        return ops.FusedBatchNorm(bottom + [blob['data'] for blob in self._blobs], **self._param)


class GNLayer(Layer):
    """The implementation of ``GNLayer``.

    Parameters
    ----------
    group : int
        Refer ``GroupNormParameter.group``.
    eps : float
        Refer ``GroupNormParameter.eps``.
    filler : FillerParameter
        The filler of scale parameter. Refer `ScaleParameter.filler`_.
    bias_filler : FillerParameter
        The filler of bias parameter. Refer `ScaleParameter.bias_filler`_.

    """
    def __init__(self, LayerParameter):
        super(GNLayer, self).__init__(LayerParameter)
        gn_param = LayerParameter.group_norm_param
        scale_param = LayerParameter.scale_param
        self._param = {'group': int(gn_param.group),
                       'eps': gn_param.eps,
                       'axis': 1}
        scope = LayerParameter.name
        scale = Tensor(scope + '/param:0')
        scale_diff = Tensor(scope + '/param:0_grad')
        bias = Tensor(scope + '/param:1')
        bias_diff = Tensor(scope + '/param:1_grad')

        if scale_param.HasField('filler'):
            self.Fill(scale, scale_param, 'filler')
        else: scale.Constant(value=1.0)
        self.Fill(bias, scale_param, 'bias_filler')
        self.scale_blobs = [{'data': scale, 'diff': scale_diff},
                            {'data': bias, 'diff': bias_diff}]
        self._blobs.extend(self.scale_blobs)

    def Setup(self, bottom):
        super(GNLayer, self).Setup(bottom)
        return ops.FusedGroupNorm(bottom + [blob['data'] for blob in self._blobs], **self._param)


class NormalizeLayer(Layer):
    """The implementation of ``NormalizeLayer``.

    Parameters
    ----------
    across_spatial : boolean
        Whether to stat spatially. Refer `NormalizeParameter.across_spatial`_.
    scale_filler : FillerParameter
        The filler of scale parameter. Refer `NormalizeParameter.scale_filler`_.
    channel_shared : boolean
        Whether to scale across channels. Refer `NormalizeParameter.channel_shared`_.
    eps : float
        The eps. Refer `NormalizeParameter.eps`_.

    """
    def __init__(self, LayerParameter):
        super(NormalizeLayer, self).__init__(LayerParameter)
        param = LayerParameter.normalize_param
        self._l2norm_param = {
            'axis': 1,
            'num_axes': -1 if param.across_spatial else 1,
            'eps': param.eps}
        self._scale_param = {
            'axis': 1,
            'num_axes': 0 if param.channel_shared else 1}
        scope = LayerParameter.name
        scale = Tensor(scope + '/param:0')
        if param.HasField('scale_filler'):
            self.Fill(scale, param, 'scale_filler')
        else: scale.Constant(value=1.0)
        self.scale_blobs = [{'data': scale, 'diff': Tensor(scale.name + '_grad')}]
        self._blobs.extend(self.scale_blobs)

    def Setup(self, bottom):
        super(NormalizeLayer, self).Setup(bottom)
        norm_out = [ops.L2Norm(bottom[0], **self._l2norm_param)]
        scale_out = ops.Affine(norm_out + [blob['data'] for blob in self.scale_blobs],
                               **self._scale_param)
        return scale_out


class TileLayer(Layer):
    """The extended implementation of ``TileLayer``.

    Parameters
    ----------
    multiples : caffe_pb2.BlobShape
        The multiples. Refer `TileParameter.multiples`_.

    """
    def __init__(self, LayerParameter):
        super(TileLayer, self).__init__(LayerParameter)
        param = LayerParameter.tile_param
        multiples = param.multiples
        self._param = {'multiples': [int(multiple) for multiple in multiples.dim]}

    def Setup(self, bottom):
        super(TileLayer, self).Setup(bottom)
        input = bottom[0] if isinstance(bottom, list) else bottom
        return ops.Tile(input, **self._param)


class ReductionLayer(Layer):
    """The extended implementation of ``ReductionLayer``.

    Parameters
    ----------
    operation : caffe_pb2.ReductionOp
        The operation. Refer `ReductionParameter.operation`_.
    axis : int
        The axis to to reduce. Refer `ReductionParameter.axis`_.

    """
    def __init__(self, LayerParameter):
        super(ReductionLayer, self).__init__(LayerParameter)
        param = LayerParameter.reduction_param
        if param.axis < 0:
            if param.axis != -1:
                raise ValueError('The negative axis can only be -1(reduce all).')
        self._param = {'operation': {1: 'SUM', 4: 'MEAN'}[param.operation],
                       'axis': param.axis}

    def Setup(self, bottom):
        super(ReductionLayer, self).Setup(bottom)
        input = bottom[0] if isinstance(bottom, list) else bottom
        return ops.Reduce(input, **self._param)


class ExpandDimsLayer(Layer):
    """The implementation of ``ExpandDimsLayer``.

    Parameters
    ----------
    axis : int
        This axis to expand at. Refer `ExpandDimsParameter.axis`_.

    """
    def __init__(self, LayerParameter):
        super(ExpandDimsLayer, self).__init__(LayerParameter)
        param = LayerParameter.expand_dims_param
        self._param = {'axis': param.axis}

    def Setup(self, bottom):
        super(ExpandDimsLayer, self).Setup(bottom)
        input = bottom[0] if isinstance(bottom, list) else bottom
        return ops.ExpandDims(input, **self._param)


class StopGradientLayer(Layer):
    """
    The implementation of ``StopGradientLayer``.

    """
    def __init__(self, LayerParameter):
        super(StopGradientLayer, self).__init__(LayerParameter)

    def Setup(self, bottom):
        super(StopGradientLayer, self).Setup(bottom)
        input = bottom[0] if isinstance(bottom, list) else bottom
        return ops.StopGradient(input, **self._param)


class ProposalLayer(Layer):
    """The implementation of ``ProposalLayer``.

    Parameters
    ----------
    stride : list of int
        The stride of anchors. Refer ``ProposalParameter.stride``.
    scale : list of float
        The scales of anchors. Refer `ProposalParameter.scale`_.
    ratio : list of float
        The ratios of anchors. Refer `ProposalParameter.ratio`_.
    pre_nms_top_n : int
        The num of anchors before nms. Refer `ProposalParameter.pre_nms_topn`_.
    post_nms_top_n : int
        The num of anchors after nms. Refer `ProposalParameter.post_nms_topn`_.
    nms_thresh : float
        The threshold of nms. Refer `ProposalParameter.nms_thresh`_.
    min_size : int
        The min size of anchors. Refer `ProposalParameter.min_size`_.
    min_level : int
        Finest level of the FPN pyramid. Refer ``ProposalParameter.min_level``.
    max_level : int
        Coarsest level of the FPN pyramid. Refer ``ProposalParameter.max_level``.
    canonical_scale : int
        The baseline scale of mapping policy. Refer ``ProposalParameter.canonical_scale``.
    canonical_level : int
        Heuristic level of the canonical scale. Refer ``ProposalParameter.canonical_level``.

    """
    def __init__(self, LayerParameter):
        super(ProposalLayer, self).__init__(LayerParameter)
        param = LayerParameter.proposal_param
        self._param = {'strides': param.stride,
                       'ratios': param.ratio,
                       'scales': param.scale,
                       'pre_nms_top_n': param.pre_nms_top_n,
                       'post_nms_top_n': param.post_nms_top_n,
                       'nms_thresh': param.nms_thresh,
                       'min_size': param.min_size,
                       'min_level': param.min_level,
                       'max_level': param.max_level,
                       'canonical_scale': param.canonical_scale,
                       'canonical_level': param.canonical_level}

    def Setup(self, bottom):
        super(ProposalLayer, self).Setup(bottom)
        return ops.Proposal(bottom, **self._param)
