#ifdef WITH_CUDNN

#include "operators/activation/softmax_op.h"

namespace dragon {

template <class Context> template <typename T>
void CuDNNSoftmaxOp<Context>::RunWithType() {
    Tensor fake_tensor(vector<TIndex>(
        { outer_dim, Input(0).dim(axis), inner_dim }));
    cudnnSetTensorDesc<T>(&input_desc, &fake_tensor);
    cudnnSetTensorDesc<T>(&output_desc, &fake_tensor);

    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    CUDNN_CHECK(cudnnSoftmaxForward(ctx()->cudnn_handle(),
        CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
            CUDNNType<T>::one, input_desc, Xdata,
                CUDNNType<T>::zero, output_desc, Ydata));
}

template <class Context>
void CuDNNSoftmaxOp<Context>::RunOnDevice() {
    if (axis == -1) axis = (TIndex)Input(0).ndim() - 1;
    outer_dim = Input(0).count(0, axis);
    inner_dim = Input(0).count(axis + 1);
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
#ifdef WITH_CUDA_FP16
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
#endif
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CUDNN(Softmax);

template <class Context> template <typename T>
void CuDNNSoftmaxGradientOp<Context>::RunWithType() {
    Tensor fake_tensor(vector<TIndex>(
        { outer_dim, Input(0).dim(axis), inner_dim }));
    cudnnSetTensorDesc<T>(&input_desc, &fake_tensor);
    cudnnSetTensorDesc<T>(&output_desc, &fake_tensor);

    auto* dYdata = Input(-1).template data<T, Context>();
    auto* Ydata = Input(0).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    CUDNN_CHECK(cudnnSoftmaxBackward(ctx()->cudnn_handle(),
        CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
            CUDNNType<T>::one, input_desc, Ydata, input_desc, dYdata,
                CUDNNType<T>::zero, output_desc, dXdata));
}

template <class Context>
void CuDNNSoftmaxGradientOp<Context>::RunOnDevice() {
    if (axis == -1) axis = (TIndex)Input(0).ndim() - 1;
    outer_dim = Input(0).count(0, axis);
    inner_dim = Input(0).count(axis + 1);
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
#ifdef WITH_CUDA_FP16
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
#endif
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CUDNN(SoftmaxGradient);

}    // namespace dragon

#endif    // WITH_CUDNN