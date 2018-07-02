// ------------------------------------------------------------
// Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// -------------------------------------------------------------

#ifndef DRAGON_OPERATORS_NDARRAY_STACK_OP_H_
#define DRAGON_OPERATORS_NDARRAY_STACK_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class StackOp final : public Operator<Context> {
 public:
    StackOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis(OperatorBase::Arg<int>("axis", 0)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex axis, outer_dim, inner_dim;
    TIndex x_concat_dim, y_concat_dim;
    TIndex x_offset, y_offset, concat_offset;
    vector<TIndex> stack_dims, concat_dims;
};

template <class Context>
class StackGradientOp final : public Operator<Context> {
 public:
    StackGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis(OperatorBase::Arg<int>("axis", 0)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex axis, outer_dim, inner_dim;
    TIndex x_concat_dim, y_concat_dim;
    TIndex x_offset, y_offset, concat_offset;
    vector<TIndex> concat_dims;
}; 

}    // namespace dragon

#endif    // DRAGON_OPERATORS_NDARRAY_STACK_OP_H_