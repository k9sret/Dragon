#include "core/graph.h"
#include "core/workspace.h"
#include "core/graph_gradient.h"
#include "utils/string.h"
#include "utils/math_functions.h"
#include "utils/proto_utils.h"
#include "operators/control_flow/scan_op.h"
#include "operators/ndarray/slice_op.h"

namespace dragon {

template <class Context>
void ScanOp<Context>::InitTemplate() {
    string func_str = OperatorBase::Arg<string>("func_str", "");
    ParseProtoFromText(func_str, &func_def);
    nrepeats = func_def.op_size();
    OperatorDef slice_def;
    slice_def.set_type("Slice");
    Argument arg_axis, arg_nout;
    arg_axis.set_name("axis"); arg_axis.set_i(axis);
    arg_nout.set_name("num_output"); arg_nout.set_i(1);
    slice_def.add_arg()->CopyFrom(arg_axis);
    slice_def.add_arg()->CopyFrom(arg_nout);
    template_def.mutable_device_option()
        ->CopyFrom(def().device_option());
    //  init for the first step
    for (int i = 0; i < nseqs; i++) {
        OperatorDef* op = template_def.add_op();
        op->CopyFrom(slice_def);
        op->set_name(name() + "(BodyOp." + std::to_string(i) + ")");
        op->add_input(Input(i).name());
        terms[Input(i).name()] = Input(i).name() + "@1";
    }
    for (int i = 0; i < nrepeats; i++) {
        OperatorDef* op = template_def.add_op();
        op->CopyFrom(func_def.op(i));
        op->set_name(name() + "(BodyOp." + std::to_string(i + nseqs) + ")@1");
        //  replace inputs term
        for (int j = 0; j < op->input_size(); j++) {
            string* input = op->mutable_input(j);
            if (terms.count(*input)) *input = terms[*input];
        }
        //  replace outputs term
        for (int j = 0; j < op->output_size(); j++) {
            string* output = op->mutable_output(j);
            terms[*output] = *output + "@1";
            *output = terms[*output];
        }
    }
    //  handle pre outputs
    for (int i = 0; i < nout; i++) {
        if (default_outputs[i].empty()) continue;
        terms[default_outputs[i]] = func_def.target(i) + "@1";
    }
}

template <class Context>
void ScanOp<Context>::UpdateTerms(int cur_step) {
    string prev, now;
    //  update sequences term
    for (int i = 0; i < nseqs; i++) {
        prev = Input(i).name() + "@" + std::to_string(cur_step - 1);
        now = Input(i).name() + "@" + std::to_string(cur_step);
        terms[prev] = now;
    }
    if (cur_step < 3) return;
    //  update recurrent term
    //  only support the latest one-step (as Theano's done)
    for (int i = 0; i < nout; i++) {
        if (default_outputs[i].empty()) continue;
        prev = Output(i)->name() + "@" + std::to_string(cur_step - 2);
        now = Output(i)->name() + "@" + std::to_string(cur_step - 1);
        terms[prev] = now;
    }
}

template <class Context>
void ScanOp<Context>::UnrollTemplate() {
    if (step_type == "Dynamic") {
        CHECK(!step_tensor.empty())
            << "Dynamic nsteps must provide a step tensor.";
        nsteps = ws()->GetTensor(step_tensor)
                     ->template data<int, CPUContext>()[0];
    } else if (step_type == "Default") nsteps = Input(0).dim(axis);
    CHECK_GE(nsteps, 1);
    for (int i = 0; i < nseqs; i++)
        CHECK_EQ(Input(i).dim(axis), nsteps);
    if (graphs.count(nsteps)) return;

    new_def.CopyFrom(template_def);
    new_def.set_name(name() + "(ScanLen." + std::to_string(nsteps) + ")");
    Argument phase; phase.set_name("phase");
    phase.set_s(this->phase()); new_def.add_arg()->CopyFrom(phase);
    for (int idx = 0; idx < nseqs; idx++) {
        OperatorDef *op = new_def.mutable_op(idx);
        int nslices = Input(idx).dim(axis);
        //  alter the num of slices for all sequences
        op->mutable_arg(1)->set_i(nslices);
        //  add slices as outputs
        for (int t = 1; t <= nslices; t++) {
            string slice = op->input(0) + "@" + std::to_string(t);
            op->add_output(slice);
        }
    }
    //  main loop
    for (int t = 2; t <= nsteps; t++) {
        UpdateTerms(t);
        int copy_r = new_def.op_size(), copy_l = copy_r - nrepeats;
        for (int idx = copy_l; idx < copy_r; idx++) {
            OperatorDef* op = new_def.add_op();
            op->CopyFrom(new_def.op(idx));
            op->set_name(str::split(op->name(), "@")[0]
                + "@" + std::to_string(t));
            //  replace inputs
            for (int j = 0; j < op->input_size(); j++) {
                string* input = op->mutable_input(j);
                if (terms.count(*input)) *input = terms[*input];
            }
            //  replace outputs
            for (int j = 0; j < op->output_size(); j++) {
                string* output = op->mutable_output(j);
                terms[*output] = str::split(*output, "@")[0]
                    + "@" + std::to_string(t);
                *output = terms[*output];
            }
        }
    }
    for (int i = 0; i < nout; i++) {
        //  solve the last step only
        new_def.add_target(func_def.target(i) + "@" + std::to_string(nsteps));
        //  concat all steps if necessary
        if (Output(i)->name() == "ignore") continue;
        OperatorDef* op = new_def.add_op();
        op->set_name(name() + "(BodyOp." + std::to_string(nseqs + nrepeats + i) + ")");
        op->set_type("Concat");
        Argument arg_axis, arg_nin;
        arg_axis.set_name("axis"); arg_axis.set_i(axis);
        arg_nin.set_name("num_input"); arg_nin.set_i(nsteps);
        op->add_arg()->CopyFrom(arg_axis);
        op->add_arg()->CopyFrom(arg_nin);
        for (int t = 1; t <= nsteps; t++)
            op->add_input(Output(i)->name() + "@" + std::to_string(t));
        op->add_output(Output(i)->name());
        //  solve all the all steps
        new_def.add_target(Output(i)->name());
    }
    //  upload
    Tensor* string_tensor = ws()->CreateTensor("/mnt/" + anchor() + "/raw_ops");
    string_tensor->Reshape({ 1 });
    string* data = string_tensor->mutable_data <string, CPUContext>();
    data[0] = new_def.SerializeAsString();
}

template <class Context>
void ScanOp<Context>::RunOnDevice() {
    UnrollTemplate();
    if (!graphs.count(nsteps)) {
        graphs[nsteps].reset(new Graph(new_def, ws()));
    }
    cur_graph = graphs[nsteps].get();
    cur_graph->Run("", "");
}

DEPLOY_CPU(Scan);
#ifdef WITH_CUDA
DEPLOY_CUDA(Scan);
#endif
OPERATOR_SCHEMA(Scan).NumInputs(1, INT_MAX).NumOutputs(1, INT_MAX);

template <class Context>
void ScanGradientOp<Context>::MakeOps(const GraphDef& forward_def, 
                                      GraphDef& new_def) {
    if (step_type == "Dynamic")
        nsteps = ws()->GetTensor(step_tensor)
                     ->template data<int, CPUContext>()[0];
    else if (step_type == "Default") nsteps = Input(0).dim(axis);
    if (graphs.count(nsteps)) return;

    //  determine the targets
    vector<string> targets;
    for (auto& t : forward_def.target())
        targets.emplace_back(t);

    //  init maker
    GraphGradientMaker maker;
    maker.SetTerms(terms);
    maker.SetOperatorPrefix(name() + "(BodyOp.");
    maker.SetOperatorSuffix(")");
    for (int i = 0; i < forward_outputs.size(); i++) {
        if (Input(i + (int)OutputSize()).name() != "ignore")
            maker.AddExternalGrad(Input(i + (int)OutputSize()).name());
    }

    //  make
    maker.Make(forward_def, targets, new_def);

    //  post-process
    new_def.set_name(name() + "(ScanLen." + std::to_string(nsteps) + ")");
    for (auto& target : targets) {
        for (int i = 0; i < OutputSize(); i++) {
            if (Output(i)->name() == "ignore") continue;
            if (Input(i).name() == "ignore") continue;
            GradientTarget* g_target = new_def.add_g_target();
            g_target->set_cost(target);
            g_target->set_wrt(Input(i).name());
            g_target->set_external(Output(i)->name());
        }
    }
}

template <class Context>
void ScanGradientOp<Context>::RunOnDevice() {
    Tensor* ops = ws()->GetTensor("/mnt/" + anchor() + "/raw_ops");
    GraphDef forward_def, new_def;
    forward_def.ParseFromString(ops->data<string, CPUContext>()[0]);
    new_def.CopyFrom(forward_def);
    MakeOps(forward_def, new_def);
    
    //  persist for different scan steps
    if (!graphs.count(nsteps)) 
        graphs[nsteps].reset(new Graph(new_def, ws()));
    cur_graph = graphs[nsteps].get();
    cur_graph->Run("Gradient", "");
}

DEPLOY_CPU(ScanGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(ScanGradient);
#endif
OPERATOR_SCHEMA(ScanGradient).NumInputs(2, INT_MAX).NumOutputs(1, INT_MAX);

class GetScanGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetScanGradient);
    vector<OperatorDef> MakeDefs() override {
        vector<string> inputs, outputs;
        for (int i = 0; i < def.input_size(); i++) {
            inputs.push_back(I(i));
            outputs.push_back(GI(i));
        }
        for (int i = 0; i < def.output_size(); i++) {
            inputs.push_back(GO(i));
        }
        return SingleDef(def.type() + "Gradient", "", inputs, outputs);
    }
};
REGISTER_GRADIENT(Scan, GetScanGradient);

}    // namespace dragon