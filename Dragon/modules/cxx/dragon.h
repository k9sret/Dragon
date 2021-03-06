// ------------------------------------------------------------
// Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// ------------------------------------------------------------

#ifndef DRAGON_CXX_DRAGON_H_
#define DRAGON_CXX_DRAGON_H_

#include <string>
#include <cstdint>
#include <vector>

#ifdef WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT
#endif

namespace dragon {

typedef int64_t TIndex;

class Workspace;

class Device {
 public:
    EXPORT Device();
    EXPORT explicit Device(std::string device_type);
    EXPORT Device(std::string device_type, int device_id);

    EXPORT const int& device_type() const { return device_type_; }
    EXPORT const int device_id() const { return device_id_; }

 private:
     int device_type_, device_id_;
};

EXPORT Workspace* CreateWorkspace(const std::string& name);

EXPORT Workspace* ResetWorkspace(const std::string& name);

EXPORT void ReleaseWorkspace(const std::string& name);

EXPORT void MoveWorkspace(Workspace* main, Workspace* sub);

EXPORT std::string CreateGraph(
    const std::string&          graph_file,
    Workspace*                  ws);

EXPORT std::string CreateGraph(
    const std::string&          graph_file,
    const Device&               device,
    Workspace*                  ws);

EXPORT void RunGraph(
    const std::string&          graph_name,
    Workspace*                  ws,
    const int                   stream_id = 1);

EXPORT void CreateTensor(
    const std::string&          name,
    Workspace*                  ws);

template <typename T>
EXPORT void FeedTensor(
    const std::string&          name,
    const std::vector<TIndex>&  shape,
    const T*                    data,
    const Device&               device,
    Workspace*                  ws);

template <typename T>
EXPORT T* FetchTensor(
    const std::string&          name,
    std::vector<TIndex>&        shape,
    Workspace*                  ws);

EXPORT void LoadCaffemodel(
    const std::string&          model_file,
    Workspace*                  ws);

EXPORT void TransplantCaffeModel(
    const std::string&          input_model,
    const std::string&          output_model);

EXPORT void LoadDragonmodel(
    const std::string&          model_file,
    Workspace*                  ws);

EXPORT void SetLogLevel(const std::string& level);

}    // namespace dragon

#endif    // DRAGON_CXX_DRAGON_H_