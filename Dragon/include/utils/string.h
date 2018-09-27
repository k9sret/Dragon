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

#ifndef DRAGON_UTILS_STRING_H_
#define DRAGON_UTILS_STRING_H_

#include <string>
#include <sstream>
#include <vector>
#include <iostream>
#include <cstdlib>

namespace dragon {

namespace str {

inline std::vector<std::string> split(
    const std::string&              str,
    const std::string&              c) {
    std::vector<std::string> ret;
    std::string temp(str);
    size_t pos;
    while (pos = temp.find(c), pos != std::string::npos) {
        ret.push_back(temp.substr(0, pos));
        temp.erase(0, pos + 1);
    }
    ret.push_back(temp);
    return ret;
}

}    // namespace str

}    // namespace dragon

#endif    // DRAGON_UTILS_STRING_H_