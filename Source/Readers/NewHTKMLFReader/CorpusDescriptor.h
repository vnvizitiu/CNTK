//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <string>
#include <memory>
#include <map>

namespace Microsoft { namespace MSR { namespace CNTK {

    class CorpusDescriptor
    {
    public:
        bool IsIncluded(const std::wstring&)
        {
            return true;
        }

        size_t GenerateSequenceId(const std::wstring& key)
        {
            auto it = nameToId_.find(key);
            if (it == nameToId_.end())
            {
                nameToId_.insert(std::make_pair(key, nameToId_.size()));
                return nameToId_.size() - 1;
            }
            return it->second;
        }

    private:
        std::map<std::wstring, size_t> nameToId_;
    };

    typedef std::shared_ptr<CorpusDescriptor> CorpusDescriptorPtr;
}}}
