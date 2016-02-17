//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <string>
#include <memory>

namespace Microsoft { namespace MSR { namespace CNTK {

    // Represents a full corpus.
    class CorpusDescriptor
    {
    public:
        CorpusDescriptor()
        {}

        bool IsIncluded(const std::wstring&)
        {
            return true;
        }
    };

    typedef std::shared_ptr<CorpusDescriptor> CorpusDescriptorPtr;
}}}
