//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <string>
#include <memory>
#include "ConfigHelper.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// Represents a full corpus.
class CorpusDescriptor
{
public:
    CorpusDescriptor(std::vector<std::wstring>&& sequences) : m_sequences(sequences)
    {
    }

    bool IsIncluded(const std::wstring&)
    {
        return true;
    }

private:
    std::vector<std::wstring> m_sequences;
};

typedef std::shared_ptr<CorpusDescriptor> CorpusDescriptorPtr;

}}}
