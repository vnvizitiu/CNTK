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
        CorpusDescriptor()
        {}

        bool IsIncluded(const std::wstring&)
        {
            return true;
        }

        size_t GetId(const std::wstring& sequence, size_t sample)
        {
            auto s = m_sequenceMap.find(sequence);
            size_t sequenceId = 0;
            if (s == m_sequenceMap.end())
            {
                sequenceId = m_sequenceMap.size();
                m_sequenceMap.insert(std::make_pair(sequence, sequenceId));
            }
            else
            {
                sequenceId = s->second;
            }

            auto it = m_ids.find(std::make_pair(sequenceId, sample));
            if (it == m_ids.end())
            {
                m_ids.insert(std::make_pair(std::make_pair(sequenceId, sample), m_ids.size()));
                return m_ids.size() - 1;
            }

            return it->second;
        }

    private:
        std::map<std::wstring, size_t> m_sequenceMap;
        std::map<std::pair<size_t, size_t>, size_t> m_ids;
    };

    typedef std::shared_ptr<CorpusDescriptor> CorpusDescriptorPtr;
}}}
