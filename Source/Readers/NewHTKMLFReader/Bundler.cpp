//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "Bundler.h"
#include "ConfigHelper.h"

namespace Microsoft { namespace MSR { namespace CNTK {

Bundler::Bundler(
    const ConfigParameters& readerConfig,
    IDataDeserializerPtr driver,
    std::vector<IDataDeserializerPtr> deserializers)
    : m_deserializers(deserializers), m_driver(driver)
{
    UNREFERENCED_PARAMETER(readerConfig);
    std::vector<StreamDescriptionPtr> streams;
    for (auto d : deserializers)
    {
        for (auto i : d->GetStreamDescriptions())
        {
            StreamDescriptionPtr stream = std::make_shared<StreamDescription>();
            stream->m_id = streams.size();
            stream->m_name = i->m_name;
            stream->m_sampleLayout = i->m_sampleLayout;
            streams.push_back(stream);
        }
    }

    m_streams = streams;
    CreateSequenceDescriptions();
}

void Bundler::CreateSequenceDescriptions()
{
    m_sequenceToChunk.resize(m_deserializers.size());
    m_sequenceDescriptions.resize(m_driver->GetSequenceDescriptions().size());

    size_t previousChunk = SIZE_MAX;
    for (int i = 0; i < m_driver->GetSequenceDescriptions().size(); ++i)
    {
        auto sequenceDescription = m_driver->GetSequenceDescriptions()[i];

        bool isValid = true;
        for (int j = 0; j < m_deserializers.size(); ++j)
        {
            auto s = m_deserializers[j]->GetSequenceDescriptions()[i];
            if (!s->m_isValid)
            {
                isValid = false;
                break;
            }

            m_sequenceToChunk[j][s->m_id] = s->m_chunkId;
        }

        if (isValid)
        {
            if (sequenceDescription->m_chunkId != previousChunk)
            {
                m_chunkOffsets.push_back(m_sequenceDescriptions.size());
                previousChunk = sequenceDescription->m_chunkId;
            }

            m_sequenceDescriptions.push_back(*sequenceDescription);
        }
    }

    m_sequences.resize(m_sequenceDescriptions.size());
    for (int k = 0; k < m_sequenceDescriptions.size(); ++k)
    {
        m_sequences[k] = &m_sequenceDescriptions[k];
    }
}

class BundlingChunk : public Chunk
{
public:
    BundlingChunk(const std::map<size_t, std::vector<ChunkPtr>>& sequences) : m_sequences(sequences)
    {}

    virtual std::vector<SequenceDataPtr> GetSequence(const size_t& sequenceId) override
    {
        const auto& chunks = m_sequences[sequenceId];
        std::vector<SequenceDataPtr> result;
        result.resize(chunks.size());

        for (int i = 0; i < chunks.size(); ++i)
        {
            chunks[i]->GetSequence(sequenceId);
        }

        return result;
    }

private:
    std::map<size_t, std::vector<ChunkPtr>> m_sequences;
};

ChunkPtr Bundler::GetChunk(size_t chunkId)
{
    std::map<size_t, std::vector<ChunkPtr>> sequences;
    for (size_t j = m_chunkOffsets[chunkId]; j < m_chunkOffsets[chunkId + 1]; ++j)
    {
        size_t sequenceId = m_sequenceDescriptions[j].m_id;
        sequences[j].resize(m_deserializers.size());
        for (size_t i = 0; i < m_deserializers.size(); ++i)
        {
            size_t innerChunkId = m_sequenceToChunk[i][sequenceId];
            sequences[sequenceId][i] = m_deserializers[i]->GetChunk(innerChunkId);
        }
    }

    return std::make_shared<BundlingChunk>(sequences);
}

const SequenceDescriptions& Bundler::GetSequenceDescriptions() const
{
    return m_sequences;
}

std::vector<StreamDescriptionPtr> Bundler::GetStreamDescriptions() const
{
    return m_streams;
}

}}}
