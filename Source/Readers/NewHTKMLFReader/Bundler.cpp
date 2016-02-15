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
            StreamDescriptionPtr stream = std::make_shared<StreamDescription>(*i);
            stream->m_id = streams.size();
            streams.push_back(stream);
        }
    }

    m_streams = streams;
    CreateSequenceDescriptions();
}

void Bundler::CreateSequenceDescriptions()
{
    m_sequenceToSequence.resize(m_deserializers.size());
    m_sequenceToChunk.resize(m_deserializers.size());
    m_sequenceDescriptions.reserve(m_driver->GetSequenceDescriptions().size());

    size_t maxNumberOfSequences = m_driver->GetSequenceDescriptions().size();
    for (int i = 0; i < m_deserializers.size(); ++i)
    {
        m_sequenceToSequence[i].resize(maxNumberOfSequences);
    }

    size_t previousChunk = SIZE_MAX;
    size_t currentMapping = 0;
    for (int i = 0; i < m_driver->GetSequenceDescriptions().size(); ++i)
    {
        auto sequenceDescription = m_driver->GetSequenceDescriptions()[i];

        bool isValid = true;
        for (int j = 1; j < m_deserializers.size(); ++j)
        {
            auto description = m_deserializers[j]->GetSequenceDescriptionByKey(sequenceDescription->m_key);
            if (!description->m_isValid)
            {
                isValid = false;
                break;
            }

            m_sequenceToChunk[j][description->m_id] = description->m_chunkId;
            m_sequenceToSequence[j][currentMapping] = description->m_id;
        }

        m_sequenceToChunk[0][sequenceDescription->m_id] = sequenceDescription->m_chunkId;
        m_sequenceToSequence[0][currentMapping] = sequenceDescription->m_id;

        if (isValid)
        {
            if (sequenceDescription->m_chunkId != previousChunk)
            {
                m_chunkOffsets.push_back(m_sequenceDescriptions.size());
                previousChunk = sequenceDescription->m_chunkId;
            }

            m_sequenceDescriptions.push_back(*sequenceDescription);
            m_sequenceDescriptions.back().m_id = m_sequenceDescriptions.size() - 1;
            m_sequenceToSequence[0][currentMapping] = sequenceDescription->m_id;
            currentMapping++;
        }
    }

    for (int i = 0; i < m_deserializers.size(); ++i)
    {
        m_sequenceToSequence.resize(currentMapping);
    }

    // Last
    m_chunkOffsets.push_back(m_sequenceDescriptions.size());

    m_sequences.resize(m_sequenceDescriptions.size());
    for (int k = 0; k < m_sequenceDescriptions.size(); ++k)
    {
        m_sequences[k] = &m_sequenceDescriptions[k];
    }
}

class BundlingChunk : public Chunk
{
    size_t m_numberOfInputs;
    Bundler* m_parent;

public:
    BundlingChunk(size_t numberOfInputs, Bundler* parent, const std::map<size_t, std::vector<ChunkPtr>>& sequences)
        : m_sequences(sequences), m_numberOfInputs(numberOfInputs), m_parent(parent)
    {}

    virtual std::vector<SequenceDataPtr> GetSequence(const size_t& sequenceId) override
    {
        const auto& chunks = m_sequences[sequenceId];
        std::vector<SequenceDataPtr> result;
        result.reserve(m_numberOfInputs);

        for (int i = 0; i < chunks.size(); ++i)
        {
            auto sequences = chunks[i]->GetSequence(m_parent->m_sequenceToSequence[i][sequenceId]);
            result.insert(result.end(), sequences.begin(), sequences.end());
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

    return std::make_shared<BundlingChunk>(m_streams.size(), this, sequences);
}

const SequenceDescriptions& Bundler::GetSequenceDescriptions() const
{
    return m_sequences;
}

std::vector<StreamDescriptionPtr> Bundler::GetStreamDescriptions() const
{
    return m_streams;
}

const SequenceDescription* Bundler::GetSequenceDescriptionByKey(const KeyType&)
{
    throw std::logic_error("Not implemented");
}

}}}
