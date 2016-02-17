//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS

#include "SampleModePacker.h"
#include "ElementTypeUtils.h"

namespace Microsoft { namespace MSR { namespace CNTK {

SampleModePacker::SampleModePacker(
    MemoryProviderPtr memoryProvider,
    TransformerPtr transformer,
    size_t minibatchSize,
    const std::vector<StreamDescriptionPtr>& streams) : m_transformer(transformer),
                                                        m_minibatchSize(minibatchSize),
                                                        m_outputStreams(streams),
                                                        m_minibatchLayout(std::make_shared<MBLayout>()),
                                                        m_memoryProvider(memoryProvider)
{
    m_inputStreams = m_transformer->GetStreamDescriptions();
    assert(m_inputStreams.size() == m_outputStreams.size());
    assert(
        std::find_if(
            m_outputStreams.begin(),
            m_outputStreams.end(),
            [](const StreamDescriptionPtr& s)
            {
                return s->m_storageType == StorageType::sparse_csc;
            }) == m_outputStreams.end());

    assert(m_minibatchSize > 0);
    for (int i = 0; i < m_outputStreams.size(); ++i)
    {
        const auto& stream = m_outputStreams[i];
        // Input and output should match in everything except for sparse/dense.
        assert(stream->m_elementType == ElementType::tfloat || stream->m_elementType == ElementType::tdouble);
        assert(stream->m_name == m_inputStreams[i]->m_name);
        assert(stream->m_id == m_inputStreams[i]->m_id);
        assert(GetSampleSize(m_inputStreams[i]) == GetSampleSize(stream));

        m_streamBuffers.push_back(
            AllocateBuffer(m_minibatchSize * stream->m_sampleLayout->GetNumElements(), GetSizeByType(stream->m_elementType)));
    }
}

Minibatch SampleModePacker::ReadMinibatch()
{
    m_transformer->GetNextSequences(m_minibatchSize, m_bufferSequences);

    Minibatch minibatch;
    minibatch.m_endOfEpoch = m_bufferSequences.m_endOfEpoch;

    // For each sequence iterating thru all the streams with this sequence id and copying to the buffer.
    for (size_t sequenceIndex = 0; sequenceIndex < m_bufferSequences.m_data.size(); sequenceIndex++)
    {
        assert(m_streamBuffers.size() == m_bufferSequences.m_data[sequenceIndex].size());
        // Iterating for sequences inside the batch of sequences.
        for (size_t streamIndex = 0; streamIndex < m_bufferSequences.m_data[sequenceIndex].size(); ++streamIndex)
        {
            CopySequenceToBuffer(m_bufferSequences.m_data[sequenceIndex][streamIndex], streamIndex, sequenceIndex);
        }
    }

    if (m_bufferSequences.m_data.size() == 0)
    {
        return minibatch;
    }

    // Creating output minibatch with shared layout between all streams.
    m_minibatchLayout->InitAsFrameMode(m_bufferSequences.m_data.size());
    minibatch.m_data.reserve(m_outputStreams.size());
    for (int i = 0; i < m_outputStreams.size(); ++i)
    {
        auto stream = std::make_shared<StreamMinibatch>();
        stream->m_data = m_streamBuffers[i].get();
        stream->m_dataSize = m_bufferSequences.m_data.size() * GetSampleSize(m_outputStreams[i]);
        stream->m_layout = m_minibatchLayout;

        minibatch.m_data.push_back(stream);
    }

    return minibatch;
}

size_t SampleModePacker::GetSampleSize(StreamDescriptionPtr stream)
{
    assert(stream != nullptr);
    size_t elementSize = GetSizeByType(stream->m_elementType);
    return stream->m_sampleLayout->GetNumElements() * elementSize;
}

void SampleModePacker::CopySequenceToBuffer(SequenceDataPtr sample, size_t streamIndex, size_t sampleIndex)
{
    // In framemode sequence just contains a single sample.
    size_t sampleSize = GetSampleSize(m_inputStreams[streamIndex]);
    auto sampleData = reinterpret_cast<const char*>(sample->m_data);

    const auto& stream = m_inputStreams[streamIndex];
    auto elementSize = GetSizeByType(stream->m_elementType);
    auto buffer = m_streamBuffers[streamIndex].get();

    if (stream->m_storageType == StorageType::dense)
    {
        auto data = reinterpret_cast<DenseSequenceData&>(*sample);
        // Expect single sample.
        assert(data.m_numberOfSamples == 1);

        // Copying the sequence to its position in the buffer. Effectivly a buffer contains concatenation of samples for a stream.
        std::copy(sampleData, sampleData + sampleSize, buffer + sampleIndex * sampleSize);
    }
    else if (stream->m_storageType == StorageType::sparse_csc)
    {
        auto data = reinterpret_cast<SparseSequenceData&>(*sample);
        // Expect single sample.
        assert(data.m_indices.size() == 1);

        // Currently sparse data has to be unpacked to the dense one. Possibly can be done later
        // in the network or as a transformation.

        // Fill it in with zeros.
        std::fill(buffer + sampleIndex * sampleSize, buffer + (sampleIndex + 1) * sampleSize, 0);

        // Copy the non zero data to the buffer.
        size_t nonZeroCount = data.m_indices[0].size();
        for (size_t nonZeroIndex = 0; nonZeroIndex < nonZeroCount; ++nonZeroIndex)
        {
            size_t rowIndex = data.m_indices[0][nonZeroIndex];
            char* destination = buffer + sampleIndex * sampleSize + rowIndex * elementSize;
            std::copy(sampleData + nonZeroIndex * elementSize, sampleData + (nonZeroIndex + 1) * elementSize, destination);
        }
    }
    else
    {
        RuntimeError("Storage type %d is not supported.", m_inputStreams[streamIndex]->m_storageType);
    }
}

std::shared_ptr<char> SampleModePacker::AllocateBuffer(size_t numElements, size_t elementSize)
{
    return std::shared_ptr<char>(
        reinterpret_cast<char*>(m_memoryProvider->Alloc(elementSize, numElements)),
        [this](char* p)
        {
            m_memoryProvider->Free(p);
        });
}
} } }
