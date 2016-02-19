//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "DataDeserializer.h"
#include "Config.h"
#include "htkfeatio.h"
#include "ssematrix.h"
#include "CorpusDescriptor.h"
#include "UtteranceDescription.h"
#include "ChunkDescription.h"

namespace Microsoft { namespace MSR { namespace CNTK {

struct Frame : public SequenceDescription
{
    Frame(UtteranceDescription* u) : u(u)
    {
    }

    UtteranceDescription* u;
    size_t frameIndexInUtterance;
};

class HTKDataDeserializer : public IDataDeserializer
{
public:
    HTKDataDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& feature, const std::wstring& featureName);

    virtual const SequenceDescriptions& GetSequenceDescriptions() const override;
    virtual std::vector<StreamDescriptionPtr> GetStreamDescriptions() const override;
    virtual ChunkPtr GetChunk(size_t) override;
    virtual size_t GetTotalNumberOfChunks() override;
    virtual const SequenceDescription* GetSequenceDescriptionByKey(const KeyType& key) override;

private:
    DISABLE_COPY_AND_MOVE(HTKDataDeserializer);

    class HTKChunk;
    std::vector<SequenceDataPtr> GetSequenceById(size_t id);

    size_t m_dimension;
    std::vector<UtteranceDescription> m_utterances;
    std::vector<Frame> m_frames;

    ElementType m_elementType;
    SequenceDescriptions m_sequences;

    std::vector<ChunkDescription> m_chunks;
    std::vector<std::weak_ptr<Chunk>> m_weakChunks;

    std::pair<size_t, size_t> m_augmentationWindow;
    std::vector<StreamDescriptionPtr> m_streams;
    int m_verbosity;

    // for reference check about the data in the feature file.
    unsigned int m_samplePeriod;
    size_t m_ioFeatureDimension;
    std::string m_featureKind;
};

typedef std::shared_ptr<HTKDataDeserializer> HTKDataDeserializerPtr;

}}}
