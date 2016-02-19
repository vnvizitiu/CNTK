//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "HTKDataDeserializer.h"
#include "ConfigHelper.h"
#include "Basics.h" // for attempt()
#include "minibatchsourcehelpers.h"
#include <numeric>

namespace Microsoft { namespace MSR { namespace CNTK {

    HTKDataDeserializer::HTKDataDeserializer(
        CorpusDescriptorPtr corpus,
        const ConfigParameters& feature,
        const std::wstring& featureName)
        : m_ioFeatureDimension(0),
          m_samplePeriod(0),
          m_verbosity(0)
{
    bool frameMode = feature.Find("frameMode", "true");
    if (!frameMode)
    {
        LogicError("Currently only reader only supports fram mode. Please check your configuration.");
    }

    ConfigHelper config(feature);

    config.CheckFeatureType();

    std::vector<std::wstring> featureFiles = config.GetFeaturePaths();

    auto context = config.GetContextWindow();
    m_elementType = config.GetElementType();
    m_dimension = config.GetFeatureDimension();
    m_dimension = m_dimension * (1 + context.first + context.second);

    size_t numSequences = featureFiles.size();

    m_augmentationWindow = config.GetContextWindow();

    m_utterances.reserve(numSequences);
    size_t totalFrames = 0;
    foreach_index (i, featureFiles)
    {
        UtteranceDescription description(std::move(msra::asr::htkfeatreader::parsedpath(featureFiles[i])));
        size_t numberOfFrames = description.GetNumberOfFrames();
        description.m_id = i;

        // we need at least 2 frames for boundary markers to work
        if (numberOfFrames < 2)
        {
            fprintf(stderr, "HTKDataDeserializer::HTKDataDeserializer: skipping utterance with %d frames because it has less than 2 frames: %ls\n", 
                (int)numberOfFrames, description.GetKey().c_str());
            description.m_isValid = false;
            description.m_numberOfSamples = 0;
        }
        else
        {
            description.m_isValid = true;
            description.m_numberOfSamples = numberOfFrames;
        }

        m_utterances.push_back(description);
        totalFrames += description.m_numberOfSamples;
    }

    size_t totalSize = std::accumulate(
        m_utterances.begin(),
        m_utterances.end(),
        static_cast<size_t>(0),
        [](size_t sum, const UtteranceDescription& s)
        {
            return s.m_numberOfSamples + sum;
        });

    // distribute them over chunks
    // We simply count off frames until we reach the chunk size.
    // Note that we first randomize the chunks, i.e. when used, chunks are non-consecutive and thus cause the disk head to seek for each chunk.
    const size_t framespersec = 100;                   // we just assume this; our efficiency calculation is based on this
    const size_t chunkframes = 15 * 60 * framespersec; // number of frames to target for each chunk

    // Loading an initial 24-hour range will involve 96 disk seeks, acceptable.
    // When paging chunk by chunk, chunk size ~14 MB.

    m_chunks.resize(0);
    m_chunks.reserve(totalSize / chunkframes);

    int chunkId = -1;
    foreach_index(i, m_utterances)
    {
        // if exceeding current entry--create a new one
        // I.e. our chunks are a little larger than wanted (on av. half the av. utterance length).
        if (m_chunks.empty() || m_chunks.back().totalframes > chunkframes || m_chunks.back().numutterances() >= 65535)
        {
            // TODO > instead of >= ? if (thisallchunks.empty() || thisallchunks.back().totalframes > chunkframes || thisallchunks.back().numutterances() >= frameref::maxutterancesperchunk)
            m_chunks.push_back(chunkdata());
            chunkId++;
        }

        // append utterance to last chunk
        chunkdata& currentchunk = m_chunks.back();
        m_utterances[i].SetIndexInsideChunk(currentchunk.numutterances());
        currentchunk.push_back(&m_utterances[i]); // move it out from our temp array into the chunk
        m_utterances[i].m_chunkId = chunkId;
        // TODO: above push_back does not actually 'move' because the internal push_back does not accept that
    }

    fprintf(stderr, "minibatchutterancesource: %d utterances grouped into %d chunks, av. chunk size: %.1f utterances, %.1f frames\n",
        static_cast<int>(m_utterances.size()),
        static_cast<int>(m_chunks.size()),
        m_utterances.size() / (double)m_chunks.size(),
        totalSize / (double)m_chunks.size());
    // Now utterances are stored exclusively in allchunks[]. They are never referred to by a sequential utterance id at this point, only by chunk/within-chunk index.

    m_frames.reserve(totalFrames);


    foreach_index(i, m_utterances)
    {
            std::wstring key = m_utterances[i].GetKey();
            for (size_t k = 0; k < m_utterances[i].m_numberOfSamples; ++k)
            {
                Frame f(&m_utterances[i]);
                f.m_key.major = key;
                f.m_key.minor = k;
                f.m_id = m_frames.size();
                f.m_chunkId = m_utterances[i].m_chunkId;
                f.m_numberOfSamples = 1;
                f.frameIndexInUtterance = k;
                assert(m_utterances[i].m_isValid); // TODO
                f.m_isValid = m_utterances[i].m_isValid;
                m_frames.push_back(f);

                m_sequences.push_back(&m_frames[f.m_id]);
            }
    }

    m_weakChunks.resize(m_chunks.size());

    StreamDescriptionPtr stream = std::make_shared<StreamDescription>();
    stream->m_id = 0;
    stream->m_name = featureName;
    stream->m_sampleLayout = std::make_shared<TensorShape>(m_dimension);
    stream->m_elementType = m_elementType;
    stream->m_storageType = StorageType::dense;
    m_streams.push_back(stream);
}

const SequenceDescriptions& HTKDataDeserializer::GetSequenceDescriptions() const
{
    return m_sequences;
}

std::vector<StreamDescriptionPtr> HTKDataDeserializer::GetStreamDescriptions() const
{
    return m_streams;
}

class matrixasvectorofvectors // wrapper around a matrix that views it as a vector of column vectors
{
    void operator=(const matrixasvectorofvectors&); // non-assignable
    msra::dbn::matrixbase& m;

public:
    matrixasvectorofvectors(msra::dbn::matrixbase& m)
        : m(m)
    {
    }
    size_t size() const
    {
        return m.cols();
    }
    const_array_ref<float> operator[](size_t j) const
    {
        return array_ref<float>(&m(0, j), m.rows());
    }
};

class HTKDataDeserializer::HTKChunk : public Chunk
{
    HTKDataDeserializer* m_parent;
    size_t m_chunkId;
public:
    HTKChunk(HTKDataDeserializer* parent, size_t chunkId) : m_parent(parent), m_chunkId(chunkId)
    {
        auto& chunkdata = m_parent->m_chunks[chunkId];
        if (chunkdata.isinram())
        {
            LogicError("Trying to load the chunk that is already in memory.");
        }

        // possibly distributed read.
        msra::util::attempt(5, [&]()
        {
            chunkdata.requiredata(m_parent->m_featureKind, m_parent->m_ioFeatureDimension, m_parent->m_samplePeriod, m_parent->m_verbosity);
        });
    }

    virtual std::vector<SequenceDataPtr> GetSequence(const size_t& sequenceId) override
    {
        return m_parent->GetSequenceById(sequenceId);
    }

    ~HTKChunk()
    {
        auto& chunkdata = m_parent->m_chunks[m_chunkId];
        if (chunkdata.isinram())
        {
            chunkdata.releasedata();
        }
        else
        {
            LogicError("Trying to unload the chunk that is already unloaded.");
        }
    }
};

ChunkPtr HTKDataDeserializer::GetChunk(size_t chunkId)
{
    if (!m_weakChunks[chunkId].expired())
    {
        return m_weakChunks[chunkId].lock();
    }

    auto chunk = std::make_shared<HTKChunk>(this, chunkId);
    m_weakChunks[chunkId] = chunk;
    return chunk;
};

struct HTKSequenceData : DenseSequenceData
{
    msra::dbn::matrix utterance;

    ~HTKSequenceData()
    {
        msra::dbn::matrixstripe frame(utterance, 0, utterance.cols());
        if (m_data != &frame(0, 0))
        {
            delete[] reinterpret_cast<double*>(m_data);
            m_data = nullptr;
        }
    }
};

typedef std::shared_ptr<HTKSequenceData> HTKSequenceDataPtr;

std::vector<SequenceDataPtr> HTKDataDeserializer::GetSequenceById(size_t id)
{
        const auto& frame = m_frames[id];
        UtteranceDescription* utterance = frame.u;

        HTKSequenceDataPtr r = std::make_shared<HTKSequenceData>();
        r->utterance.resize(m_dimension, 1);

        const auto& chunkdata = m_chunks[utterance->m_chunkId];

        auto uttframes = chunkdata.getutteranceframes(utterance->GetIndexInsideChunk());
        matrixasvectorofvectors uttframevectors(uttframes); // (wrapper that allows m[j].size() and m[j][i] as required by augmentneighbors())

        size_t leftextent, rightextent;
        // page in the needed range of frames
        if (m_augmentationWindow.first == 0 && m_augmentationWindow.second == 0)
        {
            // should this be moved up?
            leftextent = rightextent = msra::dbn::augmentationextent(uttframevectors[0].size(), m_dimension);
        }
        else
        {
            leftextent = m_augmentationWindow.first;
            rightextent = m_augmentationWindow.second;
        }

        const std::vector<char> noboundaryflags; // dummy
        msra::dbn::augmentneighbors(uttframevectors, noboundaryflags, frame.frameIndexInUtterance, leftextent, rightextent, r->utterance, 0);

        r->m_numberOfSamples = frame.m_numberOfSamples;
        msra::dbn::matrixstripe featOr(r->utterance, 0, r->utterance.cols());
        const size_t dimensions = featOr.rows();

        if (m_elementType == ElementType::tfloat)
        {
            r->m_data = &featOr(0, 0);
        }
        else 
        {
            // TODO allocate double, convert in-place from end to start instead
            assert(m_elementType == ElementType::tdouble);
            double *doubleBuffer = new double[dimensions];
            const float *floatBuffer = &featOr(0, 0);

            for (size_t i = 0; i < dimensions; i++)
            {
                doubleBuffer[i] = floatBuffer[i];
            }

            r->m_data = doubleBuffer;
        }

        return std::vector<SequenceDataPtr>(1, r); // TODO would std::move help?
}

const SequenceDescription* HTKDataDeserializer::GetSequenceDescriptionByKey(const KeyType&)
{
    LogicError("HTKDataDeserializer::GetSequenceDescriptionByKey: currently not implemented. Supported only as primary deserializer.");
}

size_t HTKDataDeserializer::GetTotalNumberOfChunks()
{
    return m_chunks.size();
}

} } }
