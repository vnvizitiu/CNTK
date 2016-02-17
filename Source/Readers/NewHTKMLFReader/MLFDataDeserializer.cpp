//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "MLFDataDeserializer.h"
#include "ConfigHelper.h"
#include "htkfeatio.h"
#include "msra_mgram.h"
#include <ElementTypeUtils.h>

namespace Microsoft { namespace MSR { namespace CNTK {

MLFDataDeserializer::MLFDataDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& label, const std::wstring& name)
    : m_mlfPaths(std::move(ConfigHelper::GetMlfPaths(label))), m_name(name)
{
    m_frameMode = label.Find("frameMode", "true");
    assert(m_frameMode);

    ConfigHelper::CheckLabelType(label);

    m_elementType = ConfigHelper::GetElementType(label);
    m_elementSize = GetSizeByType(m_elementType);

    m_dimension = ConfigHelper::GetLabelDimension(label);
    m_layout = std::make_shared<TensorShape>(m_dimension);

    m_stateListPath = static_cast<std::wstring>(label(L"labelMappingFile", L""));

    // TODO: currently assumes all Mlfs will have same root name (key)
    // restrict MLF reader to these files--will make stuff much faster without having to use shortened input files

    // get labels
    const double htktimetoframe = 100000.0; // default is 10ms

    const msra::lm::CSymbolSet* wordTable = nullptr;
    std::map<string, size_t>* symbolTable = nullptr;

    msra::asr::htkmlfreader<msra::asr::htkmlfentry, msra::lattices::lattice::htkmlfwordsequence> labels(m_mlfPaths, std::set<wstring>(), m_stateListPath, wordTable, symbolTable, htktimetoframe);

    // Make sure 'msra::asr::htkmlfreader' type has a move constructor
    static_assert(
        std::is_move_constructible<
            msra::asr::htkmlfreader<msra::asr::htkmlfentry,
                                    msra::lattices::lattice::htkmlfwordsequence>>::value,
        "Type 'msra::asr::htkmlfreader' should be move constructible!");

    MLFUtterance description;
    //description.m_id = 0;
    description.m_isValid = true; // right now we throw for invalid sequences
    // TODO .chunk, .key

    size_t totalFrames = 0;
    // Have to iterate in the same order as utterances inside the HTK data de-serializer to be aligned.
    for (const auto& l : labels)
    {
        description.m_key.major = l.first;

        // todo check that actually exists.
        const auto& labseq = l.second;

        description.sequenceStart = m_classIds.size(); // TODO
        description.m_isValid = true;
        size_t numofframes = 0;
        //description.m_id = corpus->GetId(key, 0);

        foreach_index (i, labseq)
        {
            // TODO Why will these yield a run-time error as opposed to making the utterance invalid?
            const auto& e = labseq[i];
            if ((i == 0 && e.firstframe != 0) ||
                (i > 0 && labseq[i - 1].firstframe + labseq[i - 1].numframes != e.firstframe))
            {
                RuntimeError("minibatchutterancesource: labels not in consecutive order MLF in label set: %ls", l.first.c_str());
            }

            if (e.classid >= m_dimension)
            {
                RuntimeError("minibatchutterancesource: class id %d exceeds model output dimension %d in file",
                             static_cast<int>(e.classid),
                             static_cast<int>(m_dimension));
            }

            if (e.classid != static_cast<msra::dbn::CLASSIDTYPE>(e.classid))
            {
                RuntimeError("CLASSIDTYPE has too few bits");
            }

            for (size_t t = e.firstframe; t < e.firstframe + e.numframes; t++)
            {
                m_classIds.push_back(e.classid);
                numofframes++;
            }
        }

        description.m_numberOfSamples = numofframes;
        totalFrames += numofframes;
        m_utterances.push_back(description);
        m_keyToSequence[description.m_key.major] = m_utterances.size() - 1;
    }

    if (m_frameMode)
    {
        m_frames.reserve(totalFrames);
        m_sequences.reserve(totalFrames);

    }
    else
    {
        m_sequences.reserve(m_utterances.size());
    }

    foreach_index(i, m_utterances)
    {
        if (m_frameMode)
        {
            m_utterances[i].frameStart = m_frames.size();
            for (size_t k = 0; k < m_utterances[i].m_numberOfSamples; ++k)
            {
                MLFFrame f;
                f.m_id = m_frames.size();
                f.m_key.major = m_utterances[i].m_key.major;
                f.m_key.minor = k;
                f.m_chunkId = 0;
                f.m_numberOfSamples = 1;
                f.index = m_utterances[i].sequenceStart + k;
                assert(m_utterances[i].m_isValid); // TODO
                f.m_isValid = m_utterances[i].m_isValid;
                m_frames.push_back(f);
                m_sequences.push_back(&m_frames[f.m_id]);
            }
        }
        else
        {
            assert(false);
            m_sequences.push_back(&m_utterances[i]);
        }
    }
}

const SequenceDescriptions& Microsoft::MSR::CNTK::MLFDataDeserializer::GetSequenceDescriptions() const
{
    return m_sequences;
}

std::vector<StreamDescriptionPtr> MLFDataDeserializer::GetStreamDescriptions() const
{
    StreamDescriptionPtr stream = std::make_shared<StreamDescription>();
    stream->m_id = 0;
    stream->m_name = m_name;
    stream->m_sampleLayout = std::make_shared<TensorShape>(m_dimension);
    stream->m_storageType = StorageType::sparse_csc;
    stream->m_elementType = m_elementSize == sizeof(float) ? ElementType::tfloat : ElementType::tdouble;
    return std::vector<StreamDescriptionPtr>{stream};
}

class MLFDataDeserializer::MLFChunk : public Chunk
{
    MLFDataDeserializer* m_parent;
public:
    MLFChunk(MLFDataDeserializer* parent) : m_parent(parent)
    {}

    virtual std::vector<SequenceDataPtr> GetSequence(const size_t& sequenceId) override
    {
        return m_parent->GetSequenceById(sequenceId);
    }
};

ChunkPtr MLFDataDeserializer::GetChunk(size_t chunkId)
{
    assert(chunkId == 0);
    UNUSED(chunkId);
    return std::make_shared<MLFChunk>(this);
}

std::vector<SequenceDataPtr> MLFDataDeserializer::GetSequenceById(size_t sequenceId)
{
    auto id = sequenceId;

    static float oneFloat = 1.0;
    static double oneDouble = 1.0;

    size_t label = m_classIds[m_frames[id].index];
    SparseSequenceDataPtr r = std::make_shared<SparseSequenceData>();
    r->m_indices.resize(1);
    r->m_indices[0] = std::vector<size_t>{ label };

    if (m_elementSize == sizeof(float))
    {
        r->m_data = &oneFloat;
    }
    else
    {
        r->m_data = &oneDouble;
    }

    return std::vector<SequenceDataPtr> { r };
}

const SequenceDescription* MLFDataDeserializer::GetSequenceDescriptionByKey(const KeyType& key)
{
    size_t sequenceId = m_keyToSequence[key.major];
    size_t index = m_utterances[sequenceId].frameStart + key.minor;
    return m_sequences[index];
}

}}}
