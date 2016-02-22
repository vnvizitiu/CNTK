//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "DataDeserializer.h"
#include "../HTKMLFReader/htkfeatio.h"
#include "ssematrix.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// Class describes a chunk of data.
class ChunkDescription
{
    // utterances in this set
    std::vector<UtteranceDescription*> m_utteranceSet;

    std::vector<size_t> m_firstFrames;    // [utteranceindex] first frame for given utterance
    mutable msra::dbn::matrix m_frames;   // stores all frames consecutively (mutable since this is a cache)
    size_t m_totalFrames;                 // total #frames for all utterances in this chunk

public:
    ChunkDescription() : m_totalFrames(0)
    {
    }

    size_t GetNumberOfUtterances() const
    {
        return m_utteranceSet.size();
    }

    void Add(UtteranceDescription* utterance)
    {
        if (IsInRam())
        {
            LogicError("utterancechunkdata: frames already paged into RAM--too late to add data");
        }

        m_firstFrames.push_back(m_totalFrames);
        m_totalFrames += utterance->GetNumberOfFrames();
        m_utteranceSet.push_back(utterance);
    }

    size_t GetTotalFrames() const
    {
        return m_totalFrames;
    }

    size_t GetUtteranceNumberOfFrames(size_t i) const
    {
        return m_utteranceSet[i]->GetNumberOfFrames();
    }

    // return the frame set for a given utterance
    msra::dbn::matrixstripe GetUtteranceFrames(size_t i) const
    {
        if (!IsInRam())
        {
            LogicError("getutteranceframes: called when data have not been paged in");
        }

        const size_t ts = m_firstFrames[i];
        const size_t n = GetUtteranceNumberOfFrames(i);
        return msra::dbn::matrixstripe(m_frames, ts, n);
    }

    // test if data is in memory at the moment
    bool IsInRam() const
    {
        return !m_frames.empty();
    }

    // page in data for this chunk
    // We pass in the feature info variables by ref which will be filled lazily upon first read
    // this function supports retrying since we read from the unreliable network, i.e. do not return in a broken state
    void RequireData(const string& featureKind, size_t featureDimension, unsigned int samplePeriod, int verbosity = 0) const
    {
        if (GetNumberOfUtterances() == 0)
        {
            LogicError("requiredata: cannot page in virgin block");
        }

        if (IsInRam())
        {
            LogicError("requiredata: called when data is already in memory");
        }

        try
        {
            // feature reader (we reinstantiate it for each block, i.e. we reopen the file actually)
            // if this is the first feature read ever, we explicitly open the first file to get the information such as feature dimension
            msra::asr::htkfeatreader reader;

            // read all utterances; if they are in the same archive, htkfeatreader will be efficient in not closing the file
            m_frames.resize(featureDimension, m_totalFrames);
            foreach_index(i, m_utteranceSet)
            {
                // read features for this file
                auto framesWrapper = GetUtteranceFrames(i);
                reader.read(m_utteranceSet[i]->GetPath(), featureKind, samplePeriod, framesWrapper);
            }

            if (verbosity)
                fprintf(stderr, "requiredata: %d utterances read\n", (int)m_utteranceSet.size());
        }
        catch (...)
        {
            ReleaseData();
            throw;
        }
    }

    // page out data for this chunk
    void ReleaseData() const
    {
        if (GetNumberOfUtterances() == 0)
        {
            LogicError("releasedata: cannot page out virgin block");
        }

        if (!IsInRam())
        {
            LogicError("releasedata: called when data is not memory");
        }

        // release frames
        m_frames.resize(0, 0);
    }
};

}}}
