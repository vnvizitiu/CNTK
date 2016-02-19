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

namespace Microsoft { namespace MSR { namespace CNTK {

// data store (incl. paging in/out of features and lattices)
class UtteranceDescription : public SequenceDescription // data descriptor for one utterance
{
    // archive filename and frame range in that file
    msra::asr::htkfeatreader::parsedpath m_path;
    size_t m_indexInsideChunk;

public:
    UtteranceDescription(msra::asr::htkfeatreader::parsedpath&& path)
        : m_path(std::move(path))
    {
    }

    const msra::asr::htkfeatreader::parsedpath& GetPath() const
    {
        return m_path;
    }

    size_t GetNumberOfFrames() const
    {
        return m_path.numframes();
    }

    wstring GetKey() const
    {
        std::wstring filename(m_path);
        return filename.substr(0, filename.find_last_of(L"."));
    }

    size_t GetIndexInsideChunk() const
    {
        return m_indexInsideChunk;
    }

    void SetIndexInsideChunk(size_t indexInsideChunk)
    {
        m_indexInsideChunk = indexInsideChunk;
    }
};

struct chunkdata // data for a chunk of utterances
{
    std::vector<UtteranceDescription*> utteranceset; // utterances in this set
    size_t numutterances() const
    {
        return utteranceset.size();
    }

    std::vector<size_t> firstframes;                                                       // [utteranceindex] first frame for given utterance
    mutable msra::dbn::matrix frames;                                                      // stores all frames consecutively (mutable since this is a cache)
    size_t totalframes;                                                                    // total #frames for all utterances in this chunk

    // construction
    chunkdata()
        : totalframes(0)
    {
    }

    void push_back(UtteranceDescription* utt)
    {
        if (isinram())
            LogicError("utterancechunkdata: frames already paged into RAM--too late to add data");
        firstframes.push_back(totalframes);
        totalframes += utt->GetNumberOfFrames();
        utteranceset.push_back(utt);
    }

    // accessors to an utterance's data
    size_t numframes(size_t i) const
    {
        return utteranceset[i]->GetNumberOfFrames();
    }

    msra::dbn::matrixstripe getutteranceframes(size_t i) const // return the frame set for a given utterance
    {
        if (!isinram())
            LogicError("getutteranceframes: called when data have not been paged in");
        const size_t ts = firstframes[i];
        const size_t n = numframes(i);
        return msra::dbn::matrixstripe(frames, ts, n);
    }

    // paging
    // test if data is in memory at the moment
    bool isinram() const
    {
        return !frames.empty();
    }
    // page in data for this chunk
    // We pass in the feature info variables by ref which will be filled lazily upon first read
    void requiredata(string& featkind, size_t& featdim, unsigned int& sampperiod, int verbosity = 0) const
    {
        if (numutterances() == 0)
            LogicError("requiredata: cannot page in virgin block");
        if (isinram())
            LogicError("requiredata: called when data is already in memory");
        try // this function supports retrying since we read from the unrealible network, i.e. do not return in a broken state
        {
            msra::asr::htkfeatreader reader; // feature reader (we reinstantiate it for each block, i.e. we reopen the file actually)
            // if this is the first feature read ever, we explicitly open the first file to get the information such as feature dimension
            if (featdim == 0)
            {
                reader.getinfo(utteranceset[0]->GetPath(), featkind, featdim, sampperiod);
                fprintf(stderr, "requiredata: determined feature kind as %d-dimensional '%s' with frame shift %.1f ms\n",
                        static_cast<int>(featdim), featkind.c_str(), sampperiod / 1e4);
            }
            // read all utterances; if they are in the same archive, htkfeatreader will be efficient in not closing the file
            frames.resize(featdim, totalframes);
            foreach_index (i, utteranceset)
            {
                //fprintf (stderr, ".");
                // read features for this file
                auto uttframes = getutteranceframes(i);                                                    // matrix stripe for this utterance (currently unfilled)
                reader.read(utteranceset[i]->GetPath(), (const string&)featkind, sampperiod, uttframes); // note: file info here used for checkuing only
                // page in lattice data
            }

            if (verbosity)
                fprintf(stderr, "requiredata: %d utterances read\n", (int) utteranceset.size());
        }
        catch (...)
        {
            releasedata();
            throw;
        }
    }

    // page out data for this chunk
    void releasedata() const
    {
        if (numutterances() == 0)
            LogicError("releasedata: cannot page out virgin block");
        if (!isinram())
            LogicError("releasedata: called when data is not memory");
        // release frames
        frames.resize(0, 0);
        // release lattice data
    }
};

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
    class HTKChunk;
    std::vector<SequenceDataPtr> GetSequenceById(size_t id);

    size_t m_dimension;
    std::vector<UtteranceDescription> m_utterances;
    std::vector<Frame> m_frames;

    ElementType m_elementType;
    SequenceDescriptions m_sequences;

    std::vector<chunkdata> m_chunks;
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
