//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// NewHTKMLFReaderShim.cpp: implementation for shim that wraps new HTKMLF reader
//

#include "stdafx.h"
#ifdef _WIN32
#include <objbase.h>
#endif
#include "Basics.h"

#include "htkfeatio.h"      // for reading HTK features
#include "latticearchive.h" // for reading HTK phoneme lattices (MMI training)

#define DATAREADER_EXPORTS // creating the exports here
#include "DataReader.h"
#include "Config.h"
#include "NewHTKMLFReaderShim.h"
#ifdef LEAKDETECT
#include <vld.h> // for memory leak detection
#endif

#ifdef __unix__
// TODO probably not needed anymore
#include <limits.h>
#endif

#include "SubstitutingMemoryProvider.h"
#include "CudaMemoryProvider.h"
#include "HeapMemoryProvider.h"
#include "ConfigHelper.h"
#include "HTKDataDeserializer.h"
#include "MLFDataDeserializer.h"
#include "Bundler.h"
#include "LegacyBlockRandomizer.h"
#include "Utils.h"
#include "SampleModePacker.h"

namespace Microsoft { namespace MSR { namespace CNTK {

std::vector<IDataDeserializerPtr> CreateDeserializers(const ConfigParameters& readerConfig,
    bool framemode,
    size_t elementSize)
{
    std::vector<std::wstring> featureNames;
    std::vector<std::wstring> labelNames;

    std::vector<std::wstring> notused;
    ConfigHelper::GetDataNamesFromConfig(readerConfig, featureNames, labelNames, notused, notused);
    if (featureNames.size() < 1 || labelNames.size() < 1)
    {
        // eldak: Don't we support unsupervised training?
        InvalidArgument("network needs at least 1 input and 1 output specified!");
    }

    std::vector<HTKDataDeserializerPtr> featureDeserializers;
    std::vector<MLFDataDeserializerPtr> labelDeserializers;
    CorpusDescriptorPtr corpus = std::make_shared<CorpusDescriptor>();
    for (const auto& featureName : featureNames)
    {
        auto deserializer = std::make_shared<HTKDataDeserializer>(corpus, readerConfig(featureName), elementSize, framemode, featureName);
        featureDeserializers.push_back(deserializer);
    }

    assert(featureDeserializers.size() == 1);

    for (const auto& labelName : labelNames)
    {
        auto deserializer = std::make_shared<MLFDataDeserializer>(corpus, readerConfig(labelName), elementSize, framemode, labelName);

        labelDeserializers.push_back(deserializer);
    }

    assert(labelDeserializers.size() == 1);

    std::vector<IDataDeserializerPtr> deserializers;
    deserializers.insert(deserializers.end(), featureDeserializers.begin(), featureDeserializers.end());
    deserializers.insert(deserializers.end(), labelDeserializers.begin(), labelDeserializers.end());

    // SIMILAR CHECKS ALREADY DONE IN THE BUNDLER.
    // Checking end sequences.
    /*size_t expected = deserializers[0]->GetSequenceDescriptions().size();
    std::vector<bool> isValid(expected, true);
    for (auto d : deserializers)
    {
    const auto& sequences = d->GetSequenceDescriptions();
    if (sequences.size() != expected)
    {
    RuntimeError("We have some invalid alignment\n");
    }

    foreach_index(i, sequences)
    {
    isValid[i] = isValid[i] && sequences[i]->m_isValid;
    assert(isValid[i]);
    }
    }

    // shouldn't this be checked (again) later? more utterances can be invalidated...
    size_t invalidUtts = 0;
    foreach_index(i, isValid)
    {
    if (!isValid[i])
    {
    invalidUtts++;
    }
    }
    assert(invalidUtts == 0); // For us it's zero

    if (invalidUtts > isValid.size() / 2)
    {
    RuntimeError("minibatchutterancesource: too many files with inconsistent durations, assuming broken configuration\n");
    }
    else if (invalidUtts > 0)
    {
    fprintf(stderr,
    "Found inconsistent durations across feature streams in %d out of %d files\n",
    static_cast<int>(invalidUtts),
    static_cast<int>(isValid.size()));
    }*/

    return deserializers;
}

template <class ElemType>
void NewHTKMLFReaderShim<ElemType>::Init(const ConfigParameters& config)
{
    m_layout = make_shared<MBLayout>();

    assert(config(L"frameMode", true));
    m_memoryProvider = std::make_shared<HeapMemoryProvider>();

    size_t window = ConfigHelper::GetRandomizationWindow(config);

    auto deserializers = CreateDeserializers(config, true, sizeof(ElemType));
    assert(deserializers.size() == 2);

    auto bundler = std::make_shared<Bundler>(config, deserializers[0], deserializers);
    m_streams = bundler->GetStreamDescriptions();

    std::wstring readMethod = ConfigHelper::GetRandomizer(config);
    if (_wcsicmp(readMethod.c_str(), L"blockRandomize"))
    {
        RuntimeError("readMethod must be 'blockRandomize'");
    }

    m_verbosity = config(L"verbosity", 2);
    m_transformer = std::make_shared<LegacyBlockRandomizer>(m_verbosity, window, bundler);

    intargvector numberOfuttsPerMinibatchForAllEpochs =
        config(L"nbruttsineachrecurrentiter", ConfigParameters::Array(intargvector(vector<int>{1})));
    Utils::CheckMinibatchSizes(numberOfuttsPerMinibatchForAllEpochs);

    // (SGD will ask before entering actual reading --TODO: This is hacky.)
    /*m_numSeqsPerMB = m_numSeqsPerMBForAllEpochs[0];
    m_pMBLayout->Init(m_numSeqsPerMB, 0);
    m_noData = false;*/

    if (config.Exists(L"legacyMode"))
        RuntimeError("legacy mode has been deprecated\n");

    // eldak: we should introduce a separate class describing inputs with proper interface.
    /*for (size_t i = 0; i < m_streams.size(); ++i)
    {
        m_nameToId.insert(std::make_pair(m_streams[i]->m_name, m_streams[i]->m_id));
    }*/

    //size_t iFeat = 0, iLabel = 0;

    std::vector<std::wstring> featureNames;
    std::vector<std::wstring> labelNames;
    std::vector<std::wstring> notused;

    ConfigHelper::GetDataNamesFromConfig(config, featureNames, labelNames, notused, notused);

    /*intargvector numberOfuttsPerMinibatchForAllEpochs =
        config(L"nbruttsineachrecurrentiter", ConfigParameters::Array(intargvector(vector<int>{1})));*/

    auto numSeqsPerMBForAllEpochs = numberOfuttsPerMinibatchForAllEpochs;
    m_layout->Init(numSeqsPerMBForAllEpochs[0], 0);
    m_streams = m_transformer->GetStreamDescriptions();
}

template <class ElemType>
void NewHTKMLFReaderShim<ElemType>::StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples)
{
    return StartDistributedMinibatchLoop(mbSize, epoch, 0, 1, requestedEpochSamples);
}

template <class ElemType>
void NewHTKMLFReaderShim<ElemType>::StartDistributedMinibatchLoop(size_t requestedMBSize, size_t epoch, size_t subsetNum, size_t numSubsets, size_t requestedEpochSamples /*= requestDataSize*/)
{
    EpochConfiguration config;
    config.m_workerRank = subsetNum;
    config.m_numberOfWorkers = numSubsets;
    config.m_minibatchSizeInSamples = requestedMBSize;
    config.m_totalEpochSizeInSamples = requestedEpochSamples;
    config.m_epochIndex = epoch;
    m_endOfEpoch = false;

    m_transformer->StartEpoch(config);
    m_packer = std::make_shared<SampleModePacker>(m_memoryProvider, m_transformer, requestedMBSize, m_streams);
}

template <class ElemType>
bool NewHTKMLFReaderShim<ElemType>::GetMinibatch(std::map<std::wstring, Matrix<ElemType>*>& matrices)
{
    // eldak: Hack.
    if (m_endOfEpoch)
    {
        return false;
    }

    int deviceId = matrices.begin()->second->GetDeviceId();
    for (auto mx : matrices)
    {
        if (mx.second->GetDeviceId() != deviceId)
        {
            assert(false);
        }
    }

    Minibatch m = m_packer->ReadMinibatch();
    if (m.m_endOfEpoch)
    {
        m_endOfEpoch = true;
        if (m.m_data.empty())
        {
            return false;
        }
    }

    auto streams = m_transformer->GetStreamDescriptions();
    std::map<size_t, wstring> idToName;
    for (auto i : streams)
    {
        idToName.insert(std::make_pair(i->m_id, i->m_name));
    }

    for (int i = 0; i < m.m_data.size(); i++)
    {
        const auto& stream = m.m_data[i];
        const std::wstring& name = idToName[i];
        if (matrices.find(name) == matrices.end())
        {
            continue;
        }

        // Current hack.
        m_layout = stream->m_layout;
        size_t columnNumber = m_layout->GetNumCols();
        size_t rowNumber = m_streams[i]->m_sampleLayout->GetNumElements();

        auto data = reinterpret_cast<const ElemType*>(stream->m_data);
        matrices[name]->SetValue(rowNumber, columnNumber, matrices[name]->GetDeviceId(), const_cast<ElemType*>(data), matrixFlagNormal);
    }

    return !m.m_data.empty();
}

template <class ElemType>
bool NewHTKMLFReaderShim<ElemType>::DataEnd(EndDataType /*endDataType*/)
{
    return false;
}

template <class ElemType>
void NewHTKMLFReaderShim<ElemType>::CopyMBLayoutTo(MBLayoutPtr layout)
{
    layout->CopyFrom(m_layout);
}

template <class ElemType>
size_t NewHTKMLFReaderShim<ElemType>::GetNumParallelSequences()
{
    return m_layout->GetNumParallelSequences(); // (this function is only used for validation anyway)
}

template class NewHTKMLFReaderShim<float>;
template class NewHTKMLFReaderShim<double>;

}}}
