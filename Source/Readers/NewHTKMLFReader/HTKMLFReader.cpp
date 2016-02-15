//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "HTKMLFReader.h"
#include "Config.h"
#include "BlockRandomizer.h"
#include "NoRandomizer.h"
#include "HTKDataDeserializer.h"
#include "MLFDataDeserializer.h"
#include <omp.h>
#include "ConfigHelper.h"
#include <HeapMemoryProvider.h>
#include "Bundler.h"
#include <StringUtil.h>
#include <LegacyBlockRandomizer.h>
#include "Utils.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    std::vector<IDataDeserializerPtr> CreateDeserializers(const ConfigParameters& config)
    {
        std::vector<std::wstring> featureNames;
        std::vector<std::wstring> labelNames;
        std::vector<std::wstring> notused;
        ConfigHelper::GetDataNamesFromConfig(config, featureNames, labelNames, notused, notused);
        if (featureNames.size() < 1 || labelNames.size() < 1)
        {
            InvalidArgument("Network needs at least 1 feature and 1 label specified.");
        }

        std::vector<HTKDataDeserializerPtr> featureDeserializers;
        std::vector<MLFDataDeserializerPtr> labelDeserializers;
        CorpusDescriptorPtr corpus = std::make_shared<CorpusDescriptor>();
        for (const auto& featureName : featureNames)
        {
            auto deserializer = std::make_shared<HTKDataDeserializer>(corpus, config(featureName), featureName);
            featureDeserializers.push_back(deserializer);
        }
        assert(featureDeserializers.size() == 1);

        for (const auto& labelName : labelNames)
        {
            auto deserializer = std::make_shared<MLFDataDeserializer>(corpus, config(labelName), labelName);

            labelDeserializers.push_back(deserializer);
        }
        assert(labelDeserializers.size() == 1);

        std::vector<IDataDeserializerPtr> deserializers;
        deserializers.insert(deserializers.end(), featureDeserializers.begin(), featureDeserializers.end());
        deserializers.insert(deserializers.end(), labelDeserializers.begin(), labelDeserializers.end());
        return deserializers;
    }

    HTKMLFReader::HTKMLFReader(MemoryProviderPtr provider,
        const ConfigParameters& config)
        : m_seed(0), m_provider(provider)
    {
        // In the future, deserializers and transformers will be dynamically loaded
        // from external libraries based on the configuration/brain script.
        // We will provide ability to implement the transformer and
        // deserializer interface not only in C++ but in scripting languages as well.

        assert(config(L"frameMode", true));

        size_t window = ConfigHelper::GetRandomizationWindow(config);
        auto deserializers = CreateDeserializers(config);
        assert(deserializers.size() == 2);

        auto bundler = std::make_shared<Bundler>(config, deserializers[0], deserializers);

        std::wstring readMethod = ConfigHelper::GetRandomizer(config);
        if (!AreEqualIgnoreCase(readMethod, std::wstring(L"blockRandomize")))
        {
            RuntimeError("readMethod must be 'blockRandomize'");
        }

        int verbosity = config(L"verbosity", 2);
        m_randomizer = std::make_shared<LegacyBlockRandomizer>(verbosity, window, bundler);
        m_randomizer->Initialize(nullptr, config);

        intargvector numberOfuttsPerMinibatchForAllEpochs =
            config(L"nbruttsineachrecurrentiter", ConfigParameters::Array(intargvector(vector<int>{1})));
        Utils::CheckMinibatchSizes(numberOfuttsPerMinibatchForAllEpochs);

        m_streams = m_randomizer->GetStreamDescriptions();
    }

    std::vector<StreamDescriptionPtr> HTKMLFReader::GetStreamDescriptions()
    {
        assert(!m_streams.empty());
        return m_streams;
    }

    void HTKMLFReader::StartEpoch(const EpochConfiguration& config)
    {
        if (config.m_totalEpochSizeInSamples <= 0)
        {
            RuntimeError("Unsupported minibatch size '%d'.", (int)config.m_totalEpochSizeInSamples);
        }

        m_randomizer->StartEpoch(config);
        m_packer = std::make_shared<SampleModePacker>(
            m_provider,
            m_randomizer,
            config.m_minibatchSizeInSamples,
            m_streams);
    }

    Minibatch HTKMLFReader::ReadMinibatch()
    {
        assert(m_packer != nullptr);
        return m_packer->ReadMinibatch();
    }

}}}
