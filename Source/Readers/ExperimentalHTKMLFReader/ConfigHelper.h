//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include <utility>
#include <string>
#include <vector>
#include "Config.h"
#include "Reader.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    // Simple config helper - a HtkMlf wrapper around config parameters.
    class ConfigHelper
    {
    public:
        ConfigHelper(const ConfigParameters& config) : m_config(config)
        {}

        std::pair<size_t, size_t> GetContextWindow();
        size_t GetFeatureDimension();
        size_t GetLabelDimension();

        ElementType GetElementType();

        void CheckFeatureType();
        void CheckLabelType();

        void GetDataNamesFromConfig(
            std::vector<std::wstring>& features,
            std::vector<std::wstring>& labels,
            std::vector<std::wstring>& hmms,
            std::vector<std::wstring>& lattices);

        std::vector<std::wstring> GetMlfPaths();
        std::vector<std::wstring> GetFeaturePaths();

        size_t GetRandomizationWindow();
        std::wstring GetRandomizer();

        intargvector GetNumberOfUtterancesPerMinibatchForAllEppochs();

    private:
        DISABLE_COPY_AND_MOVE(ConfigHelper);

        void ExpandDotDotDot(std::wstring& featPath, const std::wstring& scpPath, std::wstring& scpDirCached);

        const ConfigParameters& m_config;
    };

}}}
