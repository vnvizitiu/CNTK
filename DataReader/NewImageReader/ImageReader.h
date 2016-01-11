//
// <copyright company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#pragma once

#include "Reader.h"
#include "ImageTransformers.h"
#include "FrameModePacker.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    class ImageReader : public Reader
    {
    public:
        ImageReader(MemoryProviderPtr provider,
            const ConfigParameters& parameters,
            ElementType elementType);

        std::vector<StreamDescriptionPtr> GetStreams() override;
        void StartEpoch(const EpochConfiguration& config) override;
        Minibatch ReadMinibatch() override;

    private:
        std::vector<StreamDescriptionPtr> m_streams;
        TransformerPtr m_transformer;
        FrameModePackerPtr m_packer;
        MemoryProviderPtr m_provider;
        unsigned int m_seed;
        ElementType m_elementType;
    };
}}}
