//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "Reader.h"
#include "SampleModePacker.h"
#include "LegacyBlockRandomizer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    // Implementation of the HTKMLF reader.
    // Currently represents a factory for connecting the packer,
    // transformers and deserializer together.
    class HTKMLFReader : public Reader
    {
    public:
        HTKMLFReader(MemoryProviderPtr provider,
            const ConfigParameters& parameters);

        // Description of streams that this reader provides.
        std::vector<StreamDescriptionPtr> GetStreamDescriptions() override;

        // Starts a new epoch with the provided configuration.
        void StartEpoch(const EpochConfiguration& config) override;

        // Reads a single minibatch.
        Minibatch ReadMinibatch() override;

    private:
        // All streams this reader provides.
        std::vector<StreamDescriptionPtr> m_streams;

        // Packer.
        SampleModePackerPtr m_packer;

        // Seed for the random generator.
        unsigned int m_seed;

        // Memory provider (TODO: this will possibly change in the near future.)
        MemoryProviderPtr m_provider;

        std::shared_ptr<LegacyBlockRandomizer> m_randomizer;
    };

}}}
