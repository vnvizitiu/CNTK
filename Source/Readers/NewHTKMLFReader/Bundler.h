//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "DataDeserializer.h"
#include "Config.h" // for ConfigParameters

namespace Microsoft { namespace MSR { namespace CNTK {

class Bundler : public IDataDeserializer
{
public:
    Bundler(const ConfigParameters& readerConfig, IDataDeserializerPtr driver, std::vector<IDataDeserializerPtr> deserializers);

    virtual const SequenceDescriptions& GetSequenceDescriptions() const override;
    virtual std::vector<StreamDescriptionPtr> GetStreamDescriptions() const override;
    virtual ChunkPtr GetChunk(size_t) override;

private:
    void CreateSequenceDescriptions();

    void operator=(const Bundler& other); // non-assignable

    std::vector<StreamDescriptionPtr> m_streams;
    std::vector<IDataDeserializerPtr> m_deserializers;
    IDataDeserializerPtr m_driver;

    std::vector<SequenceDescription> m_sequenceDescriptions;
    std::vector<std::map<size_t, size_t>> m_sequenceToChunk;
    std::vector<size_t> m_chunkOffsets;
    SequenceDescriptions m_sequences;
};

}}}
