//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// DataReader.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "Basics.h"

#define DATAREADER_EXPORTS
#include "DataReader.h"
#include "Config.h"
#include "ReaderShim.h"
#include "HTKMLFReader.h"
#include "HeapMemoryProvider.h"
#include <CudaMemoryProvider.h>

namespace Microsoft { namespace MSR { namespace CNTK {

    auto factory = [](const ConfigParameters& parameters) -> ReaderPtr
    {
        return std::make_shared<HTKMLFReader>(std::make_shared<CudaMemoryProvider>(0), parameters);
    };

    extern "C" DATAREADER_API void GetReaderF(IDataReader<float>** preader)
    {
        *preader = new ReaderShim<float>(factory);
    }

    extern "C" DATAREADER_API void GetReaderD(IDataReader<double>** preader)
    {
        *preader = new ReaderShim<double>(factory);
    }

#ifdef _WIN32
// Utility function, in ConfigFile.cpp, but NewHTKMLFReader doesn't need that code...

// Trim - trim white space off the start and end of the string
// str - string to trim
// NOTE: if the entire string is empty, then the string will be set to an empty string
void Trim(std::string& str)
{
    auto found = str.find_first_not_of(" \t");
    if (found == npos)
    {
        str.erase(0);
        return;
    }
    str.erase(0, found);
    found = str.find_last_not_of(" \t");
    if (found != npos)
        str.erase(found + 1);
}
#endif
} } }
