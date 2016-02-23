//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"

using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

struct ImageReaderFixture : ReaderFixture
{
    ImageReaderFixture() : ReaderFixture("Data/Image", "")
    {
    }
};

BOOST_FIXTURE_TEST_SUITE(ImageReaderTestSuite, ImageReaderFixture)

BOOST_AUTO_TEST_CASE(ImageReaderCheckSeveralMinibatches)
{
    string configFileName = testDataPath() + "/Config/Image/ReaderConfig.cntk";
    string controlDataFilePath = testDataPath() + "/Control/ImageReaderCheckSeveralMinibatches_Control.txt";
    string outputDataFilePath = testDataPath() + "/Control/ImageReaderCheckSeveralMinibatches_Output.txt";
    string actionSectionName = "Train";
    string readerSectionName = "reader";
    size_t epochSize = 7;
    size_t minibatchSize = 7;
    size_t epochs = 1;
    size_t numberOfFeatureStreams = 1;
    size_t numberOfLabelStreams = 1;
    size_t workerRank = 0;
    size_t numberOfWorkers = 1;

    HelperRunReaderTest<float>(configFileName,
                               controlDataFilePath,
                               outputDataFilePath,
                               actionSectionName,
                               readerSectionName,
                               epochSize,
                               minibatchSize,
                               epochs,
                               numberOfFeatureStreams,
                               numberOfLabelStreams,
                               workerRank,
                               numberOfWorkers);
};

BOOST_AUTO_TEST_SUITE_END()

}}}}
