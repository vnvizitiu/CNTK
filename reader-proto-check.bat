@echo off
setlocal
cd %~dp0
for %%f in (%*) do set a_%%f=1

if not defined CNTK_ENABLE_1BitSGD echo Enable 1-bit SGD, cf. https://github.com/Microsoft/CNTK/wiki/Enabling-1bit-SGD&exit /b 1
@REM optionally do clean ?

set ACML_FMA=0
set CYGWIN_BIN=c:\cygwin64\bin
if not exist %CYGWIN_BIN% (
    set CYGWIN_BIN=c:\cygwin\bin
    if not exist %CYGWIN_BIN% (
        echo Can't find Cygwin, is it installed?
        exit /b 1
    )
)
echo on

set UNIT_TEST_SPEC=^
  -t ReaderTestSuite/NewHTKMLFReaderSimpleDataLoop1 ^
  -t +ReaderTestSuite/NewHTKMLFReaderSimpleDataLoop5 ^
  -t +ReaderTestSuite/NewHTKMLFReaderSimpleDataLoop11 ^
  -t +ReaderTestSuite/NewHTKMLFReaderSimpleDataLoop21_0 ^
  -t +ReaderTestSuite/NewHTKMLFReaderSimpleDataLoop21_1

set END2END_TEST_SPEC=^
    Speech/DNN/DiscriminativePreTraining2 ^
    Speech/DNN/ParallelNoQuantization2 ^
    Speech/DNN/ParallelNoQuantizationBufferedAsyncGradientAggregation2 ^
    Speech/QuickE2E2 ^
    Speech/SVD2

set END2END_TEST_SPEC_1B=^
    Speech/DNN/Parallel1BitQuantization2 ^
    Speech/DNN/ParallelBufferedAsyncGradientAggregation2

call "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat"
if errorlevel 1 exit /b 1

if not defined a_nodebug (
    msbuild /m /p:Platform=x64 /p:Configuration=Debug CNTK.sln
    if errorlevel 1 exit /b 1

if not defined a_nocpuonly (
    msbuild /m /p:Platform=x64 /p:Configuration=Debug_CpuOnly CNTK.sln
    if errorlevel 1 exit /b 1
)

if not defined a_notests (
if not defined a_nounittests (
        .\x64\Debug\UnitTests\ReaderTests.exe %UNIT_TEST_SPEC%
        if errorlevel 1 exit /b 1
)
)
)

if not defined a_norelease (
    msbuild /m /p:Platform=x64 /p:Configuration=Release CNTK.sln
    if errorlevel 1 exit /b 1

if not defined a_nocpuonly (
    msbuild /m /p:Platform=x64 /p:Configuration=Release_CpuOnly CNTK.sln
    if errorlevel 1 exit /b 1
)

if not defined a_notests (
if not defined a_nounittests (
    .\x64\Release\UnitTests\ReaderTests.exe %UNIT_TEST_SPEC%
    if errorlevel 1 exit /b 1
)
)
)

set PATH=%PATH%;%CYGWIN_BIN%

if not defined a_nospeech (
if not defined a_noe2e (
if not defined a_notests (
if not defined a_norelease (
if not defined a_nogpu (
    python2.7.exe Tests/EndToEndTests/TestDriver.py run -t nightly -d gpu -f release -s 1bitsgd %END2END_TEST_SPEC_1B%
    python2.7.exe Tests/EndToEndTests/TestDriver.py run -t nightly -d gpu -f release %END2END_TEST_SPEC%
    if errorlevel 1 exit /b 1
)

if not defined a_nocpu (
    python2.7.exe Tests/EndToEndTests/TestDriver.py run -t nightly -d cpu -f release %END2END_TEST_SPEC%
    if errorlevel 1 exit /b 1
)
)

if not defined a_nodebug (
if not defined a_nogpu (
    python2.7.exe Tests/EndToEndTests/TestDriver.py run -t nightly -d gpu -f debug -s 1bitsgd %END2END_TEST_SPEC_1B%
    python2.7.exe Tests/EndToEndTests/TestDriver.py run -t nightly -d gpu -f debug %END2END_TEST_SPEC%
    if errorlevel 1 exit /b 1
)

if not defined a_nocpu (
    python2.7.exe Tests/EndToEndTests/TestDriver.py run -t nightly -d cpu -f debug %END2END_TEST_SPEC%
    if errorlevel 1 exit /b 1
)
)
)
)
)
