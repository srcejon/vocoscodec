add_library(vocoscodec_codec STATIC
  codec.cc
  codec.h
  concealment.cc
  concealment.h
  decoder.cc
  decoder.h
  encoder.cc
  encoder.h
  fft.cc
  fft.h
  half.h
  kernels.cc
  kernels.h
  kernels_internal.h
  model_file.cc
  model_file.h
  packet.cc
  packet.h
  weights.h)
add_library(vocoscodec::codec ALIAS vocoscodec_codec)
if(VOCOSCODEC_HALF_WEIGHTS)
  target_compile_definitions(vocoscodec_codec PUBLIC VOCOSCODEC_HALF_WEIGHTS)
endif()

# SIMD kernels, chosen at run time after a CPU check: on x86-64 AVX2 + FMA +
# F16C, else SSE4.2; NEON on arm64; portable C++ elsewhere. Only the files
# holding those kernels are compiled for the newer instruction sets.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|x64)$")
  target_sources(vocoscodec_codec PRIVATE kernels_avx2.cc kernels_sse.cc)
  target_compile_definitions(vocoscodec_codec PRIVATE
    VOCOSCODEC_HAVE_AVX2 VOCOSCODEC_HAVE_SSE42)
  if(MSVC)
    # MSVC x64 compiles SSE4 intrinsics without an /arch option.
    set_source_files_properties(kernels_avx2.cc PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  else()
    set_source_files_properties(kernels_avx2.cc PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    set_source_files_properties(kernels_sse.cc PROPERTIES COMPILE_OPTIONS "-msse4.2")
  endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
  target_sources(vocoscodec_codec PRIVATE kernels_neon.cc)
  target_compile_definitions(vocoscodec_codec PRIVATE VOCOSCODEC_HAVE_NEON)
endif()

target_include_directories(vocoscodec_codec PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(vocoscodec_codec PUBLIC cxx_std_17)
set_target_properties(vocoscodec_codec PROPERTIES
  OUTPUT_NAME vocoscodec_codec
  EXPORT_NAME codec
  POSITION_INDEPENDENT_CODE ON
  CXX_EXTENSIONS OFF
  DEBUG_POSTFIX d)
if(MSVC)
  target_compile_options(vocoscodec_codec PRIVATE /W4 /wd4324)
else()
  target_compile_options(vocoscodec_codec PRIVATE -Wall -Wextra)
endif()

if(VOCOSCODEC_BUILD_RUNTIME)
  add_library(vocoscodec_runtime SHARED c_api.cc c_api.h)
  add_library(vocoscodec::runtime ALIAS vocoscodec_runtime)
  target_include_directories(vocoscodec_runtime PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  target_compile_features(vocoscodec_runtime PUBLIC cxx_std_17)
  target_compile_definitions(vocoscodec_runtime PRIVATE
    VOCOSCODEC_RUNTIME_BUILD
    VOCOSCODEC_VERSION="${PROJECT_VERSION}"
    VOCOSCODEC_RUNTIME_MODEL_DIR="${VOCOSCODEC_RUNTIME_MODEL_DIR}")
  target_link_libraries(vocoscodec_runtime PRIVATE vocoscodec_codec)
  set_target_properties(vocoscodec_runtime PROPERTIES
    OUTPUT_NAME vocoscodec
    EXPORT_NAME runtime
    CXX_EXTENSIONS OFF
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    DEBUG_POSTFIX d)
endif()

if(VOCOSCODEC_BUILD_TESTS)
  set(model_dir "${PROJECT_SOURCE_DIR}/model")
  set(golden "${PROJECT_SOURCE_DIR}/testdata/golden.bin")

  add_executable(vocoscodec_kernels_test tests/kernels_test.cc)
  target_link_libraries(vocoscodec_kernels_test PRIVATE vocoscodec_codec)
  add_test(NAME vocoscodec_kernels_test COMMAND vocoscodec_kernels_test)

  add_executable(vocoscodec_golden_test tests/golden_test.cc)
  target_link_libraries(vocoscodec_golden_test PRIVATE vocoscodec_codec)
  add_test(NAME vocoscodec_golden_test COMMAND vocoscodec_golden_test "${model_dir}" "${golden}")
  add_test(NAME vocoscodec_golden_generic_test
    COMMAND vocoscodec_golden_test "${model_dir}" "${golden}" --kernels generic)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|x64)$")
    add_test(NAME vocoscodec_golden_sse42_test
      COMMAND vocoscodec_golden_test "${model_dir}" "${golden}" --kernels sse4.2)
  endif()

  add_executable(vocoscodec_codec_test tests/codec_test.cc)
  target_link_libraries(vocoscodec_codec_test PRIVATE vocoscodec_codec)
  add_test(NAME vocoscodec_codec_test COMMAND vocoscodec_codec_test "${model_dir}" "${golden}")

  add_executable(vocoscodec_packet_loss_test tests/packet_loss_test.cc)
  target_link_libraries(vocoscodec_packet_loss_test PRIVATE vocoscodec_codec)
  add_test(NAME vocoscodec_packet_loss_test
    COMMAND vocoscodec_packet_loss_test "${model_dir}" "${golden}")

  if(VOCOSCODEC_BUILD_RUNTIME)
    add_executable(vocoscodec_c_api_test tests/c_api_test.cc)
    target_link_libraries(vocoscodec_c_api_test PRIVATE vocoscodec_runtime vocoscodec_codec)
    add_test(NAME vocoscodec_c_api_test COMMAND vocoscodec_c_api_test "${model_dir}" "${golden}")

    add_executable(vocoscodec_runtime_load_test tests/runtime_load_test.cc)
    target_compile_features(vocoscodec_runtime_load_test PRIVATE cxx_std_17)
    target_link_libraries(vocoscodec_runtime_load_test PRIVATE ${CMAKE_DL_LIBS})
    add_test(NAME vocoscodec_runtime_load_test
      COMMAND vocoscodec_runtime_load_test $<TARGET_FILE:vocoscodec_runtime> "${model_dir}")
  endif()
endif()

if(VOCOSCODEC_BUILD_TOOLS)
  add_library(vocoscodec_wav_io STATIC tools/wav_io.cc tools/wav_io.h)
  target_include_directories(vocoscodec_wav_io PRIVATE "${PROJECT_SOURCE_DIR}")
  target_compile_features(vocoscodec_wav_io PUBLIC cxx_std_17)
  set_target_properties(vocoscodec_wav_io PROPERTIES CXX_EXTENSIONS OFF)

  add_executable(vocoscodec_encode tools/encode_main.cc)
  target_link_libraries(vocoscodec_encode PRIVATE vocoscodec_codec vocoscodec_wav_io)

  add_executable(vocoscodec_decode tools/decode_main.cc)
  target_link_libraries(vocoscodec_decode PRIVATE vocoscodec_codec vocoscodec_wav_io)

  add_executable(vocoscodec_benchmark tools/benchmark_main.cc)
  target_link_libraries(vocoscodec_benchmark PRIVATE vocoscodec_codec vocoscodec_wav_io)

  install(TARGETS vocoscodec_encode vocoscodec_decode vocoscodec_benchmark
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
endif()

install(TARGETS vocoscodec_codec EXPORT VocoscodecTargets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
if(VOCOSCODEC_BUILD_RUNTIME)
  install(TARGETS vocoscodec_runtime EXPORT VocoscodecTargets
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
endif()
install(FILES codec.h c_api.h
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/vocoscodec")
install(FILES
  "${PROJECT_SOURCE_DIR}/model/encodec_encoder.bin"
  "${PROJECT_SOURCE_DIR}/model/encodec_codebooks.bin"
  "${PROJECT_SOURCE_DIR}/model/vocos_decoder.bin"
  "${PROJECT_SOURCE_DIR}/model/SHA256SUMS"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/vocoscodec/model")

include(CMakePackageConfigHelpers)
configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/VocoscodecConfig.cmake.in"
  "${PROJECT_BINARY_DIR}/VocoscodecConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Vocoscodec")
write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/VocoscodecConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion)
install(EXPORT VocoscodecTargets
  FILE VocoscodecTargets.cmake
  NAMESPACE vocoscodec::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Vocoscodec")
install(FILES
  "${PROJECT_BINARY_DIR}/VocoscodecConfig.cmake"
  "${PROJECT_BINARY_DIR}/VocoscodecConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Vocoscodec")
