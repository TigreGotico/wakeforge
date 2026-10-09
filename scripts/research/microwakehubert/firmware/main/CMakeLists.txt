idf_component_register(SRCS "main.cc" "logmel.c" "model_data.cc" "test_audio.cc"
                       INCLUDE_DIRS ".")
target_compile_options(${COMPONENT_LIB} PRIVATE -O2)
