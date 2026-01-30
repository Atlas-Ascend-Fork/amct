
# 开启debug
set(CMAKE_VERBOSE_MAKEFILE on)

set(AMCT_PYTORCH_INNER_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(AMCT_TORCH_PROTO_DIR ${AMCT_PYTORCH_INNER_DIR}/amct_pytorch/proto)
set(AMCT_TORCH_OP_DIR ${AMCT_PYTORCH_INNER_DIR}/amct_pytorch/custom_op)
set(PYTHON_INCLUDE ${HI_PYTHON_INC})
set(AMCT_TORCH_TMP_DIR ${AMCT_PYTORCH_INNER_DIR}/amct_pytorch/custom_op/tmp)
set(AMCT_TORCH_BUILD_DIR ${BASE_DIR}/build)

# compile op
add_subdirectory(custom_op)
add_subdirectory(amct_tensor_decompose)

message("-------pytorch -----")
if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
  set(PROTOC_PROGRAM ${AMCT_DIR}/build/protobuf_host/bin/protoc)
  message("--- torch protoc ----- ${PROTOC_PROGRAM}")
  set(HI_PYTHON python3)
endif()

# compile proto
add_custom_target(amct_pytorch_proto_compile
  COMMAND cd ${AMCT_PYTORCH_INNER_DIR} && ${PROTOC_PROGRAM} -I=./ --python_out=./ amct_pytorch/proto/basic_info.proto
  COMMAND cd ${AMCT_PYTORCH_INNER_DIR} && ${PROTOC_PROGRAM} -I=./ --python_out=./ amct_pytorch/proto/calibration_config_pytorch.proto
  COMMAND cd ${AMCT_PYTORCH_INNER_DIR} && ${PROTOC_PROGRAM} -I=./ --python_out=./ amct_pytorch/proto/scale_offset_record_pytorch.proto
  COMMAND cd ${AMCT_PYTORCH_INNER_DIR} && ${PROTOC_PROGRAM} -I=./ --python_out=./ amct_pytorch/proto/post_quant_record_pytorch.proto
  COMMAND cd ${AMCT_PYTORCH_INNER_DIR} && ${PROTOC_PROGRAM} -I=./ --python_out=./ amct_pytorch/proto/retrain_config_pytorch.proto
  COMMAND cd ${AMCT_TORCH_PROTO_DIR} && ${PROTOC_PROGRAM} --python_out=./ distill_config_pytorch.proto
  COMMAND cd ${AMCT_TORCH_PROTO_DIR} && ${PROTOC_PROGRAM} --python_out=./ post_quant_config_pytorch.proto
  COMMAND cd ${AMCT_TORCH_PROTO_DIR} && ${PROTOC_PROGRAM} --python_out=./ quant_calibration_config_pytorch.proto
)
if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
add_dependencies(amct_pytorch_proto_compile protobuf_host_build ascend_protobuf_static)
endif()

set(COMMON_LIB_INC_FILES
  nuq.h util.h dump.h cast_util.h error_codes.h
)

add_custom_target(amct_pytorch_prepare_package
  DEPENDS amct_pytorch_lib
  # add common header files
  COMMAND cd ${AMCT_PYTORCH_INNER_DIR}/common_cpp/quantize_lib/inc/ && cp ${COMMON_LIB_INC_FILES} ${AMCT_PYTORCH_INNER_DIR}/custom_op/inc
)
