file(GLOB RVTV_SRC "*.cpp")
add_executable(rvtv ${RVTV_SRC})

if(NOT LLVM_SOURCE_DIR)
  message(FATAL_ERROR "Please specify LLVM_SOURCE_DIR")
endif()

if(NOT LLVM_BUILD_DIR)
  message(FATAL_ERROR "Please specify LLVM_BUILD_DIR")
endif()

message(STATUS "LLVM_SOURCE_DIR: " ${LLVM_SOURCE_DIR})
message(STATUS "LLVM_BUILD_DIR: " ${LLVM_BUILD_DIR})

include_directories(${LLVM_SOURCE_DIR})
include_directories(${LLVM_BUILD_DIR}/include)
include_directories(${LLVM_SOURCE_DIR}/llvm/lib/Target/RISCV)
include_directories(${LLVM_BUILD_DIR}/lib/Target/RISCV)

llvm_map_components_to_libnames(LLVM_LIBS support core analysis passes RISCV codegen mc targetparser irreader transformutils)
target_link_libraries(rvtv PRIVATE ${LLVM_LIBS})
