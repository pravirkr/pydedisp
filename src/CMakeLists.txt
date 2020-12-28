# dedisp shared library
add_library(${PROJECT_NAME} SHARED ${PROJECT_SOURCE_DIR}/src/dedisp.cu)
set_target_properties(${PROJECT_NAME}
    PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/src/dedisp.h
        CUDA_ARCHITECTURES OFF
)

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/src)

install(TARGETS ${PROJECT_NAME}
	LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/lib
    PUBLIC_HEADER DESTINATION ${PROJECT_SOURCE_DIR}/include
)
