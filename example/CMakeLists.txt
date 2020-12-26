add_executable(testdedisp
    testdedisp.c
    gasdev.c
    ran1.c)

target_include_directories(testdedisp PRIVATE ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(testdedisp ${PROJECT_NAME} m)

install(TARGETS testdedisp
        DESTINATION ${PROJECT_SOURCE_DIR}/example)