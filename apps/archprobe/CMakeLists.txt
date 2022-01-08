set(APP_NAME ArchProbe)

add_executable(${APP_NAME} "app.cpp" "env.cpp")
target_link_libraries(${APP_NAME} ArchProbeCore)
