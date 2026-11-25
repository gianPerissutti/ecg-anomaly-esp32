idf_component_register(
    SRCS "main.cc" "neural_network.cc" "model.cc"
    INCLUDE_DIRS "."
    PRIV_REQUIRES esp_driver_uart
    REQUIRED_IDF_TARGETS esp32s3
)
