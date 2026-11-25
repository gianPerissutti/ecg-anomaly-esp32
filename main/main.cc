#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "driver/uart.h"
#include "dsps_fir.h"
#include "esp_err.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "neural_network.h"

namespace {

constexpr uart_port_t kUart = UART_NUM_0;
constexpr int kUartBaudRate = 115200;
constexpr int kUartTxPin = 43;
constexpr int kUartRxPin = 44;
constexpr int kUartReadTimeoutMs = 20;
constexpr int kSamplesPerBlock = 930;
constexpr int kSamplesBeforePeak = 126;
constexpr int kSamplesToSkipAfterPeak = 250;
constexpr float kPeakThreshold = 4e16f;
constexpr int kFirSignalTaps = 166;
constexpr int kFirTaps = kFirSignalTaps + 2;
constexpr int kFirDelaySamples = kFirSignalTaps / 2;
constexpr float kDerivativeTaps[] = {1.8f, 3.6f, 0.0f, 3.6f, 1.8f};
constexpr int kDerivativeTapCount = sizeof(kDerivativeTaps) / sizeof(kDerivativeTaps[0]);
constexpr char kTag[] = "ECG";

static_assert(sizeof(float) == 4);

// Two leading zeros preserve the filter's delay and make the tap count a multiple of four.
alignas(16) float fir_coeffs[kFirTaps] = {
    0.0f, 0.0f,
    0.04612394795, -0.009035871364, -0.008109617047, -0.007214183453, -0.006331882905,
    -0.005457166117, -0.004566338379, -0.003655595472, -0.002709696535, -0.001729679992,
    -0.0007101712981, 0.0003325945872, 0.001394866966, 0.002444523619, 0.00346738915,
    0.004418040626, 0.00528076943, 0.006001201458, 0.006568639539, 0.006927566137,
    0.007084739394, 0.006981942803, 0.006636795122, 0.006102534477, 0.005227134563,
    0.004214514513, 0.003027746454, 0.001703438116, 0.0002666138171, -0.001190845389,
    -0.00261410512, -0.003931258339, -0.005097680725, -0.00605944125, -0.006784807891,
    -0.007230259944, -0.007383060176, -0.007225668058, -0.006779009011, -0.006057185587,
    -0.0051189675, -0.004008492455, -0.002814643783, -0.001596869435, -0.0004646505986,
    0.0005156437983, 0.00126669649, 0.001675310894, 0.001713494421, 0.00131939922,
    0.0004537724599, -0.0009029207868, -0.002725079423, -0.004979110789, -0.007611145265,
    -0.01053853426, -0.01366079599, -0.01685412601, -0.01998092048, -0.02290024236,
    -0.02546765096, -0.02754794434, -0.02900485322, -0.02972797677, -0.0296139475,
    -0.02860241942, -0.02664270811, -0.02374350838, -0.0199210681, -0.01523789763,
    -0.009805934504, -0.00373735046, 0.002814912237, 0.009671472013, 0.01663221978,
    0.02350517176, 0.03008355014, 0.03615712374, 0.04154535383, 0.04607350379,
    0.04960471764, 0.05202403665, 0.05325509235, 0.05325509235, 0.05202403665,
    0.04960471764, 0.04607350379, 0.04154535383, 0.03615712374, 0.03008355014,
    0.02350517176, 0.01663221978, 0.009671472013, 0.002814912237, -0.00373735046,
    -0.009805934504, -0.01523789763, -0.0199210681, -0.02374350838, -0.02664270811,
    -0.02860241942, -0.0296139475, -0.02972797677, -0.02900485322, -0.02754794434,
    -0.02546765096, -0.02290024236, -0.01998092048, -0.01685412601, -0.01366079599,
    -0.01053853426, -0.007611145265, -0.004979110789, -0.002725079423, -0.0009029207868,
    0.0004537724599, 0.00131939922, 0.001713494421, 0.001675310894, 0.00126669649,
    0.0005156437983, -0.0004646505986, -0.001596869435, -0.002814643783, -0.004008492455,
    -0.0051189675, -0.006057185587, -0.006779009011, -0.007225668058, -0.007383060176,
    -0.007230259944, -0.006784807891, -0.00605944125, -0.005097680725, -0.003931258339,
    -0.00261410512, -0.001190845389, 0.0002666138171, 0.001703438116, 0.003027746454,
    0.004214514513, 0.005227134563, 0.006102534477, 0.006636795122, 0.006981942803,
    0.007084739394, 0.006927566137, 0.006568639539, 0.006001201458, 0.00528076943,
    0.004418040626, 0.00346738915, 0.002444523619, 0.001394866966, 0.0003325945872,
    -0.0007101712981, -0.001729679992, -0.002709696535, -0.003655595472, -0.004566338379,
    -0.005457166117, -0.006331882905, -0.007214183453, -0.008109617047, -0.009035871364,
    0.04612394795
};
alignas(16) float delay_line[kFirTaps + 4];
alignas(16) float ecg_block[kSamplesPerBlock];
alignas(16) float filtered[kSamplesPerBlock];
fir_f32_t filter;

void process_block()
{
    ESP_ERROR_CHECK(dsps_fir_f32(&filter, ecg_block, filtered, kSamplesPerBlock));

    // Account for the FIR delay when mapping filtered peaks to the raw ECG.
    constexpr int first_peak = kSamplesBeforePeak + kFirDelaySamples;
    constexpr int last_peak = kSamplesPerBlock - ecg::kWindowSamples + first_peak;
    static_assert(first_peak >= kDerivativeTapCount - 1 && last_peak < kSamplesPerBlock);
    for (int peak = first_peak; peak <= last_peak; ++peak) {
        float response = 0.0f;
        // The detection threshold depends on squaring after each tap.
        for (int tap = 0; tap < kDerivativeTapCount; ++tap) {
            response += kDerivativeTaps[tap] * filtered[peak - tap];
            response *= response;
        }
        if (response >= kPeakThreshold) {
            const float* window = ecg_block + (peak - first_peak);
            const float mae = ecg::reconstruction_error(window);
            std::printf("%f,", static_cast<double>(mae));
            std::fflush(stdout);
            peak += kSamplesToSkipAfterPeak;
        }
    }
}

}  // namespace

extern "C" void app_main(void)
{
    if (!ecg::initialize_model()) {
        ESP_LOGE(kTag, "Error al inicializar el modelo ECG");
        return;
    }
    ESP_ERROR_CHECK(dsps_fir_init_f32(&filter, fir_coeffs, delay_line, kFirTaps));

    uart_config_t uart_config{};
    uart_config.baud_rate = kUartBaudRate;
    uart_config.data_bits = UART_DATA_8_BITS;
    uart_config.parity = UART_PARITY_DISABLE;
    uart_config.stop_bits = UART_STOP_BITS_1;
    uart_config.flow_ctrl = UART_HW_FLOWCTRL_DISABLE;

    constexpr std::size_t kBlockBytes = sizeof(ecg_block);
    ESP_ERROR_CHECK(uart_driver_install(kUart, kBlockBytes * 2, 0, 0, nullptr, 0));
    ESP_ERROR_CHECK(uart_param_config(kUart, &uart_config));
    ESP_ERROR_CHECK(uart_set_pin(kUart, kUartTxPin, kUartRxPin, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
    ESP_LOGI(kTag, "ECG listo por UART, %d muestras por bloque", kSamplesPerBlock);

    // Receive little-endian float32 samples directly into the ESP32-S3 block buffer.
    auto* bytes = reinterpret_cast<std::uint8_t*>(ecg_block);
    std::size_t received = 0;
    while (true) {
        // A UART read can end inside a sample. Keep the bytes already received.
        const int count = uart_read_bytes(kUart, bytes + received, kBlockBytes - received,
                                         pdMS_TO_TICKS(kUartReadTimeoutMs));
        if (count <= 0) {
            continue;
        }
        received += static_cast<std::size_t>(count);
        if (received == kBlockBytes) {
            process_block();
            received = 0;
        }
    }
}
