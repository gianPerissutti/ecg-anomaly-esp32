#ifndef ECG_NEURAL_NETWORK_H
#define ECG_NEURAL_NETWORK_H

namespace ecg {

inline constexpr int kWindowSamples = 305;

// Prepare the embedded model for inference from the application task.
[[nodiscard]] bool initialize_model();

// Consume kWindowSamples in millivolts and return the mean absolute error in model units.
// Nonfinite samples and inference failures return NaN.
[[nodiscard]] float reconstruction_error(const float* samples);

}  // namespace ecg

#endif  // ECG_NEURAL_NETWORK_H
