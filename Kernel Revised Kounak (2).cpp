#include <iostream>
#include <cmath>

extern "C" {

struct IsolationResult {
    int status_code;         // 200 = NOMINAL, 403 = QUARANTINE
    double drift_error_px;   // Nano-drift vector magnitude
    double latency_ms;       // Process duration
    int fault_type;          // 0=NONE,1=SNR_NOISE,2=DRIFT_LIMIT,3=LATENCY
};

IsolationResult CXX_Evaluate_Signal(
    double pred_x, double pred_y,
    double true_x, double true_y,
    double snr_value, double exec_time_ms,
    double max_drift_tol, double min_snr,
    double max_latency)
{
    IsolationResult res{};

    res.latency_ms = exec_time_ms;

    if (!std::isfinite(pred_x) || !std::isfinite(pred_y) ||
        !std::isfinite(true_x) || !std::isfinite(true_y) ||
        !std::isfinite(snr_value) || !std::isfinite(exec_time_ms))
    {
        res.status_code = 403;
        res.fault_type = 3;
        return res;
    }

    double dx = pred_x - true_x;
    double dy = pred_y - true_y;

    res.drift_error_px = std::hypot(dx, dy);

    if (snr_value < min_snr) {
        res.status_code = 403;
        res.fault_type = 1;
    }
    else if (res.drift_error_px > max_drift_tol) {
        res.status_code = 403;
        res.fault_type = 2;
    }
    else if (exec_time_ms > max_latency) {
        res.status_code = 403;
        res.fault_type = 3;
    }
    else {
        res.status_code = 200;
        res.fault_type = 0;
    }

    return res;
}

}