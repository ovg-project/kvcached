import numpy as np
import matplotlib.pyplot as plt

def total_rps_curve(peak: int, num_instances: int = 6):
    start_rps = 0
    end_rps = 1
    inc = 1

    # Matches get_ramp_up_down_requests() logic
    ramp_up_duration = max(1, (peak - start_rps) // inc)      # = peak
    ramp_down_duration = max(1, (peak - end_rps) // inc)      # = peak-1
    rps_up = [min(start_rps + (s + 1) * inc, peak) for s in range(ramp_up_duration)]   # 1..peak
    rps_down = [max(peak - (s + 1) * inc, end_rps) for s in range(ramp_down_duration)] # peak-1..1
    schedule = np.array(rps_up + rps_down, dtype=float)       # length = 2*peak-1, sum = peak^2

    # Matches your bash MODEL_DELAY computation
    model_delay = -(ramp_up_duration // 4) + (ramp_up_duration * 2)

    total_time = model_delay * (num_instances - 1) + len(schedule)
    t = np.arange(total_time, dtype=int)
    total = np.zeros(total_time, dtype=float)

    for i in range(num_instances):
        s = i * model_delay
        total[s:s+len(schedule)] += schedule

    return t, total, schedule, model_delay

# Representative peaks
for peak in [10, 15, 20]:
    t, total, schedule, model_delay = total_rps_curve(peak, num_instances=6)
    plt.figure()
    plt.plot(t, total)

    plt.xlabel("Time (seconds)")
    plt.ylabel("Total offered RPS (sum over 6 instances)")
    plt.title(f"Total offered RPS vs time (peak={peak}, model_delay={model_delay}s, schedule_len={len(schedule)}s)")
    plt.savefig(f"total_rps_curve_peak_{peak}.png")