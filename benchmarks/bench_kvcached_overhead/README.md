# kvcached overhead benchmark

The following results are collected on GCP-a100-80G GPU.

The configuration can be found in the server/client start scripts.

## SGLang results

### Request rate = 32

<table>
<tr><th>with kvcached</th><th>without kvcached</th></tr>
<tr>
<td>

```
============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    32.0
Max request concurrency:                not set
Successful requests:                     1920
Benchmark duration (s):                  213.71
Total input tokens:                      1843657
Total generated tokens:                  157031
Total generated tokens (retokenized):    156230
Request throughput (req/s):              8.98
Input token throughput (tok/s):          8626.75
Output token throughput (tok/s):         734.77
Total token throughput (tok/s):          9361.52
Concurrency:                             1418.05
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   157842.85
Median E2E Latency (ms):                 157290.17
---------------Time to First Token----------------
Mean TTFT (ms):                          112043.40
Median TTFT (ms):                        140974.91
P99 TTFT (ms):                           145546.45
---------------Inter-Token Latency----------------
Mean ITL (ms):                           569.94
Median ITL (ms):                         172.52
P95 ITL (ms):                            561.59
P99 ITL (ms):                            798.81
Max ITL (ms):                            151008.49
==================================================
```

</td>
<td>

```
============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    32.0
Max request concurrency:                not set
Successful requests:                     1920
Benchmark duration (s):                  71.62
Total input tokens:                      1843657
Total generated tokens:                  157031
Total generated tokens (retokenized):    156236
Request throughput (req/s):              26.81
Input token throughput (tok/s):          25741.91
Output token throughput (tok/s):         2192.53
Total token throughput (tok/s):          27934.44
Concurrency:                             514.76
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   19201.86
Median E2E Latency (ms):                 18354.78
---------------Time to First Token----------------
Mean TTFT (ms):                          2187.97
Median TTFT (ms):                        2308.73
P99 TTFT (ms):                           5356.42
---------------Inter-Token Latency----------------
Mean ITL (ms):                           216.53
Median ITL (ms):                         143.18
P95 ITL (ms):                            658.38
P99 ITL (ms):                            1253.05
Max ITL (ms):                            3768.23
==================================================
```

</td>
</tr>
</table>

### Request rate = 16

<table>
<tr><th>with kvcached</th><th>without kvcached</th></tr>
<tr>
<td>

```
============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    16.0
Max request concurrency:                not set
Successful requests:                     960
Benchmark duration (s):                  64.67
Total input tokens:                      931890
Total generated tokens:                  76078
Total generated tokens (retokenized):    75597
Request throughput (req/s):              14.84
Input token throughput (tok/s):          14409.78
Output token throughput (tok/s):         1176.39
Total token throughput (tok/s):          15586.17
Concurrency:                             21.04
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   1417.55
Median E2E Latency (ms):                 1330.63
---------------Time to First Token----------------
Mean TTFT (ms):                          87.27
Median TTFT (ms):                        76.26
P99 TTFT (ms):                           341.90
---------------Inter-Token Latency----------------
Mean ITL (ms):                           17.10
Median ITL (ms):                         7.90
P95 ITL (ms):                            55.93
P99 ITL (ms):                            146.19
Max ITL (ms):                            895.50
==================================================
```

</td>
<td>

```
============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    16.0
Max request concurrency:                not set
Successful requests:                     960
Benchmark duration (s):                  64.54
Total input tokens:                      931890
Total generated tokens:                  76078
Total generated tokens (retokenized):    75616
Request throughput (req/s):              14.87
Input token throughput (tok/s):          14439.01
Output token throughput (tok/s):         1178.78
Total token throughput (tok/s):          15617.79
Concurrency:                             20.95
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   1408.25
Median E2E Latency (ms):                 1316.92
---------------Time to First Token----------------
Mean TTFT (ms):                          91.67
Median TTFT (ms):                        75.73
P99 TTFT (ms):                           416.68
---------------Inter-Token Latency----------------
Mean ITL (ms):                           16.93
Median ITL (ms):                         8.03
P95 ITL (ms):                            51.85
P99 ITL (ms):                            152.85
Max ITL (ms):                            954.80
==================================================
```

</td>
</tr>
</table>

### Request rate = 8

<table>
<tr><th>with kvcached</th><th>without kvcached</th></tr>
<tr>
<td>

```
============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    8.0
Max request concurrency:                not set
Successful requests:                     480
Benchmark duration (s):                  59.53
Total input tokens:                      468878
Total generated tokens:                  39948
Total generated tokens (retokenized):    39812
Request throughput (req/s):              8.06
Input token throughput (tok/s):          7875.69
Output token throughput (tok/s):         671.00
Total token throughput (tok/s):          8546.69
Concurrency:                             7.12
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   883.28
Median E2E Latency (ms):                 880.06
---------------Time to First Token----------------
Mean TTFT (ms):                          68.61
Median TTFT (ms):                        65.09
P99 TTFT (ms):                           168.36
---------------Inter-Token Latency----------------
Mean ITL (ms):                           9.96
Median ITL (ms):                         7.10
P95 ITL (ms):                            29.62
P99 ITL (ms):                            56.46
Max ITL (ms):                            201.50
==================================================
```

</td>
<td>

```
============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    8.0
Max request concurrency:                not set
Successful requests:                     480
Benchmark duration (s):                  59.48
Total input tokens:                      468878
Total generated tokens:                  39948
Total generated tokens (retokenized):    39807
Request throughput (req/s):              8.07
Input token throughput (tok/s):          7882.70
Output token throughput (tok/s):         671.60
Total token throughput (tok/s):          8554.30
Concurrency:                             6.97
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   864.32
Median E2E Latency (ms):                 864.35
---------------Time to First Token----------------
Mean TTFT (ms):                          65.48
Median TTFT (ms):                        61.78
P99 TTFT (ms):                           156.96
---------------Inter-Token Latency----------------
Mean ITL (ms):                           9.76
Median ITL (ms):                         7.08
P95 ITL (ms):                            27.52
P99 ITL (ms):                            53.50
Max ITL (ms):                            207.75
==================================================
```

</td>
</tr>
</table>
