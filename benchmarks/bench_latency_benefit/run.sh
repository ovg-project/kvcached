for max_rps in 13 14 15 16 17 18 19 20; do
    for completion_len in 256 400 1024; do
# for max_rps in 12; do
#     for completion_len in 256; do
        ./run_benchmark_fixed_rate.sh $max_rps $completion_len
    done
done