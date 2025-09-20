# for max_rps in 5; do
#     for completion_len in 64; do
for max_rps in 1 2 3 4 5 6 7 8 9 10 15 20 25 30 40; do
    for completion_len in 64 128 256; do
        ./run_benchmark_fixed_rate.sh $max_rps $completion_len
    done
done