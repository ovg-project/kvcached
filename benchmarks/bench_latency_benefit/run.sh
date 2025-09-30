# for max_rps in 5; do
#     for completion_len in 64; do
for max_rps in 40; do
    for completion_len in 64 128 256 512 1024; do
        for seq_len in 512 1024; do
            ./run_benchmark_fixed_rate.sh $max_rps $completion_len $seq_len
        done
    done
done