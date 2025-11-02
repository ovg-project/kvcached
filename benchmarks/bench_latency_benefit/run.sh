# Configuration
# Adjust num_prompts as needed (leave empty to use default calculation)
num_prompts=2000
prompt_len=4096  # Default prompt length

for max_rps in 1; do
    for completion_len in 5; do
# for max_rps in 1 2 3 4 5 6 7 8 9 10 15 20 25 30 40; do
#     for completion_len in 64 128 256; do
        ./run_benchmark_fixed_rate.sh $max_rps $completion_len $num_prompts "" "" "" $prompt_len
    done
done