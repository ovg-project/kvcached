for max_rps in 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 64 68 72 76 80; do
    for completion_len in 128 256 400 512 1024; do 
        ./run_benchmark.sh $max_rps $completion_len
    done
done