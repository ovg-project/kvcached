for max_rps in 13 14 15 17 18 19; do
    for completion_len in 256 400; do 
        ./run_benchmark.sh $max_rps $completion_len
    done
done