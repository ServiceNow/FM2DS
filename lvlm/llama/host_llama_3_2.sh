CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --model meta-llama/Llama-3.2-90B-Vision-Instruct \
    --tensor-parallel-size "4" \
    --dtype bfloat16 \
    --served-model-name meta-llama/Meta-Llama-3.2-90B-Vision-Instruct \
    --gpu-memory-utilization "0.95" \
    --port "9095" \
    --enforce-eager \
    --max-num-seqs "16" \
    --max-model-len "8192"