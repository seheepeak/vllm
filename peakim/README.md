### 빌드 및 설치
```bash
conda create -n vllm python=3.11
conda activate vllm

# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#installing-previous-cuda-releases
conda install cuda -c nvidia/label/cuda-12.1.1 
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia

git clone ~~~~~~; cd ~~~;
pip install -e .
pip install flash-attn --no-build-isolation # flash_attn 을 찾지 못한다고 할때
```

### upstream(https://github.com/vllm-project/vllm.git) 반영하기
```bash
git remote add upstream https://github.com/vllm-project/vllm.git
# 원본 저장소의 변경 사항을 로컬 시스템으로 가져옵니다
git fetch upstream
# peakim 브랜치로 checkout 하고, upstream/main 을 merge
git checkout peakim ; git merge upstream/main
# merge conflict 가 발생한 경우 Resolve using yours
git checkout upstream/main -- path/to/your/file
# 최종 commit / push
git push origin peakim
```

### OAI 서버 실행
```bash
~/anaconda3/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
--port 35865 --model ~/workspace/hf_models/c4ai-command-r-plus-gptq \
--served-model-name cmdr --tensor-parallel-size 2 --max-num-seqs 1 \
--gpu-memory-utilization 0.95 --max-model-len 32768 --enforce-eager

# system + user
curl http://192.168.51.3:35865/v1/chat/completions -H "Content-Type: application/json" -d '{ "model": "cmdr", "messages": [{"role": "system", "content": "You are Command-R, a brilliant, sophisticated, AI-assistant trained to assist human users by providing thorough responses. You are trained by Cohere."},{"role": "user", "content": "What is a large language model?"}], "temperature": 0.0, "stream": false, "max_tokens": 256}'

# user + assistant (prefill)
curl http://192.168.51.3:35865/v1/chat/completions -H "Content-Type: application/json" -d '{ "model": "cmdr", "messages": [{"role": "user", "content": "What is a large language model?"},{"role": "assistant", "content": "The modern neural"}], "temperature": 0.0, "stream": false, "max_tokens": 256}'
```

### TODO
...