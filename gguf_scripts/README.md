### Run miqu gguf in tensor parallelism
llama.cpp 로 miqu 를 돌리면 pipeline parallel 만 지원해서 2-GPU 파워를 충분히 쓰지 못한다.
throughput 을 최대한 끌어올리기 위해, vllm 에 gguf plugin 을 커스텀 구현해서 miqu tensor parallel 을 가능하도록 하자.
현재는 Q5_K, Q6_K 만 지원한다.
cuda kernel 은 llama.cpp 의 kernel 코드를 거의 그대로 가져와 구현한다.

### 빌드 및 설치
```bash
conda create -n vllm python=3.11
conda activate vllm

# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#installing-previous-cuda-releases
conda install cuda -c nvidia/label/cuda-12.1.1 
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia

conda install ninja
pip install packaging
python setup.py develop
python -m cupyx.tools.install_library --library nccl --cuda 12.x # ray 가 실행되기 위해서 nccl 을 찾는다
pip install -U outlines
pip install flash-attn --no-build-isolation # flash_attn 을 찾지 못한다고 할때
```

```bash
# 설치 후 c/c++/cu 파일 변경을 간단하게 적용하려면 
python setup.py build_ext --inplace
# rebuild (ex. upstream/main 과 merge 후)
pip uninstall vllm; \
rm -rf .pytest_cache; rm -rf build; rm -rf vllm.egg-info; rm vllm/*.so; \
python setup.py develop
```

```bash
# 필요한 경우 ggml_cuda_kernel.cu 커스텀 빌드
~/anaconda3/envs/vllm/bin/nvcc  -I/home/sehee/anaconda3/envs/vllm/lib/python3.11/site-packages/torch/include -I/home/sehee/anaconda3/envs/vllm/lib/python3.11/site-packages/torch/include/torch/csrc/api/include -I/home/sehee/anaconda3/envs/vllm/lib/python3.11/site-packages/torch/include/TH -I/home/sehee/anaconda3/envs/vllm/lib/python3.11/site-packages/torch/include/THC -I/home/sehee/anaconda3/envs/vllm/include -I/home/sehee/anaconda3/envs/vllm/include/python3.11 -c -c /home/sehee/workspace/vllm/csrc/quantization/gguf/ggml_cuda_kernel.cu -o /home/sehee/workspace/vllm/build/temp.linux-x86_64-cpython-311/csrc/quantization/gguf/ggml_cuda_kernel.o --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -gencode arch=compute_86,code=sm_86 --threads 8 -DENABLE_FP8_E5M2 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=_C -D_GLIBCXX_USE_CXX11_ABI=0
```

### upstream(https://github.com/vllm-project/vllm.git) 반영하기
```bash
git remote add upstream https://github.com/vllm-project/vllm.git
# 원본 저장소의 변경 사항을 로컬 시스템으로 가져옵니다
git fetch upstream
# gguf 브랜치로 checkout 하고, upstream/main 을 merge
git checkout gguf ; git merge upstream/main
# merge conflict 가 발생한 경우 Resolve using yours
git checkout upstream/main -- path/to/your/file
# 최종 commit / push
git push origin gguf
```

### example 테스트
```bash
python gguf_scripts/llm_engine_example.py --model ~/workspace/hf_models/miqu-from-gguf \
--dtype float16 --tensor-parallel-size 2 --max-num-seqs 1 --enforce-eager

python gguf_scripts/llm_engine_example.py \
--model ~/workspace/hf_models/TheBloke_Mixtral-8x7B-Instruct-v0.1-GPTQ_gptq-8bit-128g-actorder_True \
--dtype float16 --tensor-parallel-size 2 --max-num-seqs 1 --enforce-eager
```

### OAI 서버 실행
```bash
~/anaconda3/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
--port 35865 --model ~/workspace/hf_models/miqu-from-gguf \
--served-model-name miqu --tensor-parallel-size 2 --max-num-seqs 1
# --enforce-eager

curl http://192.168.51.3:35865/v1/chat/completions -H "Content-Type: application/json" -d '{ "model": "miqu", "messages": [{"role": "user", "content": "What is a large language model?"}], "temperature": 0.0, "stream": false, "max_tokens": 256}'
```

### TODO
...