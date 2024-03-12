```bash
conda create -n vllm python=3.11
conda activate vllm

# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#installing-previous-cuda-releases
conda install cuda -c nvidia/label/cuda-12.1.1 
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia

conda install ninja
pip install packaging
python setup.py develop # 설치 후 c/c++/cu 파일 변경을 적용하려면 `python setup.py build_ext --inplace`
python -m cupyx.tools.install_library --library nccl --cuda 12.x # ray 가 실행되기 위해서 nccl 을 찾는다
pip uninstall outlines
pip install outlines
pip install -U pytest

# .cu 빌드 예시
# /home/sehee/anaconda3/envs/vllm/bin/nvcc  -I/home/sehee/anaconda3/envs/vllm/lib/python3.11/site-packages/torch/include -I/home/sehee/anaconda3/envs/vllm/lib/python3.11/site-packages/torch/include/torch/csrc/api/include -I/home/sehee/anaconda3/envs/vllm/lib/python3.11/site-packages/torch/include/TH -I/home/sehee/anaconda3/envs/vllm/lib/python3.11/site-packages/torch/include/THC -I/home/sehee/anaconda3/envs/vllm/include -I/home/sehee/anaconda3/envs/vllm/include/python3.11 -c -c /home/sehee/workspace/vllm/csrc/quantization/gguf/ggml_cuda_kernel.cu -o /home/sehee/workspace/vllm/build/temp.linux-x86_64-cpython-311/csrc/quantization/gguf/ggml_cuda_kernel.o --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -gencode arch=compute_86,code=sm_86 --threads 8 -DENABLE_FP8_E5M2 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=_C -D_GLIBCXX_USE_CXX11_ABI=0


python gguf_scripts/llm_engine_example.py --model ~/workspace/hf_models/miqu-from-gguf \
--dtype float16 --tensor-parallel-size 2 --max-num-seqs 1 --enforce-eager

python gguf_scripts/llm_engine_example.py \
--model ~/workspace/hf_models/TheBloke_Mixtral-8x7B-Instruct-v0.1-GPTQ_gptq-8bit-128g-actorder_True \
--dtype float16 --tensor-parallel-size 2 --max-num-seqs 1 --enforce-eager

python -m vllm.entrypoints.openai.api_server \
--model ~/workspace/hf_models/TheBloke_Mixtral-8x7B-Instruct-v0.1-GPTQ_gptq-8bit-128g-actorder_True \
--served-model-name mixtral --dtype float16 --tensor-parallel-size 2

python -m vllm.entrypoints.openai.api_server \
--model ~/workspace/hf_models/miqu-from-gguf \
--served-model-name miqu --dtype float16 --tensor-parallel-size 2 
# --enforce-eager

curl http://192.168.51.3:8000/v1/chat/completions -H "Content-Type: application/json" -d '{ "model": "mixtral", "messages": [{"role": "user", "content": "What is a large language model?"}], "temperature": 0.0, "stream": false, "max_tokens": 256}'
curl http://192.168.51.3:8000/v1/chat/completions -H "Content-Type: application/json" -d '{ "model": "miqu", "messages": [{"role": "user", "content": "What is a large language model?"}], "temperature": 0.0, "stream": false, "max_tokens": 256}'

```

### TODO
// ColumnLinear, RowLiner
// QKV/Attention 이 tensor parallel 에 대해서 어떻게 동작하지?
// 왜 throughput 이 낮을까?
// enforce_eager==False 인 경우 cuda graph 는 ([bs, 1] for bs in batch_size_capture_list) 의 input shape 에 대해서 각각 생성된다 (ModelRunner.capture_model), 즉 gguf_gemm 에서 bs 에 대한 분기가 있더라도 괜찮음
// 왜 응답의 퀄리티가 떨어지는 것 처럼 보일까? get_rope 에서 neox_style 을 off 해야한다 (그런데 이게 뭘까?)
// gguf -> safetensors 로 다시 컨버팅 해야한다 (float32 + q5_k/q6_k)
- float16 으로 하면 F32 weight 들이 float16 으로 변환되나?
- rope 가 뭔지 좀 보자, neox style 은 또 뭘까?
