```bash
conda create -n vllm python=3.11
conda activate vllm

# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#installing-previous-cuda-releases
conda install cuda -c nvidia/label/cuda-12.1.1 
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia

conda install ninja
pip install packaging
python setup.py develop # 설치 후 c/c++/cu 파일 변경을 적용하려면 `python setup.py build_ext --inplace`

# .cu 빌드 예시
# /home/sehee/anaconda3/envs/vllm/bin/nvcc  -I/home/sehee/anaconda3/envs/vllm/lib/python3.11/site-packages/torch/include -I/home/sehee/anaconda3/envs/vllm/lib/python3.11/site-packages/torch/include/torch/csrc/api/include -I/home/sehee/anaconda3/envs/vllm/lib/python3.11/site-packages/torch/include/TH -I/home/sehee/anaconda3/envs/vllm/lib/python3.11/site-packages/torch/include/THC -I/home/sehee/anaconda3/envs/vllm/include -I/home/sehee/anaconda3/envs/vllm/include/python3.11 -c -c /home/sehee/workspace/vllm/csrc/quantization/gguf/ggml_cuda_kernel.cu -o /home/sehee/workspace/vllm/build/temp.linux-x86_64-cpython-311/csrc/quantization/gguf/ggml_cuda_kernel.o --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -gencode arch=compute_86,code=sm_86 --threads 8 -DENABLE_FP8_E5M2 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=_C -D_GLIBCXX_USE_CXX11_ABI=0
```