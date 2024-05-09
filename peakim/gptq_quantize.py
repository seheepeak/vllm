# Meta-Llama-3-70B-Instruct 를 GPTQ 8-bit 로 변환
# device_map="cpu" 로 해야, cuda out of memory 가 발생하지 않음 (?)

from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_id = str(Path.home() / "workspace/hf_models/Meta-Llama-3-70B-Instruct")
out_dir = str(Path.home() / "workspace/hf_models/Meta-Llama-3-70B-Instruct-GPTQ-8Bit")

tokenizer = AutoTokenizer.from_pretrained(model_id)
gptq_config = GPTQConfig(
    bits=8,
    dataset="wikitext2",
    tokenizer=tokenizer,
)

quantized_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu",  # device_map="auto"
    quantization_config=gptq_config,
)

quantized_model.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)
