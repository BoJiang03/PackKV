from transformers import AutoTokenizer

from evaluation.evaluation import accuracy_evaluation_with_model
from models.llama_for_kv_cache_unorder import UnorderCacheStatic
from models.llama_for_kv_cache_unorder import LlamaForCausalLM

UnorderCacheStatic.enable_unorder_cache = False

model_name = "meta-llama/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

print(accuracy_evaluation_with_model(
    model=model,
    tokenizer=tokenizer,
    benchmark="coqa"
))

UnorderCacheStatic.enable_unorder_cache = True

print(accuracy_evaluation_with_model(
    model=model,
    tokenizer=tokenizer,
    benchmark="coqa"
))