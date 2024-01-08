import warnings
warnings.filterwarnings("ignore")

import llama_self_extend_patch as LlamaSE
from modify_utils import modify_method_of_instance
from functools import partial
import json
from transformers.models.phi.modeling_phi import PhiAttention
from transformers import AutoTokenizer, AutoModelForCausalLM

original_phi_forward = PhiAttention.forward
self_extend_forward = partial(LlamaSE.self_extend_forward_phi, group_size_1=8, group_size_2=1024)


model_path = 'susnato/phi-2'
cache_dir = '/mnt/c/Users/amangokrani/OneDrive - Microsoft/Personal/Huggingface/models'
# model_path = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

modify_method_of_instance(model, "PhiAttention", "forward", self_extend_forward)

inputs = tokenizer('''def print_prime(n):
    """ Print all primes between 1 and n"""''', 
    return_tensors="pt", 
    return_attention_mask=False
)

outputs = model.generate(**inputs, max_length=35)