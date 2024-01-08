import re
import warnings
warnings.filterwarnings("ignore")

import llama_self_extend_patch as LlamaSE
from modify_utils import modify_method_of_instance
from functools import partial
import json
from transformers.models.phi.modeling_phi import PhiAttention
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from datasets import load_dataset
from string import Template

original_phi_forward = PhiAttention.forward
self_extend_forward = partial(LlamaSE.self_extend_forward_phi, group_size_1=8, group_size_2=1024)


model_path = 'susnato/phi-2'
cache_dir = '/workspace/models'
dataset_path = '/workspace/git/eval-note-generation/outputs/train.gpt-4-1106-preview.pred.fullnote.json'
# model_path = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=cache_dir, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

modify_method_of_instance(model, "PhiAttention", "forward", self_extend_forward)

#inputs = tokenizer('''def print_prime(n):
#    """ Print all primes between 1 and n"""''', 
#    return_tensors="pt", 
#    return_attention_mask=False
#).to("cuda")

inputs = tokenizer('''
Alice: I don't know why, I'm struggling to maintain focus while studying. Any suggestions?
Bob: Well, have you tried creating a study schedule and sticking to it?
Alice: Yes, I have, but it doesn't seem to help much.
Bob: Hmm, maybe you should try studying in a quiet environment, like the library.
Alice: ...
''', return_tensors="pt", return_attention_mask=False).to("cuda")
streamer = TextStreamer(tokenizer)
outputs = model.generate(**inputs, streamer=streamer, max_length=3096)

#dataset = load_dataset('json', data_files=dataset_path)
#dataset = dataset['train']

#PROMPT = Template("""
 #       Instruct: summarize the conversation to generate a clinical note with four sections: HISTORY OF PRESENT ILLNESS, PHYSICAL EXAM, RESULTS, ASSESSMENT AND PLAN.

  #      The conversation is:
   #     $conversation

    #    Output:
    #    """)

#example = dataset[0]
#conversation = example['src'].replace('[doctor]', 'doctor:').replace('[patient]', 'patient:')
#conversation = re.sub(r'\s+([,.?])', r'\1', conversation)
#inputs = tokenizer(PROMPT.substitute(conversation=conversation), return_tensors="pt", return_attention_mask=False).to("cuda")

#outputs = model.generate(**inputs, streamer=streamer, max_length=3096)

import pdb;pdb.set_trace()
