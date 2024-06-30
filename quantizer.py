import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound
from huggingface_hub import HfApi
import os

model_name = "VAGOsolutions/Llama-3-SauerkrautLM-8b-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


bits, group_size, sym = 8, 128, False
##device:Optional["auto", None, "hpu", "cpu", "cuda"]
autoround = AutoRound(
    model, tokenizer, bits=bits, group_size=group_size, sym=sym, device="auto"
)
autoround.quantize()
output_dir = "./tmp_autoround"
autoround.save_quantized(
    output_dir, format="auto_gptq"
)  ##save_quantized(output_dir,format=="auto_round")

repo_id = (
    f'pL-Community/{model_name.split("/")[-1]}_w{bits}_g{group_size}_autoround_gptq'
)

hfapi = HfApi()
hfapi.create_repo(repo_id, private=True)
hfapi.upload_folder(
    repo_id=repo_id,
    folder_path=output_dir,
    commit_message="quantized model",
    repo_type="model",
)

os.removedirs(output_dir)
