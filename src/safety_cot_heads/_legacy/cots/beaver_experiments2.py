import pandas as pd
from transformers import AutoTokenizer, LlamaForCausalLM
import json

model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", device_map = "cuda:0")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

cat_list = [#'animal_abuse',
            #'child_abuse',
            #'controversial_topics,politics',
            #'discrimination,stereotype,injustice',
            #'drug_abuse,weapons,banned_substance',
            #'financial_crime,property_crime,theft',
            #'hate_speech,offensive_language',
            #'misinformation_regarding_ethics,laws_and_safety',
            #'non_violent_unethical_behavior',
            'privacy_violation',
            'self_harm',
            'sexually_explicit,adult_content',
            'terrorism,organized_crime',
            'violence,aiding_and_abetting,incitement']
for cat in cat_list:
  df = pd.read_csv(f"exp_data/Beaver_samples/{cat}.csv")
  with open(f"exp_data/beaver_results_llama/{cat}_outputs.json", "w+") as fp:
    for _,i in df.iterrows():
      inputs = tokenizer([i["prompt"]], truncation=True, padding = True, return_tensors="pt").to("cuda:0")

      generate_ids = model.generate(inputs.input_ids, max_new_tokens =256, pad_token_id = tokenizer.pad_token_id)
      output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

      #with open(f"exp_data/beaver_results/{cat}_outputs.json", "w") as fp:
      fp.write(json.dumps(output) + '\n')

