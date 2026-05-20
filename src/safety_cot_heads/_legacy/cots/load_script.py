import torch
from transformers import BitsAndBytesConfig
from peft import PeftConfig, LoraConfig, get_peft_model, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
def load_model(task_pretrained_name='meta-llama/Meta-Llama-3-8B',
               lora_alpha=8,
               cache_dir='/home/vlh769/.cache/huggingface/transformers/',
               lora_r=64,
               finetuned_path=None,
               quantize=True,
               peft_task="CAUSAL_LM",
               problem_type=None,
               num_labels =None):
    if finetuned_path is None:
        print(f"\nLoading non-finetuned model: {task_pretrained_name}...")
    elif finetuned_path is not None:
        print(f"\nLoading fine-tuned model: {finetuned_path}...")
    compute_dtype = getattr(torch, "float16")
    if finetuned_path:
        peft_params = PeftConfig.from_pretrained(finetuned_path)
    else:
        # peft_params = LoraConfig(
        #     # lora_alpha=lora_alpha,
        #     # lora_dropout=0.05,
        #     target_modules="all-linear", # qlora
        #     # r=lora_r,
        #     bias="none",
        #     task_type="CAUSAL_LM",
        # )

        peft_params = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            r=lora_r,
            bias="none",
            task_type=peft_task,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', ]
                            # 'gate_proj', 'up_proj', 'down_proj'],
        )
    if quantize:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )
    else:
        quant_config = None
    if peft_task == "SEQ_CLS":
        model = AutoModelForSequenceClassification.from_pretrained(
            task_pretrained_name,
            quantization_config=quant_config,
            device_map='cuda:0', cache_dir=cache_dir,
            problem_type=problem_type,
            num_labels=num_labels
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
                task_pretrained_name,
                quantization_config=quant_config,
                device_map='cuda:0', cache_dir=cache_dir,
                problem_type=problem_type
            )

    if finetuned_path is not None:
        print('loading model from:', finetuned_path)
        model = PeftModel.from_pretrained(model,
                                          finetuned_path,
                                          # is_trainable=True,
                                          )
        model.to('cuda:0')
    else:
        model = get_peft_model(model, peft_params)
        print(model.print_trainable_parameters())

    model.config.use_cache = False
    model.config.pretraining_tp = 1
    tokenizer = AutoTokenizer.from_pretrained(task_pretrained_name, cache_dir=cache_dir)
    return model, tokenizer