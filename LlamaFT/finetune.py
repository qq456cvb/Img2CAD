from pathlib import Path
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from PIL import Image
import torch
from trl import SFTTrainer, SFTConfig
from qwen_vl_utils import process_vision_info
from peft import LoraConfig
from huggingface_hub import login
import argparse
import os


def format_data(sample):
    image = Image.open(sample["image_path"]).convert("RGB")
    if image is None:
        print(sample["image_path"])
    return {"messages": [
    
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },{
                            "type": "image",
                            "image": image,
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": sample["response"]}],
                },
            ],
        }

# Define a custom collator function
def collate_fn(examples):
    texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]

    image_inputs = [process_vision_info(example["messages"])[0] for example in examples]

    batch = processor(text=texts, images=image_inputs, return_tensors="pt",
                    add_special_tokens=False, padding=True)  # apply chat template already adds special tokens
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Mask image tokens in the labels
    image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100
    batch["labels"] = labels

    return batch
    

def generate_description(sample_image, model, processor):
    messages = [
        {"role": "user", "content": [
            {"type": "text", 
            "text": prompt,
            },
            {"type": "image", 
            "image": sample_image,
            }, 
        ]},
    ]
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        add_special_tokens=False,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1024, top_p=1.0, temperature=1.0, do_sample=False)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', type=str, default='table')
    parser.add_argument('--nepochs', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='data/ckpts/llamaft')
    args = parser.parse_args()
    
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    category = args.category
    nepochs = args.nepochs
    prompt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompt.txt')
    prompt = open(prompt_path).read()

    train_ids = open('{}/data/splits/{}_train_ids.txt'.format(project_dir, category)).read().splitlines()
    test_ids = open('{}/data/splits/{}_test_ids.txt'.format(project_dir, category)).read().splitlines()
    
    valid_ids = [fn.stem for fn in Path('{}/data/llamaft_gt_labels/{}'.format(project_dir, category)).glob('*.txt')]
    train_ids = list(set(train_ids) & set(valid_ids))[:]
    test_ids = list(set(test_ids) & set(valid_ids))[:]


    image_paths = {
        'train': [f'{project_dir}/data/blender_renderings/{fn}.png' for fn in train_ids],
        'test': [f'{project_dir}/data/blender_renderings/{fn}.png' for fn in test_ids],
    }

    # Read prompt once and reuse
    prompt_text = open(prompt_path).read()
    text_inputs = {
        'train': [prompt_text] * len(image_paths['train']),
        'test': [prompt_text] * len(image_paths['test']),
    }

    text_outputs = {
        'train': [open(os.path.join(project_dir, 'data', 'llamaft_gt_labels', category, '{}.txt'.format(fn))).read() for fn in train_ids],
        'test': [open(os.path.join(project_dir, 'data', 'llamaft_gt_labels', category, '{}.txt'.format(fn))).read() for fn in test_ids],
    }

    train_dataset = []
    for i in range(len(image_paths['train'])):
        sample = {
            "image_path": image_paths['train'][i],
            "input": text_inputs['train'][i],
            "response": text_outputs['train'][i],
        }
        train_dataset.append(format_data(sample))
        
    test_dataset = []
    for i in range(len(image_paths['test'])):
        sample = {
            "image_path": image_paths['test'][i],
            "input": text_inputs['test'][i],
            "response": text_outputs['test'][i],
        }
        test_dataset.append(format_data(sample))
    
    # Note: Set your Hugging Face token as environment variable HF_TOKEN
    # or call login() with your token
    # login('your_hf_token_here')

    # Hugging Face model id
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16

    )

    # Load model and tokenizer
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    processor = AutoProcessor.from_pretrained(model_id)

    # LoRA configuration based on QLoRA paper
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM"
    )
    
    output_dir = '{}/{}'.format(args.output_dir, category)
    os.makedirs(output_dir, exist_ok=True)

    # Training arguments
    args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=nepochs,
        per_device_train_batch_size=2,  # Reduced batch size for stability
        gradient_accumulation_steps=1,  # Increased to maintain effective batch size
        gradient_checkpointing=True,  # Disabled to prevent backward graph issues
        optim="adamw_torch_fused",
        logging_steps=5,
        save_strategy="epoch",
        learning_rate=2e-4,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        push_to_hub=False,
        report_to="tensorboard",
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True}
    )
    
    # Trainer setup
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collate_fn,
        tokenizer=processor.tokenizer,
        peft_config=peft_config
    )
    
    # Start training with error handling
    trainer.train()