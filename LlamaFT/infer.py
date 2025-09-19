#!/usr/bin/env python3
"""
Unified inference pipeline for Img2CAD LlamaFT stage.
This script combines image-to-text generation and text-to-h5 conversion
into a single seamless pipeline.

Usage:
    python LlamaFT/infer.py --category chair --split test
"""

import os
import numpy as np
import argparse
import time
import statistics
import re
import ast
import h5py
from copy import deepcopy
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import torch
from enum import Enum
from math import sqrt, cos, sin, tan, radians
from functools import partial
from multiprocessing import cpu_count
from glob import glob

# Llama-specific imports
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from huggingface_hub import login

# Add utils to path for cmd2vec
from utils.cmd_to_vec import cmd2vec


# ============================================================================
# TEXT-TO-H5 CONVERSION UTILITIES (from original infer.py)
# ============================================================================

# Evaluable functions in GPT arguments.
EVAL_FTN = {
    'sin': lambda x: sin(radians(x)), 
    'cos': lambda x: cos(radians(x)), 
    'tan': lambda x: tan(radians(x)), 
    'sqrt': sqrt
}

# Names already reserved for commands, i.e. not evaluable.
CMD_VARS = ['NewBody', 'Join', 'Intersect', 'Cut', 
            'OneSided', 'TwoSided', 'Symmetric']

# Delimiters for comments and commands.
COMMENT_STARTS = ['//', '#', ';', '--']
SOL_STARTS = ('<SOL>', '$<SOL>$')
CUT_STARTS = ('<CUT>', '$<CUT>$')
SKETCH_STARTS = ('L:', 'A:', 'R:')
SKETCH_STARTS_without_semicolon = ('L (', 'A (', 'R (','L(', 'A(', 'R(')
EXTRUDE_STARTS = ('E:')
CMD_STARTS = ('L:', 'A:', 'R:', 'E:')

class CMD_TYPE(Enum):
    SOL = 0
    CUT = 1
    EXTRUDE = 2
    LINE = 3
    ARC = 4
    CIRCLE = 5
    
cmd_mapping = {
    "<SOL>": CMD_TYPE.SOL,
    "<CUT>": CMD_TYPE.CUT,
    "E": CMD_TYPE.EXTRUDE,
    "L": CMD_TYPE.LINE,
    "A": CMD_TYPE.ARC,
    "R": CMD_TYPE.CIRCLE,
    "C": CMD_TYPE.CIRCLE,
}


def divide_lines_into_blocks(lines):
    blocks = []
    current_block = []

    for line in lines:
        if line.strip() == "":  # Check if the line is empty
            if current_block:  # If the current block is not empty, add it to blocks
                blocks.append("\n".join(current_block))
                current_block = []  # Start a new block
        else:
            current_block.append(line)  # Add the non-empty line to the current block

    if current_block:  # Add the last block if it's not empty
        blocks.append("\n".join(current_block))

    return blocks


def filter_lines(lines):
    res = []
    for line in lines:
        if line.strip().startswith("```"):
            res.append('')
        else:
            line = line.replace('\\', '')
            if '<SOL>' in line and line.split('<SOL>')[1].strip() != '':
                res.append('<SOL>')
                res.append(line.split('<SOL>')[1].strip())
            else:
                res.append(line)
    return res


def filter_blocks(blocks):
    res = []
    for i, block in enumerate(blocks):
        if '<SOL>' in block:
            if i > 0 and block.startswith('<SOL>') and '<SOL>' not in blocks[i-1]:
                res.append(blocks[i-1] + '\n' + block)
            else:
                res.append(block)
    return res


def parse_cmd_line(line):
    # Remove unwanted characters for robustness
    line = re.sub(r'[<>$]', '', line)
    
    if line.startswith("SOL"):
        return CMD_TYPE.SOL, None
    elif line.startswith("CUT"):
        return CMD_TYPE.CUT, None
    
    # Regex pattern to capture command and arguments
    pattern = re.compile(r'(\w)\s*')
    match = pattern.search(line)
    
    if match:
        cmd_str = match.groups()[0]
        cmd_type = cmd_mapping.get(cmd_str, None)
        if cmd_type is not None:
            return cmd_type, None

    return None, None  # In case of no match


def extract_base_name(text):
    # Find the first occurrence of a number, considering any non-digit characters (excluding spaces) before it
    match = re.search(r'(.*)(\s+)\D*(\d+)', text)
    if match:
        # Extract the text before the last consecutive spaces before the number
        name_part = match.group(1).strip()
    else:
        # If no number is found, use the whole text
        name_part = text.strip()
    # to lowercase
    name_part = name_part.lower()
    return name_part


def remove_parentheses(line):
    result = re.sub(r'\([^()]*\)', '', line)
    return result


def remove_keywords(line):
    # Convert the entire line to lowercase
    line = line.lower()  

    # Remove "sketch" and "extrude" from the list of words
    line = line.replace("sketch and extrusion", "").strip()
    line = line.replace("sketch of", "").strip()
    line = line.replace("sketch for", "").strip()
    line = line.replace("extrusion of", "").strip()
    line = line.replace("start of", "").strip()
    
    # Split the line into a list of words
    words = line.split()
    # Filter out the unwanted words
    filtered_words = [word for word in words if word.lower() not in {"sketching", "sketch", "extrude", "profile", "begin", "-"}]

    # Reassemble the list of words into a string, separated by spaces
    modified_line = ' '.join(filtered_words)

    return modified_line


def extract_comments(block, blacklists=('define', 'model', 'program')):
    comments = []
    res = []
    cand_seps = ['#', '//']
    for line in block:
        # try extract comment after any #, // if any
        sep_locations = []
        seps = []
        for sep in cand_seps:
            if sep in line:
                sep_locations.append(line.index(sep))
                seps.append(sep)
        sep = seps[np.argmin(sep_locations)] if sep_locations else None
        
        if sep is not None:
            comment = line.split(sep, 1)[1].strip().lower()
            if not any(bl in comment for bl in blacklists):
                comments.append(remove_parentheses(remove_keywords(comment)))
                # remove comment from line
                line = line.split(sep, 1)[0].strip('#/')
        if line.strip() != '':
            res.append(line.strip())
    return comments, res


# Extracts all placeholder variables from argument array. 
def get_vars(args):
    vars = set([])
    if not args:
        return vars
    for arg in args:
        try:
            for node in ast.walk(ast.parse(arg)):
                if isinstance(node, ast.Name) and node.id not in EVAL_FTN.keys() | CMD_VARS:
                    vars.add(node.id)
        except SyntaxError:  # sometimes gpt won't give valid variable name
            pass
    return vars


# Evaluates each element in argument array at given placeholder values.
def eval_vars(args, var_vals):
    if not args:
        return args
    for i in range(len(args)):
        if args[i] not in CMD_VARS:
            args[i] = eval(args[i], EVAL_FTN, var_vals)
    return args


def remove_breaks(lines):
    for i in range(len(lines) - 1):
        line = lines[i].strip()
        if line and line[-1] == '\\':
            lines[i] = line[:-1] + lines[i + 1].strip()
    return lines


def get_part_to_cad(lines):
    lines = filter_lines(lines)
    blocks = divide_lines_into_blocks(lines)
    blocks = filter_blocks(blocks)
    blacklists = ('define', 'model', 'program')
    cmd_blocks = {}
    
    for block in blocks:
        comments, block_lines = extract_comments(block.splitlines(), blacklists)
        if len(comments) > 0:
            comment = comments[0]
            cmds = []
            for j, line in enumerate(block_lines):
                cmd_type, _ = parse_cmd_line(line)
                
                # TODO, no cut for now, dummy values
                if cmd_type == CMD_TYPE.SOL:
                    cmds.append([CMD_TYPE.SOL, None])
                    continue
                if cmd_type == CMD_TYPE.EXTRUDE:
                    cmds.append([CMD_TYPE.EXTRUDE, ['0', '1.', '0', '1.', '0', '0', '0', '0', '0', '1.', '0', 'NewBody', 'OneSided']])
                    continue
                if cmd_type == CMD_TYPE.LINE:
                    nargs = 2
                elif cmd_type == CMD_TYPE.ARC:
                    nargs = 4
                elif cmd_type == CMD_TYPE.CIRCLE:
                    nargs = 3
                cmds.append([cmd_type, ['0'] * nargs])
            cmd_blocks[comment] = cmds
        
    # convert my cmd to rahul's cmd
    cmd_convert_map = {
        CMD_TYPE.SOL: '<SOL>',
        CMD_TYPE.EXTRUDE: 'E',
        CMD_TYPE.LINE: 'L',
        CMD_TYPE.ARC: 'A',
        CMD_TYPE.CIRCLE: 'R',
    }
    cmd_blocked_converted = {}
    for label, cmd_block in cmd_blocks.items():
        cmd_blocked_converted[label] = []

        arc_loc = -1
        for cmd in cmd_block:
            cmd_type, args = cmd
            if cmd_type == CMD_TYPE.ARC:
                arc_loc = len(cmd_blocked_converted[label])
            if cmd_type in cmd_convert_map:
                cmd_blocked_converted[label].append([cmd_convert_map[cmd_type], [str(arg) for arg in args] if args is not None else None])
        
        if arc_loc != -1 and arc_loc < len(cmd_blocked_converted[label]) - 2:
            cmd_block_new = deepcopy(cmd_blocked_converted[label])
            cmd_blocked_converted[label] = cmd_block_new[0:1] + cmd_block_new[arc_loc+1:-1] \
                + cmd_block_new[1:arc_loc+1] + cmd_block_new[-1:]
    return cmd_blocked_converted


# Get set of placeholder variable names.
def get_var_names(part_to_cad):
    var_names = set([])
    for cad in part_to_cad.values():
        for cmd in cad:
            var_names |= get_vars(cmd[1])
    return var_names


# Evaluate PART to CAD dictionary given dictionary of variable values.
def set_var_vals(part_to_cad, var_vals):
    for part in part_to_cad.keys():
        cad = part_to_cad[part]
        for i in range(len(cad)):
            cad[i][1] = eval_vars(cad[i][1], var_vals)
        part_to_cad[part] = cad
    return part_to_cad


# Get PART to VEC, once numerical values have been filled in.
# Important: does not evaluate empty CAD parts.
def get_part_to_vec(part_to_cad, use_normal=True):
    part_to_vec = {}
    for part in part_to_cad.keys():
        cad = part_to_cad[part]
        if cad:
            full_vec = np.array([cmd2vec(*cmd, use_normal=use_normal) for cmd in cad])
            part_to_vec[part] = full_vec
    return part_to_vec


def process_text_to_h5(text_response, output_path):
    """Convert text response to h5 format and save."""
    try:
        # Extract parts to CAD, and all variables.
        part_to_cad = get_part_to_cad(text_response.splitlines())
        part_to_vec = get_part_to_vec(part_to_cad)  # Transform CAD program into vector format

        if part_to_vec:  # Only save if we have valid data
            full_vec = np.concatenate([part_to_vec[part] for part in part_to_vec.keys()])
            with h5py.File(output_path, 'w') as fp:
                fp.create_dataset("vec", data=full_vec, dtype=float)
                for part in part_to_vec.keys():
                    fp.create_dataset(part, data=part_to_vec[part], dtype=float)
            return True
        else:
            return False
            
    except Exception as e:
        print(f"Error processing text to h5: {e}")
        return False


# ============================================================================
# IMAGE-TO-TEXT GENERATION (from img2llama.py)
# ============================================================================

def generate_description(prompt, sample_image, model, processor, max_new_tokens=1024, top_p=1.0, temperature=1.0, do_sample=True):
    """Generate CAD description from image using fine-tuned Llama model."""
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
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature, do_sample=do_sample)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


def load_model_and_processor(model_id, adapter_path):
    """Load the Llama model and processor."""
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Load base model with quantization
    print(f"Loading base model: {model_id}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_use_double_quant=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    
    # Load PEFT model with adapter
    print(f"Loading adapter: {adapter_path}")
    model.load_adapter(adapter_path)
    
    model.eval()
    return model, processor


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def process_single_image(args, img_path, img_id, model, processor, prompt, generation_params):
    """Process a single image through the complete pipeline: image -> text -> h5."""
    try:
        # Load and convert image
        sample_image = Image.open(img_path).convert("RGB")
        
        # Generate text response
        start_time = time.time()
        text_response = generate_description(
            prompt, sample_image, model, processor, 
            max_new_tokens=args.num_tokens, **generation_params
        )
        generation_time = time.time() - start_time
        
        # Convert text to h5
        h5_output_path = os.path.join(args.out_dir, f'{img_id}.h5')
        h5_success = process_text_to_h5(text_response, h5_output_path)
        
        # Optionally save text response for debugging
        if args.save_text:
            text_output_path = os.path.join(args.out_dir, f'{img_id}.txt')
            with open(text_output_path, 'w') as f:
                f.write(text_response)
        
        # Count tokens
        tokens = processor.tokenizer.encode(text_response, add_special_tokens=False)
        token_count = len(tokens)
        
        return {
            'success': h5_success,
            'generation_time': generation_time,
            'token_count': token_count,
            'img_id': img_id
        }
        
    except Exception as e:
        print(f"Error processing {img_id}: {e}")
        return {
            'success': False,
            'generation_time': 0,
            'token_count': 0,
            'img_id': img_id,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Unified Img2CAD inference: images -> h5 files")
    
    # Get project directory
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Input/Output paths
    parser.add_argument('--img_dir', type=str, 
                       default=os.path.join(project_dir, 'data', 'blender_renderings'), 
                       help="source folder for input images")
    parser.add_argument('--out_dir', type=str, default=None,
                       help="output folder for h5 files (default: {project_dir}/data/output/llamaft_h5/{category})")
    parser.add_argument('--adapter_path', type=str, default=None, 
                       help="path to fine-tuned adapter (auto-detected if not provided)")
    
    # Model parameters  
    parser.add_argument('--num_tokens', type=int, default=1024, help="max tokens per response")
    parser.add_argument('--category', default='chair', help="category")
    parser.add_argument('--split', default='test', help="split")
    parser.add_argument('--max_samples', type=int, default=None, 
                       help="evaluate on first k samples only (for quick testing)")
    parser.add_argument('--save_text', action='store_true', 
                       help="save intermediate text responses for debugging")
    
    # Authentication
    parser.add_argument('--hf_token', type=str, default=None, 
                       help="Hugging Face token (or set HF_TOKEN environment variable)")
    
    args = parser.parse_args()
    
    # Set default out_dir if not provided by user
    if args.out_dir is None:
        args.out_dir = os.path.join(project_dir, 'data', 'output', 'llamaft_h5', args.category)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Handle Hugging Face authentication
    hf_token = args.hf_token or os.getenv('HF_TOKEN')
    if hf_token:
        login(hf_token)
    else:
        print("Warning: No HF_TOKEN provided. Using public models only.")
    
    # Set generation parameters based on mode
    generation_params = {
        'top_p': 1.0,
        'temperature': 1.0,
        'do_sample': False
    }
    
    prompt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompt.txt')
    if args.adapter_path is None:
        adapter_search_dir = os.path.join(project_dir, 'data', 'ckpts', 'llamaft', args.category)
        adapter_checkpoints = glob(os.path.join(adapter_search_dir, 'checkpoint-*'))
        if adapter_checkpoints:
            args.adapter_path = sorted(adapter_checkpoints, key=lambda x: int(x.split('checkpoint-')[-1]))[-1]
    
    assert args.adapter_path is not None, "Adapter path is not provided"
    
    # Load prompt
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    prompt = open(prompt_path).read()
    
    # Load model
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    model, processor = load_model_and_processor(
        model_id, 
        adapter_path=args.adapter_path
    )
    
    # Prepare image list
    print(f"Starting unified inference on {args.category} {args.split} split...")
    print(f"Output directory: {args.out_dir}")
    print(f"Parameters: {generation_params}")
    
    # Use split file to get object IDs
    splits_path = os.path.join(project_dir, 'data', 'splits', f'{args.category}_{args.split}_ids.txt')
    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"Split file not found: {splits_path}")
        
    obj_ids = open(splits_path).read().splitlines()
    if args.max_samples is not None:
        total_ids = len(obj_ids)
        obj_ids = obj_ids[:args.max_samples]
        print(f"Processing first {len(obj_ids)} samples out of {total_ids} total")
    obj_ids = ['2212']
    
    image_list = []
    for img_id in obj_ids:
        img_path = os.path.join(args.img_dir, f'{img_id}.png')
        if os.path.exists(img_path):
            image_list.append((img_path, img_id))
        else:
            print(f"Warning: Image not found: {img_path}")
    
    # Process images
    results = []
    successful_conversions = 0
    
    for img_path, img_id in tqdm(image_list, desc="Processing images"):
        result = process_single_image(
            args, img_path, img_id, model, processor, prompt, generation_params
        )
        results.append(result)
        if result['success']:
            successful_conversions += 1
    
    # Print statistics
    if results:
        generation_times = [r['generation_time'] for r in results if r['success']]
        token_counts = [r['token_count'] for r in results if r['success']]
        
        print("\n" + "="*60)
        print("UNIFIED PIPELINE STATISTICS")
        if args.max_samples is not None:
            print(f"(LIMITED EVALUATION - FIRST {args.max_samples} SAMPLES)")
        print("="*60)
        print(f"Total images processed: {len(results)}")
        print(f"Successful h5 conversions: {successful_conversions}")
        print(f"Success rate: {successful_conversions/len(results)*100:.1f}%")
        
        if generation_times:
            print(f"\nGENERATION TIME PER IMAGE:")
            print(f"  Mean: {statistics.mean(generation_times):.2f} seconds")
            print(f"  Median: {statistics.median(generation_times):.2f} seconds")
            print(f"  Min: {min(generation_times):.2f} seconds")
            print(f"  Max: {max(generation_times):.2f} seconds")
            
        if token_counts:
            print(f"\nTOKENS GENERATED:")
            print(f"  Mean: {statistics.mean(token_counts):.1f} tokens")
            print(f"  Median: {statistics.median(token_counts):.1f} tokens")
            print(f"  Min: {min(token_counts)} tokens")
            print(f"  Max: {max(token_counts)} tokens")
            print(f"  Total tokens: {sum(token_counts)} tokens")
        
        print("="*60)
        
        # Save statistics
        stats_path = os.path.join(args.out_dir, 'pipeline_stats.txt')
        with open(stats_path, 'w') as f:
            f.write("UNIFIED PIPELINE STATISTICS\n")
            f.write("="*60 + "\n")
            f.write(f"Category: {args.category}\n")
            f.write(f"Split: {args.split}\n")
            f.write(f"Total images processed: {len(results)}\n")
            f.write(f"Successful h5 conversions: {successful_conversions}\n")
            f.write(f"Success rate: {successful_conversions/len(results)*100:.1f}%\n")
            if generation_times:
                f.write(f"Mean generation time: {statistics.mean(generation_times):.2f} seconds\n")
            if token_counts:
                f.write(f"Mean tokens generated: {statistics.mean(token_counts):.1f}\n")
        
        print(f"Statistics saved to: {stats_path}")


if __name__ == '__main__':
    main()