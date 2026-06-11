"""
Qwen3-VL API: image + prompt -> answer.
Input: image (path or list of paths), prompt (str). Output: answer (str).
"""

from typing import Union, List, Optional

def predict(
    model,
    processor,
    messages = None,
    max_new_tokens: int = 4096,
) -> str:
  
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0].strip() if output_text else ""
