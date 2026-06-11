import json
import os
import re

models = ['Nano_data', 'Klein_data']
full=0
valid=0
input_path = 'InterleaveThinker/Train-Data'
output_path = 'interleaved_gen_simple.json'
final=[]
for model in models:
    model_path = os.path.join(input_path, model)
    for file in os.listdir(model_path):
        image_paths = os.path.join(model_path, file)
        json_path = f"{image_paths}/execution_record.json"
        if not os.path.exists(json_path):
            continue
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        try:
            global_context = data['global_context']
            planning = global_context['execution_plan']
            for it in planning:
                pp=it['instruction']
                pp=it['prompt']
                pp=it['auxiliary_text']
            if "None}, {'step_number'" in planning[0]['auxiliary_text']:
                raise ValueError("Invalid execution plan format")
        except:
            continue

        full_steps = len(planning)
        critic = data['steps']
        full+=1
        count=0
        images=[]
        for step in critic:
            if step['success']:
                count+=1
                images.append(step['target_img'])
        if count!=full_steps:
            continue

        prompt = data['user_input']
        plan_reasoning = global_context['reasoning']

        interleaved_sequence = f"<gthink>\n{plan_reasoning}\n</gthink>\n"
        for i in range(full_steps):
            interleaved_sequence+=f"<plan>\n{planning[i]['prompt']}\n<plan>\n"
            interleaved_sequence+='<image><img_token></image>.\n'
            # Optional
            if planning[i]['auxiliary_text'] is not None:
                interleaved_sequence+=f"<info>\n{planning[i]['auxiliary_text']}\n</info>\n"

        item = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                },
                {
                    "role": "assistant",
                    "content": interleaved_sequence
                }
            ],
            "images": images 
        }
        final.append(item)
        valid+=1

print(f"Total: {full}, Valid: {valid}, list_len: {len(final)}")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(final, f, ensure_ascii=False, indent=4)
            