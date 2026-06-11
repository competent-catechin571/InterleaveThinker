import json
import os
import re

models = ['Nano_data', 'Klein_data']
full=0
valid=0
input_path = 'InterleaveThinker/Train-Data'
final=[]
output_path = 'interleaved_gen_reflection.json'
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

        full_steps = planning[-1]['step_number']
        critic = data['steps']
        full+=1
        count=0
        for step in critic:
            if step['success']:
                count+=1
        if count!=full_steps:
            continue

        prompt = data['user_input']
        plan_reasoning = global_context['reasoning']
        images=[]

        interleaved_sequence = f"<gthink>\n{plan_reasoning}\n</gthink>\n"
        for i in range(full_steps):
            interleaved_sequence+=f"<plan>\n{planning[i]['prompt']}\n<plan>\n"
            
            for critic_step in critic:
                step_num = critic_step['step']
                if step_num!=i+1:
                    continue
                interleaved_sequence+='<image><img_token></image>.\n'
                images.append(critic_step['target_img'])

                critic_item = {
                    'success': critic_step['success'],
                    'refine_prompt': critic_step['refine_prompt']
                }
                critic_str = json.dumps(critic_item, ensure_ascii=False)
                interleaved_sequence+=f"<critic>\n<think>\n{critic_step['reasoning']}\n</think>\n{critic_str}\n</critic>\n"

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
            