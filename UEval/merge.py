import json
import glob

all_results = []
for file_path in glob.glob("result/test*.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        all_results.extend(data)

with open("result/final.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"Successfully merged {len(all_results)} items!")