# clean_captions.py
import json
import os

# Load raw captions.json with structure: { video_id: { "start": [...], "end": [...], "text": [...] } }
with open(r"C:\Aneesh\EDI VI\data\captions.json", "r") as f:
    raw_captions = json.load(f)

cleaned = []

for video_id, info in raw_captions.items():
    starts = info.get("start", [])
    texts = info.get("text", [])
    
    # Make sure starts and texts align
    if not starts or not texts or len(starts) != len(texts):
        continue
    
    for i in range(len(starts)):
        caption = texts[i].strip()
        start_time = starts[i]
        if caption:  # skip empty captions
            cleaned.append({
                "video_id": video_id,
                "start_time": float(start_time),
                "caption": caption
            })

os.makedirs(r"C:\Aneesh\EDI VI\data\processed", exist_ok=True)

with open(r"C:\Aneesh\EDI VI\data\processed\cleaned_captions.json", "w") as f:
    json.dump(cleaned, f, indent=2)

print(f"Saved {len(cleaned)} cleaned captions to data/processed/cleaned_captions.json")
