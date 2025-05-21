import json

# Load the triplets.json file
with open(r'C:\Aneesh\EDI VI\data\processed\triplets.json', 'r', encoding='utf-8') as file:
    triplets = json.load(file)

# Count total entries
total_entries = len(triplets)

# Count entries with empty or missing 'caption'
empty_captions = sum(1 for triplet in triplets if not triplet.get('caption'))

# Output results
print(f"Total entries: {total_entries}")
print(f"Entries with empty or missing captions: {empty_captions}")
