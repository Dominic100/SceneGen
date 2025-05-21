import json

# Load the JSON file
def count_entries(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    # Assuming the JSON file contains a list of entries
    if isinstance(data, list):
        return len(data)
    elif isinstance(data, dict):
        return len(data.keys())
    else:
        return "Unsupported JSON format"

# Example usage
json_file = r"C:\Aneesh\EDI VI\data\extensions.json"  # Replace with your actual JSON file path
print(f"Number of entries: {count_entries(json_file)}")
