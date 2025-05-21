import json

def print_json_entry(json_file, index):
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)

        # If JSON is a dictionary, fetch the nth key and its value
        if isinstance(data, dict):
            keys = list(data.keys())  # Convert keys to a list
            if 0 <= index < len(keys):
                key = keys[index]
                print(f"Entry at index {index}:")
                print(json.dumps({key: data[key]}, indent=4))
            else:
                print("Error: Index out of range!")

        else:
            print("Error: JSON format not supported!")

    except Exception as e:
        print(f"Error: {e}")

# User Input (without checking full length)
json_file = input("Enter the JSON file path: ")
index = int(input("Enter the index: "))

print_json_entry(json_file, index)
