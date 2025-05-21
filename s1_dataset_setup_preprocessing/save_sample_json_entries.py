import json

def save_sample_json_entries(input_file, output_file, sample_size=50):
    try:
        with open(input_file, 'r') as file:
            data = json.load(file)

        if isinstance(data, dict):
            # Take the first N entries
            keys = list(data.keys())[:sample_size]
            sample = {key: data[key] for key in keys}

            # Save to output file
            with open(output_file, 'w') as outfile:
                json.dump(sample, outfile, indent=4)
            print(f"Saved {len(sample)} entries to {output_file}")

        else:
            print("Error: JSON format not supported (must be a dictionary).")

    except Exception as e:
        print(f"Error processing {input_file}: {e}")


# Set your file paths
captions_input = r"C:\Aneesh\EDI VI\data\captions.json"
extensions_input = r"C:\Aneesh\EDI VI\data\extensions.json"

# Output sample files
captions_output = "captions_sample.json"
extensions_output = "extensions_sample.json"

# Save samples
save_sample_json_entries(captions_input, captions_output, sample_size=50)
save_sample_json_entries(extensions_input, extensions_output, sample_size=50)
