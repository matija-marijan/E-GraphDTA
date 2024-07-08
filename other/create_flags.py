import json

# Load your JSON file
with open('/home/matijamarijan/projects/GraphDTA/data/davis/final_proteins.json', 'r') as file:
    data = json.load(file)

# Initialize the binary matrix
binary_matrix = []

# Iterate through the keys in the JSON file
for key in data.keys():
    vector = [0]
    if "-phosphorylated" in key:
        vector[0] = 1
    # if "cyclinD3" in key:
    #     vector[1] = 1
    # if "(ITD)" in key:
    #     vector[2] = 1
    binary_matrix.append(vector)

# Print the binary matrix
for row in binary_matrix:
    print(row)

# If you want to save the binary vector to a file
with open('binary_vector.json', 'w') as file:
    json.dump(binary_matrix, file)
