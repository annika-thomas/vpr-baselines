import torch

def print_pt_file_contents(file_path):
    try:
        # Load the contents of the .pt file
        contents = torch.load(file_path)

        # Print the contents
        # Note: The contents can be a model, tensor, or any serialized Python object
        print(contents.shape)

    except Exception as e:
        print(f"Error occurred: {e}")

# Replace 'your_file.pt' with the path to your .pt file
print_pt_file_contents('/home/annika/Downloads/c_centers.pt')