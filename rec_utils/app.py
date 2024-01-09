# Define a custom PyTorch DataLoader class for recommendation datasets, inheriting from DataLoader
class RecommendationDataloader(DataLoader):
    # Constructor to initialize the dataloader with required parameters and additional options
    def __init__(self, dataset, original_sequences, num_items, out_key="out_sid", relevance=None, num_negatives=1, **kwargs):
        # Initialize basic parameters
        self.num_items = num_items
        # Convert original_sequences to a list of sets for faster lookup
        self.original_sequences = [set(x) for x in original_sequences]
        self.out_key = out_key

        # Get the shape of the output from the dataset
        for app in dataset:
            break
        # Set the number of positives based on the shape of the output
        self.num_positives = app[self.out_key].shape[-1]

        self.num_negatives = num_negatives

        # Generate a relevance function based on the specified relevance type and shape of the output
        self.relevance_function = generate_relevance(relevance, app[self.out_key].shape)

        # Call the constructor of the parent class (DataLoader)
        super().__init__(dataset, **kwargs)

    # Method to sample negative items for a given set of indices
    def sample_negatives(self, indices):
        negatives = torch.zeros(len(indices), self.num_negatives)
        for i, index in enumerate(indices):
            # Get possible negative items that are not in the original sequence
            possible_negatives = torch.tensor(list(set(range(1, self.num_items + 1)).difference(self.original_sequences[index])))
            # Randomly sample num_negatives negative items
            negatives[i] = possible_negatives[torch.randperm(len(possible_negatives))[:self.num_negatives]]
        return negatives

    # Custom iterator method to yield batches with additional information
    def __iter__(self):
        for out in super().__iter__():
            # Print the current batch for debugging purposes
            print(out)
            # Add negative samples and relevance scores to the batch
            out[f"{self.out_key}_negatives"] = self.sample_negatives(out["uid"])
            out["relevance"] = self.relevance_function(out)
            # Yield the modified batch
            yield out

# Function to generate a relevance function based on the specified relevance type and data shape
def generate_relevance(relevance, data_shape):
    if relevance in {None, "fixed", "linear", "exponential"}:
        # Generate a relevance tensor based on the specified type and data shape
        relevance = generate_relevance_from_type(relevance, data_shape)
        # Define a lambda function to return the generated relevance tensor
        relevance_function = lambda data: relevance
    elif isinstance(relevance, str):
        # Use an existing output key as the relevance function
        relevance_function = lambda data: data[f"out_{relevance}"]
    else:
        # Raise an error for unsupported relevance types
        raise NotImplementedError(f"Unsupported relevance: {relevance}")

    return relevance_function

# Function to generate a relevance tensor based on the specified relevance type and shape
def generate_relevance_from_type(relevance_type, shape):
    if relevance_type is None or relevance_type == "fixed":
        # Generate a tensor of ones if the relevance type is None or fixed
        app = torch.ones(shape[-1])
    else:
        # Generate a tensor with values from 1 to 0 with equal spacing
        app = torch.linspace(0, 1, shape[-1])[::-1]
        # Adjust the tensor based on the relevance type
        if relevance_type == "linear":
            pass
        elif relevance_type == "exponential":
            app = torch.exp(app)

    # Normalize the tensor
    app /= torch.sum(app)
    # Repeat the tensor to match the shape
    app = app.repeat(*shape[:-1], 1)
    return app
