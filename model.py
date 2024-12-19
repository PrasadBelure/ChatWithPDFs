from transformers import AutoTokenizer, AutoModelForCausalLM

# Define the model name
model_name = "Groq/mixtral-70B"

# Download and save the tokenizer and model locally
try:
    print("Downloading the tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)  # use_auth_token if required
    print("Downloading the model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True)

    # Save the tokenizer and model locally
    save_path = "./offline_model_mixtral_70B"
    print(f"Saving the model and tokenizer to {save_path}...")
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)

    print("Model and tokenizer saved locally!")
except Exception as e:
    print(f"An error occurred: {e}")
