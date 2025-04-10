# 1/23/2025
# this is a test of the next token prediction capability of the GPT-2 Model WITHOUT any finetuning
# this is a test of semantic shift detection using GPT2 hidden layers to compute sentence embeddings

from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
import logging
import matplotlib.pyplot as plt



logging.basicConfig(filename='semantic_sentenceTransformer.log', level=logging.INFO, format='%(message)s')
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

MODEL_NAME = "gpt2"


####################################################################################################################################
""" Input text for next-token prediction """

input_text = "The weather forecast predicted sunshine and clear skies, but instead it turned out to be stormy"

####################################################################################################################################



model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # set tokenizer to use eos_tokens as padding



embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = AutoModel.from_pretrained(embedding_model_name)
embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)




def next_token(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")  # "pt" for pytorch

    outputs = model(input_ids, labels=input_ids)
    logits = outputs.logits
    # Get the predicted next token (argmax of the last token's logits)
    next_token_logits = logits[:, -1, :]  # Logits for the last token in the sequence
    predicted_token_id = torch.argmax(next_token_logits, dim=-1).item()  # Get the token ID
    predicted_token = tokenizer.decode([predicted_token_id])  # Decode token ID to text

    #get softmax
    next_token_probs = F.softmax(next_token_logits, dim=-1)
    predicted_token_prob = next_token_probs[0, predicted_token_id].item()  

    return predicted_token, predicted_token_prob



def get_sentence_embedding(text):
    tokens = embedding_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = embedding_model(**tokens).last_hidden_state.mean(dim=1)  # Mean pooling
    return embeddings






softmax_probs = []
semantic_change_scores = []
labels = []


# test
words = input_text.split()
for i in range(1, len(words) + 1):
    substring = " ".join(words[:i])
    next_word = next_token(substring)[0]
    softmax = next_token(substring)[1]

    # semantic shift detection
    original_embedding = get_sentence_embedding(substring)
    new_embedding = get_sentence_embedding(substring + next_word)
    cosine_similarity = F.cosine_similarity(original_embedding, new_embedding).item() # Compute cosine similarity between the two embeddings
    semantic_change_score = 1 - cosine_similarity # Semantic change score (1 - similarity)


    # logging
    print(substring + bcolors.OKCYAN + next_word + bcolors.ENDC)
    logging.info(substring + next_word)
    print(f"{bcolors.OKBLUE}SoftMax: {softmax:.3f}{bcolors.ENDC}")
    logging.info(f"SoftMax: {softmax:.3f}")
    print(f"{bcolors.OKBLUE}Semantic change score: {semantic_change_score:.4f}{bcolors.ENDC}")
    logging.info(f"Semantic change score: {semantic_change_score:.4f}")



    # plotting
    softmax_probs.append(softmax)
    semantic_change_scores.append(semantic_change_score)
    actual_token = words[i] if i < len(words) else "(eos)"
    labels.append(f"{next_word}\n{actual_token}")



# Plot results
x = range(len(softmax_probs))  # X-axis positions
width = 0.4  # Width of the bars

fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot semantic change scores on the left y-axis
ax1.bar([p - width/2 for p in x], semantic_change_scores, width, color="skyblue", label="Semantic Change Score")
ax1.set_ylabel("Semantic Change Score", color="skyblue")
ax1.tick_params(axis="y", labelcolor="skyblue")

# Create a twin y-axis to plot softmax probabilities
ax2 = ax1.twinx()
ax2.bar([p + width/2 for p in x], softmax_probs, width, color="orange", label="Softmax Probability")
ax2.set_ylabel("Softmax Probability", color="orange")
ax2.tick_params(axis="y", labelcolor="orange")

# Add labels and legend
plt.xticks(x, labels, rotation=45, ha="right")
ax1.set_xlabel("Predicted vs Actual Tokens")
ax1.set_title(f"Semantic Change and Softmax Probability for Each Predicted Token\n{input_text}")

# Combine legends from both axes
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

plt.tight_layout()
plt.show()
