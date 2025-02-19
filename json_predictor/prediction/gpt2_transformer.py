from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F

class gpt2TokenPredictor:
    # model name   
    MODEL_NAME = "gpt2"

    def __init__(self):
        # initialize gpt2
        model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token  # set tokenizer to use eos_tokens as padding


        #initialize sentence transformer model to process better sentence embeddings
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

        return next_word, softmax



    def semantic_change_score(str1, str2):
        # calculates cosine simiarity of sentence transformer sentence embeddings
        tokens1 = embedding_tokenizer(str1, return_tensors="pt", padding=True, truncation=True)
        tokens2 = embedding_tokenizer(str2, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embeddings1 = embedding_model(**tokens1).last_hidden_state.mean(dim=1)  # Mean pooling
            embeddings2 = embedding_model(**tokens2).last_hidden_state.mean(dim=1)  # Mean pooling
        
        cosine_similarity = F.cosine_similarity(embeddings1, embeddings2).item() # Compute cosine similarity between the two embeddings
        return 1 - cosine_similarity
