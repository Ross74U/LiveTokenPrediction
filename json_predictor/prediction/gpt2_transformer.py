from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F

from prediction.predictor_base import Predictor, TokenInfo

class gpt2TokenPredictor(Predictor):

    def __init__(self):
        # initialize gpt2
        self.model_name = "gpt2"
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # set tokenizer to use eos_tokens as padding

        #initialize sentence transformer model to process better sentence embeddings
        self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.embedding_model = AutoModel.from_pretrained(self.embedding_model_name)
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)

        


    def next_token(self, input_text: str) -> TokenInfo: 

        def getSemanticChange(self, str1: str, str2: str) -> float:
            # calculates cosine simiarity of sentence transformer sentence embeddings
            tokens1 = self.embedding_tokenizer(str1, return_tensors="pt", padding=True, truncation=True)
            tokens2 = self.embedding_tokenizer(str2, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                embeddings1 = self.embedding_model(**tokens1).last_hidden_state.mean(dim=1)  # Mean pooling
                embeddings2 = self.embedding_model(**tokens2).last_hidden_state.mean(dim=1)  # Mean pooling
            
            cosine_similarity = F.cosine_similarity(embeddings1, embeddings2).item() # Compute cosine similarity between the two embeddings
            return 1 - cosine_similarity       


        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")  # "pt" for pytorch

        outputs = self.model(input_ids, labels=input_ids)
        logits = outputs.logits
        # Get the predicted next token (argmax of the last token's logits)
        next_token_logits = logits[:, -1, :]  # Logits for the last token in the sequence
        predicted_token_id = torch.argmax(next_token_logits, dim=-1).item()  # Get the token ID
        predicted_token = self.tokenizer.decode([predicted_token_id])  # Decode token ID to text
        #get softmax
        next_token_probs = F.softmax(next_token_logits, dim=-1)
        softmax = next_token_probs[0, predicted_token_id].item()  

        semantic_change_score = getSemanticChange(self, input_text, input_text + predicted_token)

        return predicted_token, softmax, semantic_change_score

