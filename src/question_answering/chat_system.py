import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class RFXConversationalQA:
    def __init__(self, model_id="meta-llama/Meta-Llama-3-8B-Instruct", max_input_tokens=2048):
        """
        Initialize the class with the LLaMA model and tokenizer.
        Args:
        - model_id (str): Hugging Face model ID for the LLaMA model (default is Meta LLaMA).
        - max_input_tokens (int): Maximum number of tokens to use for document truncation.
        """
        self.max_input_tokens = max_input_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.truncated_document = None
        self.conversation_history = []

    def truncate_input(self, text, max_tokens=None):
        """
        Truncate the input text to a specified maximum number of tokens.
        Args:
        - text (str): The full input text to be truncated.
        - max_tokens (int): The number of tokens to truncate to (optional).
        Returns:
        - truncated_text (str): The truncated text after limiting the number of tokens.
        """
        if not max_tokens:
            max_tokens = self.max_input_tokens
        tokenized_input = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
        truncated_text = self.tokenizer.decode(tokenized_input['input_ids'][0], skip_special_tokens=True)
        return truncated_text
    
    def load_document(self, markdown_file_path):
        """
        Load and truncate the markdown document to be used in the question-answering process.
        Args:
        - markdown_file_path (str): Path to the markdown file containing the document content.
        """
        with open(markdown_file_path, 'r', encoding='utf-8') as file:
            document_text = file.read()
        self.truncated_document = self.truncate_input(document_text)
        print(f"Document '{markdown_file_path}' loaded and truncated to {self.max_input_tokens} tokens.")
    
    def ask_question(self, question):
        """
        Ask a question about the loaded document, keeping track of the conversation context, 
        and generate a response using the LLaMA model.
        Args:
        - question (str): The user's question.
        Returns:
        - answer (str): The generated answer from the model.
        """
        if not self.truncated_document:
            raise ValueError("No document loaded. Please load a document first using `load_document`.")

        self.conversation_history.append({"role": "user", "content": question})

        conversation_context = ""
        for message in self.conversation_history:
            conversation_context += f"{message['role']}: {message['content']}\n"
        
        tokenized_conversation = self.tokenizer(conversation_context, return_tensors="pt")
        conversation_token_count = len(tokenized_conversation['input_ids'][0])
        available_tokens_for_document = self.max_input_tokens - conversation_token_count
        if available_tokens_for_document < 0:
            print("Conversation exceeds max token limit; truncating the conversation history.")
            truncated_conversation = self.truncate_input(conversation_context, max_tokens=self.max_input_tokens)
            conversation_context = truncated_conversation
            document_context = ""
        else:
            document_context = self.truncate_input(self.truncated_document, max_tokens=available_tokens_for_document)
        system_prompt = f"""
        You are an intelligent assistant that answers questions based on the provided document and conversation history.
        
        Document: {document_context}
        
        The conversation so far is: 
        {conversation_context}
        
        Now answer the latest question: {question}
        """

        messages = [
            {"role": "system", "content": system_prompt}
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        response = outputs[0][input_ids.shape[-1]:]
        answer = self.tokenizer.decode(response, skip_special_tokens=True)

        self.conversation_history.append({"role": "model", "content": answer})

        return answer

    def reset_conversation(self):
        """
        Reset the conversation history. Useful for starting a new conversation on the same or a different document.
        """
        self.conversation_history = []
        print("Conversation history has been reset.")