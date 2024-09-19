import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class RFXFeatureExtractor:
    def __init__(self, markdown_files, max_input_tokens=7048, model_id="meta-llama/Meta-Llama-3-8B-Instruct"):
        """
        Initialize the class with a list of markdown files and load the LLaMA model.
        Args:
        - markdown_files (list): List of markdown file paths to process.
        - max_input_tokens (int): Maximum number of tokens to keep in the input before truncating.
        - model_id (str): Hugging Face model ID for the LLaMA model (default is Meta LLaMA).
        """
        self.markdown_files = markdown_files
        self.max_input_tokens = max_input_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    
    def truncate_input(self, text):
        """
        Truncate the input text to a specified maximum number of tokens.
        Args:
        - text (str): The full input text to be truncated.
        Returns:
        - truncated_text (str): The truncated text after limiting the number of tokens.
        """
        tokenized_input = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_input_tokens)

        truncated_text = self.tokenizer.decode(tokenized_input['input_ids'][0], skip_special_tokens=True)
        
        return truncated_text

    def extract_features_from_rfx(self, rfx_text):
        """
        Extract features from the given RFX text using the LLaMA model.
        Args:
        - rfx_text (str): The markdown content of the RFX document.
        Returns:
        - extracted_features (str): The extracted features as a string.
        """
        truncated_text = self.truncate_input(rfx_text)
        
        # System prompt for extracting relevant features from the RFX document
        system_prompt = """
        You are an intelligent assistant that extracts procurement details from RFX documents.
        Extract the following features from the given RFX text:
        - Title: Title or name of the opportunity.
        - Scope of Work: Detailed description of the work, goods, or services required.
        - Estimated Budget or Value: If available, the approximate budget or value of the contract.
        - Bid Submission Requirements:
	        1. Format (e.g., electronic, physical submission).
	        2. 	Number of copies (if physical).
	        3.	Required forms or templates.
        - Mandatory Qualifications/Certifications: Requirements that vendors must meet (e.g., certifications, licenses, experience level).
        - Evaluation Criteria: How the bids will be evaluated (price-focused, quality, technical specifications, etc.).
        - Contract Type: Is it a lump-sum contract, time-and-materials, or other?
        - Contact Information: Details of the procurement officer or point of contact for queries.
        
        Respond with the extracted details in a structured format.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": truncated_text}
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
        extracted_features = self.tokenizer.decode(response, skip_special_tokens=True)

        return extracted_features

    def process_documents(self):
        """
        Process each markdown file, extract relevant features, and save the results as text files.
        """
        for file_path in self.markdown_files:
            with open(file_path, 'r', encoding='utf-8') as file:
                rfx_text = file.read()
            extracted_info = self.extract_features_from_rfx(rfx_text)
            
            output_file_path = f"{os.path.splitext(file_path)[0]}_extracted.txt"
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(extracted_info)
            
            print(f"Processed and saved extracted features to: {output_file_path}")
