import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

logger = logging.getLogger(__name__)

class GraniteAI:
    def __init__(self, model_name="ibm-granite/granite-3b-code-instruct"):
        try:
            logger.info(f"Loading Granite model: {model_name}")
            
            # Check if CUDA is available, otherwise use CPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            )

            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

            logger.info("Granite AI initialized successfully ✅")
        except Exception as e:
            logger.error(f"Failed to load Granite model: {e}")
            self.pipeline = None

    def generate_response(self, prompt, context=None):
        """Generate AI response based on prompt and optional financial context"""
        if not self.pipeline:
            return "⚠️ Granite AI is not available right now."

        try:
            # Create a structured prompt for financial advice
            if context:
                full_prompt = f"""You are a helpful personal finance assistant. Based on the user's financial information, provide clear and actionable advice.

Financial Context:
- Monthly Income: ${context.get('income', 0):,.2f}
- Monthly Expenses: ${context.get('total_expenses', 0):,.2f}
- Net Savings: ${context.get('net_savings', 0):,.2f}
- Currency: {context.get('currency_symbol', '$')}

User Question: {prompt}

Financial Advisor Response:"""
            else:
                full_prompt = f"""You are a helpful personal finance assistant. Provide clear and actionable financial advice.

User Question: {prompt}

Financial Advisor Response:"""

            outputs = self.pipeline(
                full_prompt, 
                num_return_sequences=1,
                return_full_text=False
            )
            
            response = outputs[0]["generated_text"].strip()
            
            # Clean up the response
            if response.startswith("Financial Advisor Response:"):
                response = response.replace("Financial Advisor Response:", "").strip()
            
            return response if response else "I'd be happy to help with your financial question. Could you provide more details?"
            
        except Exception as e:
            logger.error(f"Granite AI response error: {e}")
            return "⚠️ Sorry, I could not generate a response. Please try again."