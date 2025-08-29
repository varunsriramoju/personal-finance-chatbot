import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

logger = logging.getLogger(__name__)

class GraniteAI:
    def __init__(self, model_name="ibm-granite/granite-3.0-2b-instruct"):
        """
        Initialize Granite AI with a 2B parameter model
        Available smaller Granite models:
        - "ibm-granite/granite-3.0-2b-instruct" (2B parameters)
        - "ibm-granite/granite-3.0-1b-a400m-instruct" (1B parameters)
        """
        try:
            logger.info(f"Loading Granite model: {model_name}")
            
            # Check if CUDA is available, otherwise use CPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimized settings for smaller models
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,  # Optimize for memory usage
                use_cache=True           # Enable KV cache for faster generation
            )

            # Create generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False  # Only return generated text
            )

            logger.info("Granite AI initialized successfully ✅")
            
        except Exception as e:
            logger.error(f"Failed to load Granite model: {e}")
            self.pipeline = None
            self.model = None
            self.tokenizer = None

    def generate_response(self, prompt, context=None):
        """Generate AI response based on prompt and optional financial context"""
        if not self.pipeline:
            return "⚠️ Granite AI is not available right now."

        try:
            # Create a structured prompt for financial advice
            if context:
                income = context.get('income', 0)
                total_expenses = context.get('total_expenses', 0)
                net_savings = context.get('net_savings', 0)
                currency_symbol = context.get('currency_symbol', '₹')
                
                full_prompt = f"""You are a knowledgeable personal finance assistant. Provide clear, practical, and actionable financial advice.

Financial Information:
- Monthly Income: {currency_symbol}{income:,.0f}
- Monthly Expenses: {currency_symbol}{total_expenses:,.0f}
- Net Savings: {currency_symbol}{net_savings:,.0f}

User Question: {prompt}

Please provide helpful financial advice:"""
            else:
                full_prompt = f"""You are a personal finance expert. Provide clear and practical financial guidance.

Question: {prompt}

Financial advice:"""

            # Generate response
            outputs = self.pipeline(
                full_prompt, 
                num_return_sequences=1,
                max_new_tokens=200,
                temperature=0.6,
                top_p=0.85,
                repetition_penalty=1.1
            )
            
            response = outputs[0]["generated_text"].strip()
            
            # Clean up the response
            response = self._clean_response(response)
            
            return response if response else "I'd be happy to help with your financial question. Could you provide more specific details?"
            
        except Exception as e:
            logger.error(f"Granite AI response error: {e}")
            return "⚠️ I'm having trouble generating a response right now. Please try rephrasing your question."
    
    def _clean_response(self, response):
        """Clean and format the AI response"""
        # Remove common prompt artifacts
        cleanup_phrases = [
            "Financial advice:",
            "Please provide helpful financial advice:",
            "You are a personal finance expert.",
            "You are a knowledgeable personal finance assistant."
        ]
        
        for phrase in cleanup_phrases:
            response = response.replace(phrase, "").strip()
        
        # Remove excessive whitespace
        response = ' '.join(response.split())
        
        # Ensure response isn't too short
        if len(response.split()) < 5:
            return "Let me help you with that. Could you provide more details about your specific financial situation or question?"
        
        return response
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if not self.model:
            return {"error": "Model not loaded"}
        
        try:
            # Get model parameters count
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            return {
                "model_name": self.model.config.name_or_path if hasattr(self.model.config, 'name_or_path') else "Unknown",
                "total_parameters": f"{total_params:,}",
                "trainable_parameters": f"{trainable_params:,}",
                "device": next(self.model.parameters()).device.type,
                "dtype": str(next(self.model.parameters()).dtype)
            }
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {"error": "Could not retrieve model information"}

# Alternative smaller models you can try:
class GraniteAISmaller:
    """Even smaller Granite model options"""
    
    @staticmethod
    def get_available_models():
        """Get list of available smaller Granite models"""
        return {
            "granite-3.0-2b-instruct": {
                "name": "ibm-granite/granite-3.0-2b-instruct",
                "size": "2B parameters",
                "description": "Balanced performance and efficiency"
            },
            "granite-3.0-1b-a400m-instruct": {
                "name": "ibm-granite/granite-3.0-1b-a400m-instruct", 
                "size": "1B parameters",
                "description": "Smallest, fastest option"
            },
            "granite-3.0-2b-base": {
                "name": "ibm-granite/granite-3.0-2b-base",
                "size": "2B parameters",
                "description": "Base model (needs more prompting)"
            }
        }
    
    @classmethod
    def create_with_model(cls, model_key):
        """Create GraniteAI instance with specific model"""
        models = cls.get_available_models()
        if model_key in models:
            model_name = models[model_key]["name"]
            return GraniteAI(model_name=model_name)
        else:
            raise ValueError(f"Model {model_key} not found. Available: {list(models.keys())}")

# Quick test function
def test_granite_model():
    """Test function to verify model loading"""
    try:
        ai = GraniteAI()
        if ai.pipeline:
            print("✅ Granite AI loaded successfully!")
            
            # Get model info
            info = ai.get_model_info()
            print(f"Model Info: {info}")
            
            # Test response
            test_context = {
                "income": 5000,
                "total_expenses": 3500,
                "net_savings": 1500,
                "currency_symbol": "₹"
            }
            
            response = ai.generate_response("How can I improve my savings rate?", test_context)
            print(f"Test Response: {response[:100]}...")
            
        else:
            print("❌ Failed to load Granite AI")
            
    except Exception as e:
        print(f"❌ Error testing model: {e}")

if __name__ == "__main__":
    test_granite_model()