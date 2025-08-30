import logging
import torch
import signal
import time
from typing import Optional, Dict, Any
import traceback

# Only import transformers if available
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoModelForCausalLM = None
    AutoTokenizer = None
    pipeline = None

logger = logging.getLogger(__name__)

class TimeoutError(Exception):
    """Custom timeout exception"""
    pass

def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutError("Operation timed out")

class GraniteAI:
    def __init__(self, model_name="ibm-granite/granite-3.0-1b-a400m-instruct", timeout=300):
        """
        Initialize Granite AI with timeout protection
        
        Args:
            model_name: HuggingFace model name
            timeout: Maximum seconds to wait for model loading
        """
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.model_name = model_name
        
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers library not available. Install with: pip install transformers torch")
            return
        
        try:
            logger.info(f"Loading Granite model: {model_name}")
            
            # Set up timeout protection
            if hasattr(signal, 'SIGALRM'):  # Unix-like systems
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout)
            
            start_time = time.time()
            
            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            # Load tokenizer first (faster)
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=True  # Use fast tokenizer if available
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Loading model...")
            # Load model with very conservative settings
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_cache=True,
                attn_implementation="eager"  # Use eager attention for compatibility
            )

            logger.info("Creating pipeline...")
            # Create generation pipeline with smaller settings
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=150,  # Reduced from 256
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False,
                batch_size=1  # Process one at a time
            )

            # Clear timeout
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            
            load_time = time.time() - start_time
            logger.info(f"Granite AI initialized successfully in {load_time:.1f}s ✅")
            
        except TimeoutError:
            logger.error(f"Model loading timed out after {timeout} seconds")
            self._cleanup()
        except Exception as e:
            logger.error(f"Failed to load Granite model: {e}")
            logger.error(traceback.format_exc())
            self._cleanup()
        finally:
            # Clear any remaining timeout
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)

    def _cleanup(self):
        """Clean up model resources"""
        self.pipeline = None
        self.model = None
        self.tokenizer = None

    def generate_response(self, prompt: str, context: Optional[Dict[str, Any]] = None, 
                         timeout: int = 30) -> str:
        """Generate AI response with timeout protection"""
        
        if not self.pipeline:
            return "⚠ Granite AI is not available right now."

        try:
            # Set up timeout for response generation
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout)
            
            start_time = time.time()
            
            # Create a structured prompt for financial advice
            if context:
                income = context.get('income', 0)
                total_expenses = context.get('total_expenses', 0)
                net_savings = context.get('net_savings', 0)
                currency_symbol = context.get('currency_symbol', '₹')
                
                # Shorter, more focused prompt to reduce processing time
                full_prompt = f"""As a financial advisor, provide clear advice.

Income: {currency_symbol}{income:,.0f}/month
Expenses: {currency_symbol}{total_expenses:,.0f}/month
Savings: {currency_symbol}{net_savings:,.0f}/month

Question: {prompt}

Advice:"""
            else:
                full_prompt = f"""Financial question: {prompt}

Provide practical advice:"""

            # Generate response with reduced parameters for speed
            outputs = self.pipeline(
                full_prompt, 
                num_return_sequences=1,
                max_new_tokens=100,  # Reduced for faster generation
                temperature=0.6,
                top_p=0.85,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Clear timeout
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            
            response = outputs[0]["generated_text"].strip()
            
            # Clean up the response
            response = self._clean_response(response)
            
            generation_time = time.time() - start_time
            logger.info(f"Response generated in {generation_time:.1f}s")
            
            return response if response else "I'd be happy to help with your financial question. Could you provide more specific details?"
            
        except TimeoutError:
            logger.error(f"Response generation timed out after {timeout} seconds")
            return "⚠ Response took too long to generate. Please try a simpler question or check system resources."
        except Exception as e:
            logger.error(f"Granite AI response error: {e}")
            return "⚠ I'm having trouble generating a response right now. Please try rephrasing your question."
        finally:
            # Clear any remaining timeout
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the AI response"""
        # Remove common prompt artifacts
        cleanup_phrases = [
            "Financial advice:",
            "Advice:",
            "Provide practical advice:",
            "As a financial advisor, provide clear advice.",
            "Financial question:",
            "Question:",
            "Response:",
        ]
        
        for phrase in cleanup_phrases:
            response = response.replace(phrase, "").strip()
        
        # Remove excessive whitespace and newlines
        response = ' '.join(response.split())
        
        # Remove incomplete sentences at the end
        sentences = response.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            response = '.'.join(sentences[:-1]) + '.'
        
        # Ensure response isn't too short
        if len(response.split()) < 5:
            return "Let me help you with that. Could you provide more details about your specific financial situation or question?"
        
        return response
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.model:
            return {"error": "Model not loaded"}
        
        try:
            # Get model parameters count
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            return {
                "model_name": self.model_name,
                "total_parameters": f"{total_params:,}",
                "trainable_parameters": f"{trainable_params:,}",
                "device": next(self.model.parameters()).device.type,
                "dtype": str(next(self.model.parameters()).dtype),
                "transformers_available": TRANSFORMERS_AVAILABLE
            }
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {"error": f"Could not retrieve model information: {e}"}

    def check_health(self) -> Dict[str, Any]:
        """Check if the model is healthy and responsive"""
        if not self.pipeline:
            return {"status": "unhealthy", "reason": "Pipeline not loaded"}
        
        try:
            # Quick test generation
            test_response = self.generate_response("Test", timeout=5)
            if test_response and "error" not in test_response.lower():
                return {"status": "healthy", "test_response_length": len(test_response)}
            else:
                return {"status": "unhealthy", "reason": "Test generation failed"}
        except Exception as e:
            return {"status": "unhealthy", "reason": str(e)}

# Alternative: Lightweight fallback model
class SimpleFallbackAI:
    """Simple rule-based fallback when transformer models fail"""
    
    def __init__(self):
        self.pipeline = "fallback"  # Indicate fallback mode
        self.response_templates = self._load_response_templates()
    
    def _load_response_templates(self) -> Dict[str, list]:
        """Load pre-written response templates"""
        return {
            "savings": [
                "To improve your savings rate, focus on the 50/30/20 rule: 50% needs, 30% wants, 20% savings.",
                "Start by automating your savings - pay yourself first before other expenses.",
                "Track your expenses for 30 days to identify areas where you can cut back."
            ],
            "investment": [
                "For beginners, start with low-cost index funds that track the S&P 500.",
                "Consider your age in bonds (if you're 30, have 30% in bonds, 70% in stocks).",
                "Dollar-cost averaging helps reduce the impact of market volatility."
            ],
            "debt": [
                "Use the debt avalanche method: pay minimums on all debts, extra on highest interest rate.",
                "Consider debt consolidation if you can get a lower interest rate.",
                "Focus on building a small emergency fund first, then attack debt aggressively."
            ],
            "emergency": [
                "Build your emergency fund to cover 3-6 months of essential expenses.",
                "Keep emergency funds in a high-yield savings account for easy access.",
                "Start small - even $500 can help avoid going into debt for minor emergencies."
            ],
            "budget": [
                "Use the envelope method: allocate cash for each spending category.",
                "Review and adjust your budget monthly based on actual spending patterns.",
                "Focus on your biggest expenses first - small changes to large categories have big impact."
            ]
        }
    
    def generate_response(self, prompt: str, context: Dict[str, Any] = None, **kwargs) -> str:
        """Generate rule-based response"""
        prompt_lower = prompt.lower()
        
        # Determine category based on keywords
        if any(word in prompt_lower for word in ["save", "saving", "savings"]):
            category = "savings"
        elif any(word in prompt_lower for word in ["invest", "investment", "portfolio"]):
            category = "investment"
        elif any(word in prompt_lower for word in ["debt", "loan", "credit"]):
            category = "debt"
        elif any(word in prompt_lower for word in ["emergency", "fund"]):
            category = "emergency"
        elif any(word in prompt_lower for word in ["budget", "expense", "spending"]):
            category = "budget"
        else:
            category = "savings"  # Default
        
        # Get contextual advice
        base_response = self.response_templates[category][0]  # Get first template
        
        # Add context-specific advice if available
        if context:
            savings_rate = context.get('savings_rate', 0)
            net_savings = context.get('net_savings', 0)
            
            if savings_rate < 0:
                base_response += f"\n\nYour current situation shows expenses exceeding income. Priority #1 is to balance your budget by reducing expenses or increasing income."
            elif savings_rate < 10:
                base_response += f"\n\nWith a {savings_rate:.1f}% savings rate, focus on reaching 10-15% as your next milestone."
            else:
                base_response += f"\n\nGreat job with your {savings_rate:.1f}% savings rate! Consider optimizing for even better results."
        
        return base_response
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get fallback model info"""
        return {
            "model_name": "Rule-based Fallback",
            "type": "template_based",
            "status": "active"
        }

# Factory function to create appropriate AI instance
def create_ai_assistant(prefer_fallback=False) -> object:
    """Create AI assistant with automatic fallback"""
    
    if prefer_fallback or not TRANSFORMERS_AVAILABLE:
        logger.info("Using fallback AI assistant")
        return SimpleFallbackAI()
    
    try:
        # Try to create Granite AI
        granite_ai = GraniteAI()
        if granite_ai.pipeline:
            return granite_ai
        else:
            logger.info("Granite AI failed, using fallback")
            return SimpleFallbackAI()
    except Exception as e:
        logger.error(f"Failed to create Granite AI: {e}")
        return SimpleFallbackAI()

# Keep existing classes for compatibility
class GraniteAISmaller:
    """Smaller Granite model options"""
    
    @staticmethod
    def get_available_models():
        """Get list of available smaller Granite models"""
        return {
            "granite-3.0-1b-a400m-instruct": {
                "name": "ibm-granite/granite-3.0-1b-a400m-instruct", 
                "size": "1B parameters",
                "description": "Smallest, fastest option",
                "recommended": True
            },
            "granite-3.0-2b-instruct": {
                "name": "ibm-granite/granite-3.0-2b-instruct",
                "size": "2B parameters", 
                "description": "Balanced performance (may be slow)",
                "recommended": False
            }
        }
    
    @classmethod
    def create_with_model(cls, model_key):
        """Create GraniteAI instance with specific model"""
        models = cls.get_available_models()
        if model_key in models:
            model_name = models[model_key]["name"]
            return GraniteAI(model_name=model_name, timeout=180)  # Longer timeout for larger models
        else:
            raise ValueError(f"Model {model_key} not found. Available: {list(models.keys())}")

# Quick test function with better error handling
def test_granite_model(timeout=60):
    """Test function to verify model loading with timeout"""
    try:
        print("🧪 Testing Granite AI initialization...")
        
        # Test fallback first
        print("\n📋 Testing fallback AI...")
        fallback_ai = SimpleFallbackAI()
        fallback_response = fallback_ai.generate_response("How can I save money?")
        print(f"✅ Fallback AI working: {fallback_response[:100]}...")
        
        # Test Granite AI if transformers available
        if not TRANSFORMERS_AVAILABLE:
            print("❌ Transformers not available - install with: pip install transformers torch")
            return False
        
        print(f"\n🤖 Testing Granite AI (timeout: {timeout}s)...")
        ai = GraniteAI(timeout=timeout)
        
        if ai.pipeline:
            print("✅ Granite AI loaded successfully!")
            
            # Get model info
            info = ai.get_model_info()
            print(f"📊 Model Info: {info}")
            
            # Test response with timeout
            print("🧪 Testing response generation...")
            test_context = {
                "income": 5000,
                "total_expenses": 3500,
                "net_savings": 1500,
                "currency_symbol": "₹",
                "savings_rate": 30.0
            }
            
            response = ai.generate_response("How can I improve my savings rate?", test_context, timeout=15)
            print(f"✅ Test Response: {response[:150]}...")
            
            # Health check
            health = ai.check_health()
            print(f"🏥 Health Check: {health}")
            
            return True
            
        else:
            print("❌ Failed to load Granite AI - using fallback mode")
            return False
            
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        print(f"🔧 Stack trace: {traceback.format_exc()}")
        return False

if __name__ == "_main_":
    # Run tests
    success = test_granite_model(timeout=120)  # 2 minute timeout for testing
    
    if success:
        print("\n🎉 All tests passed! Your AI models are ready.")
    else:
        print("\n⚠  Granite AI failed, but fallback mode is available.")
        print("💡 The app will work fine in fallback mode with rule-based responses.")