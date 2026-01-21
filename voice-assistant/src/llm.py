import subprocess
import os
import re

class Phi3LLM:
    def __init__(self, model_path="models/phi3/phi-3-mini-4k-instruct.Q4_K_M.gguf",
                 llama_cpp_path="lib/llama.cpp/main",
                 context_size=2048,
                 n_threads=4):
        self.model_path = model_path
        self.llama_cpp_path = llama_cpp_path
        self.context_size = context_size
        self.n_threads = n_threads
        
        # Verify paths
        if not os.path.exists(self.llama_cpp_path):
            raise FileNotFoundError(f"llama.cpp not found at {self.llama_cpp_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
            
    def generate(self, prompt, max_tokens=256, temperature=0.7):
        """Generate response using llama.cpp"""
        # Format prompt for Phi-3
        formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        
        cmd = [
            self.llama_cpp_path,
            "-m", self.model_path,
            "-p", formatted_prompt,
            "-n", str(max_tokens),
            "-t", str(self.n_threads),
            "-c", str(self.context_size),
            "--temp", str(temperature),
            "--repeat-penalty", "1.1",
            "-s", "42"  # Fixed seed for reproducibility during testing
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # Extract response from output
                response = result.stdout
                
                # Remove the prompt from the beginning
                if formatted_prompt in response:
                    response = response.split(formatted_prompt, 1)[-1]
                
                # Clean up the response
                response = response.strip()
                
                # Remove end token if present
                if "<|end|>" in response:
                    response = response.split("<|end|>")[0]
                    
                # Remove any remaining formatting
                response = re.sub(r'<\|.*?\|>', '', response)
                
                return response.strip()
            else:
                print(f"LLM error: {result.stderr}")
                return "I apologize, but I encountered an error processing your request."
                
        except subprocess.TimeoutExpired:
            return "I'm sorry, the response took too long to generate."