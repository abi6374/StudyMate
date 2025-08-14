import os
import logging
from typing import Optional, List, Dict, Any
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

logger = logging.getLogger(__name__)

class GeminiClient:
    """Client for Google Gemini AI API integration."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Google AI API key. If not provided, will look for GEMINI_API_KEY env var.
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.model = None
        self.is_configured = False
        
        if self.api_key:
            self._configure_client()
        else:
            logger.warning("No Gemini API key provided. Gemini features will be disabled.")
    
    def _configure_client(self):
        """Configure the Gemini client with API key and safety settings."""
        try:
            genai.configure(api_key=self.api_key)
            
            # Configure safety settings to be more permissive for educational content
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
            
            # Initialize the model
            self.model = genai.GenerativeModel(
                model_name='gemini-1.5-flash',
                safety_settings=safety_settings
            )
            
            self.is_configured = True
            logger.info("✅ Gemini AI client configured successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to configure Gemini client: {str(e)}")
            self.is_configured = False
    
    def is_available(self) -> bool:
        """Check if Gemini AI is available and configured."""
        return self.is_configured and self.model is not None
    
    async def generate_answer(self, question: str, context: str = "", 
                            max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Generate an answer using Gemini AI.
        
        Args:
            question: The user's question
            context: Relevant context from documents
            max_tokens: Maximum tokens in response
            
        Returns:
            Dict containing the response and metadata
        """
        if not self.is_available():
            raise ValueError("Gemini AI is not configured or available")
        
        try:
            # Construct the prompt
            prompt = self._construct_prompt(question, context)
            
            # Generate response
            response = await self._generate_response(prompt)
            
            return {
                "answer": response.text,
                "model": "gemini-1.5-flash",
                "confidence": 0.85,  # Gemini generally provides high-quality responses
                "source": "gemini_ai",
                "token_count": len(response.text.split()) if response.text else 0
            }
            
        except Exception as e:
            logger.error(f"Error generating Gemini response: {str(e)}")
            raise
    
    def _construct_prompt(self, question: str, context: str) -> str:
        """Construct an effective prompt for Gemini."""
        if context.strip():
            prompt = f"""You are StudyMate, an intelligent academic assistant. Based on the provided document context, answer the user's question accurately and helpfully.

DOCUMENT CONTEXT:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
- Provide a clear, accurate answer based primarily on the document context
- If the context doesn't contain enough information, say so clearly
- Be concise but comprehensive
- Use educational language appropriate for students
- If relevant, suggest follow-up questions or related topics
- Cite specific parts of the document when possible

ANSWER:"""
        else:
            prompt = f"""You are StudyMate, an intelligent academic assistant. Answer the user's question helpfully and accurately.

USER QUESTION: {question}

INSTRUCTIONS:
- Provide a clear, educational response
- Be concise but comprehensive
- Use language appropriate for students
- If you need more context to provide a better answer, mention it
- Suggest related topics or follow-up questions when relevant

ANSWER:"""
        
        return prompt
    
    async def _generate_response(self, prompt: str):
        """Generate response from Gemini with error handling."""
        try:
            # Generate content asynchronously
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=1000,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=40
                )
            )
            
            # Check if response was blocked
            if response.prompt_feedback.block_reason:
                raise ValueError(f"Response blocked: {response.prompt_feedback.block_reason}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in Gemini API call: {str(e)}")
            raise
    
    def summarize_document(self, text: str, max_length: int = 500) -> str:
        """
        Summarize a document using Gemini AI.
        
        Args:
            text: The document text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Summary of the document
        """
        if not self.is_available():
            return "Gemini AI not available for summarization"
        
        try:
            prompt = f"""Please provide a concise summary of the following document. Focus on the main topics, key points, and important information that would be useful for students studying this material.

DOCUMENT TEXT:
{text[:8000]}  # Limit input to avoid token limits

Please provide a summary in approximately {max_length} words that captures the essential information."""

            response = self.model.generate_content(prompt)
            return response.text if response.text else "Unable to generate summary"
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return f"Error generating summary: {str(e)}"
    
    def generate_study_questions(self, text: str, num_questions: int = 5) -> List[str]:
        """
        Generate study questions based on document content.
        
        Args:
            text: The document text
            num_questions: Number of questions to generate
            
        Returns:
            List of study questions
        """
        if not self.is_available():
            return ["Gemini AI not available for question generation"]
        
        try:
            prompt = f"""Based on the following document content, generate {num_questions} thoughtful study questions that would help students understand and review the material. The questions should cover key concepts, important details, and encourage critical thinking.

DOCUMENT CONTENT:
{text[:8000]}

Please provide exactly {num_questions} questions, each on a new line, numbered 1-{num_questions}."""

            response = self.model.generate_content(prompt)
            
            if response.text:
                # Extract questions from response
                lines = response.text.strip().split('\n')
                questions = []
                for line in lines:
                    line = line.strip()
                    if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                        # Clean up the question
                        question = line.split('.', 1)[-1].strip() if '.' in line else line
                        question = question.replace('-', '').replace('•', '').strip()
                        if question:
                            questions.append(question)
                
                return questions[:num_questions] if questions else ["Unable to generate questions"]
            
            return ["Unable to generate questions"]
            
        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            return [f"Error generating questions: {str(e)}"]
