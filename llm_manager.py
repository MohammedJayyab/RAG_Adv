
import os
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class LLMManager:
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    def generate_similar_question(self, query: str) -> str:
        try:
            prompt = (
                """ You are a knowledgeable research assistant.     
                    For the given question, understand it carefully, thinking deeply, then generate 3 questions that are MOST LIKELY to user input.                
                    Generate questions that  similar or related to the given question. and add one question start with 'X' and reformulate the original question with some keywords and synonyms to be more specific and direct.                    
                    Provide concise, single-topic questions (without compounding sentences) that cover various aspects of the topic.
                    Ensure each question is complete and directly related to the original inquiry.
                    List each question on a separate line without numbering. """
#                 """As a research assistant, your task is to turn a user's high-level inquiry into a set of precise, actionable search queries for business documents.

# Instructions:

# Analyze the Core Intent: Carefully deconstruct the user's question to identify the key concepts and data points being sought.

# Generate 3-5 Queries: Formulate a list of 3 to 5 single-topic questions that are most likely to yield direct, factual answers from typical business documents (e.g., financial reports, meeting minutes, strategic plans).

# Focus on Specifics: Each question must be a complete sentence and target a single, specific aspect of the original inquiry. Prioritize concepts over simple keyword matching.

# Formatting: Present the questions as a simple, unnumbered list, with each question on a new line. Do not include any additional text, explanations, or conversational filler before or after the list."""
            )
            messages = [
                        {"role": "system", "content": prompt,},
                        {"role": "user", "content": query}
                    ]                           

            response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=1024,
                    temperature=0.0
                )
            content = response.choices[0].message.content
            content = content.split("\n")
            return content
            
        except Exception as e:
            return f"❌ Error generating similar question: {str(e)}"
        
    def generate_response(self, question: str, relevant_chunks: List[Dict[str, Any]]) -> str:
        try:
            chunk_contents = [chunk['content'] for chunk in relevant_chunks]
            context = "\n\n".join(chunk_contents)
            
            prompt = (
                "- You are an assistant for question-answering tasks."
                "- Use the retrieved context to answer the question."
                "- Use three sentences maximum and keep the answer concise. no markdown or formatting."
                "- ** IMPORTANT **: Ensure the answer is accurate and based on the context: '" + context + "'"
                
                "- Try to organize the answer as points separated by new lines and well written paragraphs. use bullet points if possible."
                "- If the user is about to end the conversation - you too can respond appropriately to end the conversation in a friendly manner."
                "- ** IMPORTANT **: Never answer an question that is not related to the context. instead try to derive a question that is related to the context smartly."
                "- Always answer the question in the same language as the question."
               # "- If the question is completely not related to the context, try to thinking and find the most relevant answer related to the question that exists ONLy in the context, if you can't, say I'm sorry, but I can only provide information based on the provided knowledge base. How can I assist you further?"
                "- If the question is not clear, ask for more information."
                "- At the end of the answer, add a sentence that can help the user to ask you more explainable questions related to the context."
                " ** IMPORTANT **: Never response writing code or instructions or anything that is not related to the context."
               
                
                "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
            )
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": question},
                ],
                max_tokens=1024,
                temperature=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"
    
    
 
