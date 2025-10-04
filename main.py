import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Any
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from rich.console import Console
from rich.panel import Panel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

console = Console()

class CustomerServiceBot:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1"),
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        self.faq_data = self._load_faq()
        self.reviews_vectorstore = self._setup_reviews_vectorstore()
        self.graph = self._build_graph()
        
    def _load_faq(self) -> str:
        """
        Load FAQ data from faq.md.
        """
        try:
            with open("faq.md", "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return "FAQ data not found. Please make sure faq.md exists."
    
    def _setup_reviews_vectorstore(self):
        """
        Set up vector store for customer reviews.
        """
        try:
            with open("reviews.md", "r", encoding="utf-8") as f:
                reviews_content = f.read()

            # Parse reviews
            reviews = self._parse_reviews(reviews_content)
            
            # Create document for each review
            documents = []
            for review in reviews:
                doc = Document(
                    page_content=review["content"],
                    metadata={
                        "product": review["product"],
                        "rating": review["rating"],
                        "date": review["date"],
                        "review_id": review["review_id"]
                    }
                )
                documents.append(doc)
            
            # Add documents to vector store
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory="./chroma_db"
            )
            
            return vectorstore
            
        except FileNotFoundError:
            console.print("[yellow]Warning: reviews.md not found. Reviews functionality disabled.[/yellow]")
            return None
        except Exception as e:
            console.print(f"[red]Error setting up reviews vectorstore: {str(e)}[/red]")
            return None
    
    def _parse_reviews(self, reviews_content: str) -> List[Dict[str, Any]]:
        """
        Parse reviews from markdown content into individual reviews.
        """
        reviews = []
        sections = reviews_content.split("## Review")
        
        for i, section in enumerate(sections[1:], 1):  # Skip first empty section
            lines = section.strip().split('\n')
            review_data = {
                "review_id": i,
                "product": "",
                "rating": "",
                "date": "",
                "content": ""
            }
            
            current_content = []
            for line in lines:
                line = line.strip()
                if line.startswith("**Product:**"):
                    review_data["product"] = line.replace("**Product:**", "").strip()
                elif line.startswith("**Rating:**"):
                    review_data["rating"] = line.replace("**Rating:**", "").strip()
                elif line.startswith("**Date:**"):
                    review_data["date"] = line.replace("**Date:**", "").strip()
                elif line.startswith("**Review:**"):
                    current_content.append(line.replace("**Review:**", "").strip())
                elif line and not line.startswith("**"):
                    current_content.append(line)
            
            review_data["content"] = " ".join(current_content)
            reviews.append(review_data)
        
        return reviews
    
    def _search_reviews(self, query: str, k: int = 5) -> List[Document]:
        """
        Search reviews using semantic similarity.
        """
        if not self.reviews_vectorstore:
            return []
        
        try:
            results = self.reviews_vectorstore.similarity_search(query, k=k)
            return results
        except Exception as e:
            console.print(f"[red]Error searching reviews: {str(e)}[/red]")
            return []
    
    def _analyze_reviews_for_question(self, question: str) -> str:
        """
        Analyze reviews to answer customer questions using semantic search and analysis.
        """
        if not self.reviews_vectorstore:
            return "No reviews data available."
        
        # Use semantic search to find relevant reviews
        relevant_reviews = self._search_reviews(question, k=10)
        
        if not relevant_reviews:
            return "No relevant reviews found for your question."
        
        # Enhance the search by also looking for similar questions/contexts
        enhanced_query = self._enhance_search_query(question)
        if enhanced_query != question:
            additional_reviews = self._search_reviews(enhanced_query, k=5)
            # Combine and deduplicate
            all_reviews = relevant_reviews + additional_reviews
            seen_content = set()
            unique_reviews = []
            for doc in all_reviews:
                if doc.page_content not in seen_content:
                    seen_content.add(doc.page_content)
                    unique_reviews.append(doc)
            relevant_reviews = unique_reviews[:10]  # Limit to top 10
        
        # Format reviews for analysis with similarity scores if available
        reviews_text = ""
        for i, doc in enumerate(relevant_reviews, 1):
            metadata = doc.metadata
            reviews_text += f"\nReview {i}:\n"
            reviews_text += f"Product: {metadata.get('product', 'Unknown')}\n"
            reviews_text += f"Rating: {metadata.get('rating', 'Unknown')}\n"
            reviews_text += f"Date: {metadata.get('date', 'Unknown')}\n"
            reviews_text += f"Content: {doc.page_content}\n"
            reviews_text += "-" * 50 + "\n"
        
        # Enhanced analysis prompt with more context
        analysis_prompt = f"""
You are a customer service representative analyzing customer reviews to answer questions.

Customer Question: {question}

Relevant Customer Reviews (found using semantic search):
{reviews_text}

Instructions:
- Analyze the provided reviews to answer the customer's question comprehensively
- If asked about sentiment, identify patterns, trends, and overall customer satisfaction
- If asked about ratings, calculate averages, distributions, and highlight notable patterns
- If asked about specific products, focus on reviews for those products and compare with others
- If asked about experiences, summarize common themes and outliers
- Be honest about what the data shows, including limitations
- If there are conflicting opinions, present both sides fairly
- Provide specific examples from the reviews when relevant
- Keep your response helpful, balanced, and data-driven
- If the question is about a product not well-represented in the reviews, mention this limitation

Answer the customer's question based on the review analysis:
"""
        
        response = self.llm.invoke([SystemMessage(content=analysis_prompt)])
        return response.content
    
    def _enhance_search_query(self, question: str) -> str:
        """
        Enhance the search query to find more relevant reviews using LLM.
        """
        enhancement_prompt = f"""
Given this customer question, generate alternative search queries that would help find relevant customer reviews.

Original Question: "{question}"

Generate 1-2 alternative search queries that:
- Use different wording but mean the same thing
- Focus on key concepts that might appear in reviews
- Use synonyms or related terms

Respond with just the alternative queries, separated by newlines.
"""
        
        try:
            response = self.llm.invoke([SystemMessage(content=enhancement_prompt)])
            enhanced_queries = response.content.strip().split('\n')
            # Return the first enhanced query, or original if none generated
            return enhanced_queries[0].strip() if enhanced_queries[0].strip() else question
        except Exception:
            return question  # Return original if enhancement fails
    
    def _send_assistance_email(self, customer_inquiry: str) -> bool:
        """
        Send email to human assistant with the customer's inquiry.
        """
        try:
            # Email configuration
            smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
            smtp_port = int(os.getenv("SMTP_PORT", "587"))
            email_username = os.getenv("EMAIL_USERNAME")
            email_password = os.getenv("EMAIL_PASSWORD")
            assistance_email = os.getenv("ASSISTANCE_EMAIL", "lzhouzyj@gmail.com")
            
            if not all([email_username, email_password]):
                console.print("[red]Email configuration missing. Please check your .env file.[/red]")
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = email_username
            msg['To'] = assistance_email
            msg['Subject'] = f"Customer Service Assistance Request - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            body = f"""
Customer Inquiry Summary:
{customer_inquiry}

This inquiry could not be answered by the automated chatbot and requires human assistance.

Timestamp: {datetime.now().isoformat()}
"""
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(email_username, email_password)
            text = msg.as_string()
            server.sendmail(email_username, assistance_email, text)
            server.quit()
            
            return True
            
        except Exception as e:
            console.print(f"[red]Error sending email: {str(e)}[/red]")
            return False
    
    def _can_answer_question(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if the question can be answered using FAQ data or customer reviews.
        Uses semantic analysis to determine question type and answerability.
        """
        user_question = state["messages"][-1].content
        
        # Semantic classification of question type
        classification_prompt = f"""
You are a customer service question classifier. Analyze the following customer question and determine:

1. Can this question be answered using FAQ data?
2. Is this question asking about customer reviews, ratings, feedback, or experiences?
3. What type of question is this?

Customer Question: "{user_question}"

FAQ Data Available:
{self.faq_data[:500]}...

Respond in exactly this JSON format:
{{
    "faq_can_answer": true/false,
    "is_reviews_question": true/false,
    "question_type": "faq" or "reviews" or "both" or "neither",
    "confidence": 0.0-1.0
}}

Guidelines:
- FAQ questions: policies, procedures, business hours, shipping, returns, contact info
- Reviews questions: asking about customer opinions, ratings, experiences, satisfaction, feedback
- "both": could be answered by either source
- "neither": requires human assistance
- Be conservative with confidence scores
"""
        
        try:
            classification_response = self.llm.invoke([SystemMessage(content=classification_prompt)])
            
            # Parse the JSON response
            import json
            classification = json.loads(classification_response.content.strip())
            
            faq_can_answer = classification.get("faq_can_answer", False)
            is_reviews_question = classification.get("is_reviews_question", False)
            question_type = classification.get("question_type", "neither")
            confidence = classification.get("confidence", 0.0)
            
            # Determine if we can answer and how
            if question_type == "faq" and faq_can_answer:
                state["can_answer"] = True
                state["question_type"] = "faq"
            elif question_type == "reviews" and is_reviews_question and self.reviews_vectorstore:
                state["can_answer"] = True
                state["question_type"] = "reviews"
            elif question_type == "both" and (faq_can_answer or self.reviews_vectorstore):
                # Prioritize FAQ if both can answer
                state["can_answer"] = True
                state["question_type"] = "faq" if faq_can_answer else "reviews"
            else:
                state["can_answer"] = False
                state["question_type"] = "neither"
            
            # Store confidence for debugging/logging
            state["classification_confidence"] = confidence
            
        except (json.JSONDecodeError, Exception) as e:
            # Fallback to simple keyword matching if semantic classification fails
            console.print(f"[yellow]Warning: Semantic classification failed, using fallback: {str(e)}[/yellow]")
            
            reviews_keywords = [
                "review", "reviews", "rating", "ratings", "customer", "customers",
                "feedback", "opinion", "opinions", "experience", "experiences",
                "satisfied", "happy", "unhappy", "disappointed", "love", "hate",
                "recommend", "quality", "service", "product", "feel", "think"
            ]
            
            is_reviews_question = any(keyword in user_question.lower() for keyword in reviews_keywords)
            
            # Simple FAQ check as fallback
            faq_prompt = f"""
FAQ Data:
{self.faq_data}

Customer Question: {user_question}

Can the FAQ data answer this question? Respond with "yes" or "no" only.
"""
            
            faq_response = self.llm.invoke([SystemMessage(content=faq_prompt)])
            faq_can_answer = faq_response.content.strip().lower() == "yes"
            
            # Determine routing
            if faq_can_answer or (is_reviews_question and self.reviews_vectorstore):
                state["can_answer"] = True
                state["question_type"] = "reviews" if is_reviews_question and not faq_can_answer else "faq"
            else:
                state["can_answer"] = False
                state["question_type"] = "neither"
            
            state["classification_confidence"] = 0.5  # Lower confidence for fallback
        
        return state
    
    def _answer_question(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Answer the question using FAQ data or customer reviews.
        """
        user_question = state["messages"][-1].content
        question_type = state.get("question_type", "faq")
        
        if question_type == "reviews":
            # Use reviews analysis
            response_text = self._analyze_reviews_for_question(user_question)
        else:
            # Use FAQ data
            system_prompt = f"""
You are a helpful customer service representative. Answer the customer's question using ONLY the information provided in the FAQ data below.

FAQ Data:
{self.faq_data}

Customer Question: {user_question}

Instructions:
- Provide a clear, helpful answer based on the FAQ data
- If the FAQ doesn't have all the details, be honest about what you can provide
- Keep your response concise and friendly
- Do not make up information not present in the FAQ
"""
            
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_question)
            ])
            response_text = response.content
        
        state["messages"].append(AIMessage(content=response_text))
        return state
    
    def _request_assistance(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Request human assistance for unanswered questions"""
        user_question = state["messages"][-1].content
        
        # Send email to human assistant
        email_sent = self._send_assistance_email(user_question)
        
        if email_sent:
            response_text = """I'm sorry, but I don't have enough information to answer your question. 

I've forwarded your inquiry to our human support team at lzhouzyj@gmail.com, and they will get back to you as soon as possible. 

Is there anything else I can help you with based on our frequently asked questions?"""
        else:
            response_text = """I'm sorry, but I don't have enough information to answer your question and I'm unable to send your request to our support team at the moment.

Please try contacting us directly at lzhouzyj@gmail.com, or let me know if there's anything else I can help you with based on our frequently asked questions."""
        
        state["messages"].append(AIMessage(content=response_text))
        return state
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(dict)
        
        # Add nodes
        workflow.add_node("can_answer_question", self._can_answer_question)
        workflow.add_node("answer_question", self._answer_question)
        workflow.add_node("request_assistance", self._request_assistance)
        
        # Set entry point
        workflow.set_entry_point("can_answer_question")
        
        # Add conditional edges
        def route_decision(state):
            if state.get("can_answer", False):
                return "answer_question"
            else:
                return "request_assistance"
        
        workflow.add_conditional_edges(
            "can_answer_question",
            route_decision,
            {
                "answer_question": "answer_question",
                "request_assistance": "request_assistance"
            }
        )
        
        # Add edges to end
        workflow.add_edge("answer_question", END)
        workflow.add_edge("request_assistance", END)
        
        return workflow.compile()
    
    def chat(self):
        """
        Main chat loop.
        """
        console.print(Panel.fit(
            "[bold blue]Customer Service AI Chatbot[/bold blue]\n"
            "Ask me anything! Type 'quit' to exit.",
            title="Welcome"
        ))
        
        while True:
            try:
                user_input = console.input("\n[bold green]You:[/bold green] ")
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    console.print("\n[bold blue]Thank you for using our customer service! Goodbye![/bold blue]")
                    break
                
                if not user_input.strip():
                    continue
                
                # Process the message through the graph
                state = {"messages": [HumanMessage(content=user_input)], "can_answer": False, "question_type": "faq"}
                result = self.graph.invoke(state)
                
                # Get response + display
                response = result["messages"][-1].content
                console.print(f"\n[bold blue]Bot:[/bold blue] {response}")
                
            except KeyboardInterrupt:
                console.print("\n\n[bold blue]Thank you for using our customer service! Goodbye![/bold blue]")
                break
            except Exception as e:
                console.print(f"\n[red]Error: {str(e)}[/red]")
                console.print("[yellow]Please try again or contact support at lzhouzyj@gmail.com[/yellow]")

def main():
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]Error: OPENAI_API_KEY not found in environment variables.[/red]")
        console.print("Please create a .env file with your OpenAI API key.")
        return
    
    try:
        bot = CustomerServiceBot()
        bot.chat()
    except Exception as e:
        console.print(f"[red]Failed to start chatbot: {str(e)}[/red]")

if __name__ == "__main__":
    main()
