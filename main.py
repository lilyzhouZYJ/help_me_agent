import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
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
        self.faq_data = self._load_faq()
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
        Check if the question can be answered using FAQ data.
        """
        user_question = state["messages"][-1].content

        system_prompt = f"""
You are a helpful assistant that determines if a customer question can be answered using the provided FAQ data.

FAQ Data:
{self.faq_data}

Customer Question: {user_question}

Respond with exactly one word:
- "yes" if the FAQ data contains enough information to answer the question
- "no" if the FAQ data does not contain enough information to answer the question

Only respond with "yes" or "no".
"""

        response = self.llm.invoke([SystemMessage(content=system_prompt)])
        decision = response.content.strip().lower()
        
        # Add decision to state for routing
        state["can_answer"] = decision == "yes"
        return state
    
    def _answer_question(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Answer the question using FAQ data.
        """
        user_question = state["messages"][-1].content
        
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
        
        state["messages"].append(AIMessage(content=response.content))
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
        """Main chat loop"""
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
                state = {"messages": [HumanMessage(content=user_input)], "can_answer": False}
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
