# Customer Service AI Chatbot

A simple customer service AI chatbot built with LangGraph that can answer frequently asked questions and request human assistance when needed.

## Features

- **FAQ Answering**: Answers customer questions using a comprehensive FAQ database
- **Human Assistance**: Automatically forwards complex questions to human support via email
- **CLI Interface**: Clean command-line interface with rich formatting
- **LangGraph Agent**: Built using LangGraph for robust conversation flow

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the project root. You can use the provided template:

```bash
cp env_template.txt .env
```

Then edit the `.env` file with your actual credentials:

```env
# OpenAI Configuration (required)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo  # or gpt-4, gpt-4-turbo, etc.

# Email configuration for human assistance requests (optional)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password_here
ASSISTANCE_EMAIL=lzhouzyj@gmail.com
```

**Required Environment Variables:**
- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_MODEL`: The model to use (default: gpt-3.5-turbo)

**Optional Email Variables:**
- `SMTP_SERVER`: SMTP server for sending emails (default: smtp.gmail.com)
- `SMTP_PORT`: SMTP port (default: 587)
- `EMAIL_USERNAME`: Your email address for sending assistance requests
- `EMAIL_PASSWORD`: Your email password or app password
- `ASSISTANCE_EMAIL`: Email address to send assistance requests to (default: lzhouzyj@gmail.com)

### 3. Email Setup (Optional)

If you want the bot to send assistance requests via email:

1. Use Gmail with an App Password (recommended)
2. Enable 2-factor authentication on your Gmail account
3. Generate an App Password for this application
4. Use the App Password in the `EMAIL_PASSWORD` field

## Usage

Run the chatbot:

```bash
python main.py
```

The bot will start and you can begin chatting. Type `quit` to exit.

### Example Conversation

```
🤖 Welcome
Customer Service AI Chatbot
Ask me anything! Type 'quit' to exit.

You: What are your business hours?

Bot: We are open Monday through Friday from 9:00 AM to 6:00 PM EST, and Saturday from 10:00 AM to 4:00 PM EST. We are closed on Sundays and major holidays.

You: Can you help me with a custom software development project?

Bot: I'm sorry, but I don't have enough information to answer your question. 

I've forwarded your inquiry to our human support team at lzhouzyj@gmail.com, and they will get back to you as soon as possible. 

Is there anything else I can help you with based on our frequently asked questions?
```

## How It Works

1. **Question Analysis**: The bot first determines if it can answer the question using the FAQ data
2. **FAQ Retrieval**: If yes, it provides an answer based on the FAQ content
3. **Human Assistance**: If no, it sends an email to the support team and informs the customer

## File Structure

```
help_me_agent/
├── main.py          # Main application with LangGraph agent
├── faq.md           # Frequently Asked Questions database
├── requirements.txt # Python dependencies
├── env_template.txt # Environment variables template
├── README.md        # This file
└── .env             # Environment variables (create from template)
```

## Customization

### Adding More FAQs

Edit `faq.md` to add more questions and answers. The bot will automatically use the updated content.

### Modifying Email Recipients

Change the `ASSISTANCE_EMAIL` in your `.env` file to redirect assistance requests to a different email address.

### Adjusting AI Behavior

Modify the system prompts in `main.py` to change how the bot responds or determines if it can answer questions.

### Using Different Models

Change the `OPENAI_MODEL` in your `.env` file to use different OpenAI models:
- `gpt-3.5-turbo` (default, faster and cheaper)
- `gpt-4` (more capable but slower and more expensive)
- `gpt-4-turbo` (latest GPT-4 model)

## Requirements

- Python 3.8+
- OpenAI API key
- Internet connection for AI model access
- Email credentials (optional, for assistance requests)

## Troubleshooting

### API Key Issues
- Ensure your `.env` file is in the project root directory
- Verify your OpenAI API key is correct and has sufficient credits
- Check that the API key has the necessary permissions

### Email Issues
- Verify SMTP settings are correct for your email provider
- For Gmail, ensure you're using an App Password, not your regular password
- Check that 2-factor authentication is enabled on your Gmail account

### FAQ Issues
- Ensure `faq.md` exists and contains properly formatted questions and answers
- Check that the file is readable and not corrupted

## License

This project is open source and available under the MIT License.