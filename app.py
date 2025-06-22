import gradio as gr
import json
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

# Setup OpenAI/Ollama local
openai = OpenAI(base_url="http://localhost:11434/v1", api_key='ollama')
MODEL = "llama3.2"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

class Website:
    def __init__(self, url):
        self.url = url
        response = requests.get(url, headers=headers)
        self.body = response.content
        soup = BeautifulSoup(self.body, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        if soup.body:
            for irrelevant in soup.body(["script", "style", "img", "input"]):
                irrelevant.decompose()
            self.text = soup.body.get_text(separator="\n", strip=True)
        else:
            self.text = ""
        links = [link.get('href') for link in soup.find_all('a')]
        self.links = [link for link in links if link]

    def get_contents(self):
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"

def get_all_details(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        return f"Error fetching details: {e}"

system_prompt = (
    "You are an assistant that analyzes the contents of several relevant pages from a company website "
    "and creates a short brochure about the company for prospective customers, investors and recruits. "
    "Respond in markdown. Include details of company culture, customers and careers/jobs if you have the information."
)

def get_brochure_user_prompt(company_name, url):
    user_prompt = f"You are looking at a company called: {company_name}\n"
    user_prompt += "Here are the contents of its landing page and other relevant pages:\n"
    user_prompt += get_all_details(url)
    return user_prompt[:5000]  # Truncate if needed

def create_brochure(company_name, url):
    prompt = get_brochure_user_prompt(company_name, url)
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
    )
    result = response.choices[0].message.content
    return result

# Gradio UI
gr.Interface(
    fn=create_brochure,
    inputs=[
        gr.Textbox(label="Company Name"),
        gr.Textbox(label="Website URL")
    ],
    outputs=gr.Markdown(label="AI Brochure"),
    title="📄 AI Website Brochure Generator",
    description="Enter a company name and its website URL to generate a marketing brochure using LLaMA 3 (via Ollama)."
).launch()
