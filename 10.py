!pip install langchain cohere wikipedia-api pydantic
!pip install langchain_community

from langchain import PromptTemplate, LLMChain
from langchain.llms import Cohere
from pydantic import BaseModel
import wikipediaapi, getpass
from IPython.display import display
import ipywidgets as widgets
from typing import Optional

api_key = getpass.getpass('Enter Cohere API Key: ')
llm = Cohere(cohere_api_key=api_key, model="command")

def get_ipc_content():
    wiki = wikipediaapi.Wikipedia(user_agent="IPCChatbot/1.0", language='en')
    page = wiki.page("Indian_Penal_Code")
    if not page.exists(): raise Exception("IPC page not found!")
    return page.text[:5000]

ipc_content = get_ipc_content()

class IPCResponse(BaseModel):
    section: Optional[str] = None
    explanation: str

prompt = PromptTemplate(
    input_variables=["ipc_content", "question"],
    template="""You are a legal assistant. Refer to:
{ipc_content}
Question: {question}
Give a detailed answer, mentioning section if applicable."""
)

def get_response(question):
    formatted = prompt.format(ipc_content=ipc_content, question=question)
    reply = llm.predict(formatted)
    if "Section" in reply:
        section = reply.split('Section')[1].split(':')[0].strip()
        explanation = reply.split(':', 1)[-1].strip()
    else:
        section, explanation = None, reply.strip()
    return IPCResponse(section=section, explanation=explanation)

def show_response(resp):
    print(f"Section: {resp.section or 'N/A'}")
    print(f"Explanation: {resp.explanation}")

text_box = widgets.Text(placeholder='Ask about IPC', description='You:')
button = widgets.Button(description='Ask', icon='legal')

def on_click(b):
    try:
        show_response(get_response(text_box.value))
    except Exception as e:
        print(f"Error: {e}")

button.on_click(on_click)
display(text_box, button)
