PG 10
!pip install langchain cohere wikipedia-api pydantic
!pip install langchain_community
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
import wikipediaapi, getpass
api_key = getpass.getpass('Cohere API Key: ')
llm = Cohere(cohere_api_key=api_key, model="command")
wiki = wikipediaapi.Wikipedia(user_agent="IPCBot", language='en')
ipc_summary = wiki.page("Indian_Penal_Code").text[:3000]
prompt = PromptTemplate(
    input_variables=["ipc", "q"],
    template="You are a legal assistant. Refer to:\n{ipc}\n\nQuestion: {q}\nAnswer in detail with section if known."
)
print("Ask about IPC (type 'exit' to quit):")
while True:
    ques = input("You: ")
    if ques.lower() == "exit": break
    response = llm.predict(prompt.format(ipc=ipc_summary, q=ques))
    print("Bot:", response, "\n")
