!pip install langchain
!pip install cohere
!pip install langchain-community
!pip install google-auth
!pip install google-auth-oauthlib
!pip install google-auth-httplib2
!pip install google-api-python-client

file_path = 'Student motivation.txt'
with open(file_path, 'r') as file:
    document_text = file.read()

import os
from langchain.llms import Cohere
os.environ["COHERE_API_KEY"] = "o6DdGd0awe4vhcOEBS4r3RJOt0PdNB4iC60lIE40"
llm = Cohere(cohere_api_key=os.getenv("COHERE_API_KEY"))

from langchain.prompts import PromptTemplate
prompt = PromptTemplate(
    input_variables=["document"],
    template="""
    You are a helpful assistant.
    Given the following document, summarize it in **bullet points**:
    ---
    {document}
    ---
    Summary:
    """
)

from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)
response = chain.run(document=document_text)
print(response)
