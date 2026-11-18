from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()
llm = ChatNVIDIA(model="nvidia/nemotron-3-ultra-550b-a55b")
response=llm.invoke("Explain AgenticAI in one simple line")
print(response.content)



