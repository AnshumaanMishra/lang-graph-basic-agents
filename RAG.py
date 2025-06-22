from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage
from operator import add as add_messages
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langgraph.prebuilt import ToolNode
import os

load_dotenv()

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model           =   "gemini-2.5-flash"      ,
    google_api_key  =   api_key                 ,
    temperature     =   0                       ,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model           =   "models/embedding-001"  ,
)

