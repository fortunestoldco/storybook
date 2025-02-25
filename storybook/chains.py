from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import MessagesPlaceholder

# --- Retriever ---

def get_retriever():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's questions based on the below context:\n\n{context}",
            ),
            MessagesPlaceholder(variable_name="input"),
        ]
    )

    retrieval_embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(
        [
            "The main character is a young wizard named Elara.",
            "Elara discovers a hidden portal to a magical realm.",
            "The magical realm is threatened by a dark force.",
            "Elara embarks on a quest to save the magical realm.",
            "She is joined by a wise-talking owl and a mischievous sprite.",
            "They face challenges and overcome obstacles together.",
            "Elara learns to master her magical powers.",
            "The dark force is eventually defeated, restoring balance.",
        ],
        embedding=retrieval_embeddings,
    )
    retriever = vectorstore.as_retriever()

    document_chain = create_stuff_documents_chain(ChatOpenAI(model="gpt-4-turbo-preview"), prompt)

    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain


retriever = get_retriever()


# --- Story Generation ---

def get_story_llm(model: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a world-class children's story writer.\n\n"
                "Respond with a short story for kids based on the following instruction: {instruction}",
            ),
            MessagesPlaceholder(variable_name="input"),
        ]
    )
    return prompt | ChatOpenAI(model=model, temperature=0)


def get_contextual_story_llm(model: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a world-class children's story writer.\n\n"
                "Respond with a short story for kids based on the following instruction: {input}\n\n"
                "Here is some context to help you:\n{context}",
            ),
            MessagesPlaceholder(variable_name="input"),
        ]
    )
    return prompt | ChatOpenAI(model=model)


def get_continue_story_llm(model: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a world-class children's story writer.\n\n"
                "Continue the story for kids based on the following instruction: {input}\n\n"
                "Here is the story so far:\n{story}",
            ),
            MessagesPlaceholder(variable_name="input"),
        ]
    )
    return prompt | ChatOpenAI(model=model)


# --- Translator ---

def get_translator_llm(model: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a translator. Translate the text to english.\n\n" "{text}",
            ),
            MessagesPlaceholder(variable_name="input"),
        ]
    )
    return prompt | ChatOpenAI(model=model)


# --- Narrator ---
def get_narrator_llm(model: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a world-class children's story narrator.\n\n"
                "Respond with a short narrated passage for kids based on the following instruction: {instruction}",
            ),
            MessagesPlaceholder(variable_name="input"),
        ]
    )
    return prompt | ChatOpenAI(model=model)


# --- Chains ---

def llm(model: str):
    return get_contextual_story_llm(model)

def narrator(model: str):
    return get_narrator_llm(model)

def translator(model: str):
    return get_translator_llm(model)

def midjourney(x):
    """hardcoded midjourney prompt generator for now, use return value to invoke midjourney"""
    return f"/imagine {x['text']} --ar 9:16"

# --- fix this ---
def tts(text):
    """hardcoded open ai narrator for now, use return value to invoke eleven labs"""
    from openai import OpenAI

    client = OpenAI()
    response = client.audio.speech.create(
        model="tts-1-hd", voice="nova", input=text
    )
    return response.content
