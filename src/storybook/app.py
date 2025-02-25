import os
from typing import Dict

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver

from storybook.agent import agent_node, generate_image_node, generate_story_node
from storybook.chains import (llm, midjourney, narrator, retriever,
                               translator, tts)
from storybook.config_manager import Config, ConfigManager
from storybook.messages import generate_image, generate_story, system_message
from storybook.state import State, User, Story

# Load environment variables
load_dotenv()


# --- Config ---
config_manager = ConfigManager()
config = config_manager.get_config()


# --- Multi-step workflow ---
def create_workflow(config: Config):
    model = config.model
    options = {
        "persist_to": lambda session_id: SqliteSaver.from_conn_string(
            f"{session_id}.db" if session_id else ":memory:"
        )
    }

    if config.mode == "Story":
        return (
            State.graph.compile(
                agent_node(retriever, llm(model))
                | {
                    "story": generate_story_node(narrator(model)),
                    "image": generate_image_node(translator(model), midjourney),
                    "user": lambda x: User(content=x["input"], type="input"),
                },
                options,
            )
            .map()
            .with_types(input_type=State)
        )
    else:
        return (
            State.graph.compile(
                agent_node(retriever, llm(model))
                | {
                    "text": generate_story_node(narrator(model)),
                    "image": generate_image_node(translator(model), midjourney),
                    "audio": lambda x: tts.invoke(x["text"]),
                    "user": lambda x: User(content=x["input"], type="input"),
                },
                options,
            )
            .map()
            .with_types(input_type=State)
        )


# --- State ---
if "config" not in st.session_state:
    st.session_state["config"] = config_manager.get_config()

if "workflow" not in st.session_state:
    st.session_state["workflow"] = create_workflow(config)
else:
    # Check if model, mode or voice have changed, if so, recreate the workflow
    if (
        st.session_state["config"].model != config.model
        or st.session_state["config"].mode != config.mode
    ):
        st.session_state["config"] = config
        st.session_state["workflow"] = create_workflow(config)

# --- UI ---

st.title("Storybook")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    st.session_state["config"].model = st.selectbox(
        "Model", ["gpt-4-turbo-preview", "gpt-3.5-turbo"]
    )
    st.session_state["config"].mode = st.selectbox("Mode", ["Story", "Audiobook"])

    if st.session_state["config"].mode == "Audiobook":
        st.session_state["config"].voice = st.selectbox(
            "Voice", ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        )

# Accept input prompt
prompt = st.chat_input("Once upon a time...", key="input")
if prompt:
    st.session_state.messages.append(User(content=prompt, type="input"))
    with st.spinner("Generating..."):
        workflow = st.session_state["workflow"]
        # Use previously stored session ID, if none exists, generate new id from prompt and timestamp
        config = workflow.get_config()
        session_id = config.configurable.get("session_id")
        if not session_id:
            session_id = f"{prompt}-{int(datetime.now().timestamp())}"
            config = config.with_options(configurable={"session_id": session_id})
        # Call invoke with explicit config
        result = workflow.invoke({"input": prompt, "user": ""}, config)

# Display messages
for msg in st.session_state.messages:
    if msg.type == "system":
        st.write(f"{msg.content}")
    elif msg.type == "image":
        st.image(msg.content)
    elif msg.type == "audio":
        st.audio(msg.content)
    else:
        with st.chat_message(msg.type):
            st.write(msg.content)
