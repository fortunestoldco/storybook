import gradio as gr
import asyncio
import threading
import time
import traceback
import queue
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
from uuid import uuid4
from pymongo import MongoClient
import re
import os
import warnings
import logging

from utils import init_cuda, cleanup_memory, get_memory_usage, split_manuscript, validate_state
from state import AgentState
from config import MODEL_CHOICES, DEFAULT_MODEL_CONFIG, AGENT_TEAMS, ALL_AGENTS, MONGODB_URI, create_agent_model_configs
from agent import AgentFactory
from graph import create_phase_graph, create_main_graph, create_storybook_graph
from storybook_class import storybook

# Configure logging
logger = logging.getLogger(__name__)

def visualize_storybook_graph(graph):
    """Create a visual representation of the storybook graph."""
    G = nx.DiGraph()

    # Add nodes
    for node in graph.nodes:
        G.add_node(node)

    # Add edges
    for node, edges in graph.edges.items():
        for edge in edges:
            G.add_edge(node, edge)

    # Set up the plot
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G, k=2, iterations=50)

    # Group nodes by type
    node_colors = []
    for node in G.nodes():
        if node == "executive_director":
            color = "red"  # Executive Director
        elif "_director" in node:
            color = "orange"  # Other directors
        elif "research" in node:
            color = "blue"  # Research nodes
        elif node == "END":
            color = "black"  # End node
        else:
            color = "green"  # Specialists
        node_colors.append(color)

    # Draw the graph with better aesthetics
    nx.draw(G, pos, with_labels=True,
            node_color=node_colors,
            node_size=2000,
            font_size=8,
            font_weight='bold',
            arrows=True,
            edge_color='gray',
            arrowsize=20,
            font_color='white',
            edgecolors='black',
            linewidths=1.5)

    plt.title("Storybook Agent Workflow Graph", fontsize=24)
    return plt

class writer_gui:
    def __init__(self, storybook_instance):
        # Initialize model choices first
        self.model_choices = MODEL_CHOICES

        # Then initialize other attributes
        self.storybook = storybook_instance
        self.partial_message = ""
        self.thread_id = -1
        self.thread = {"configurable": {"thread_id": str(self.thread_id)}}
        self.is_running = False
        self.agent_visits = {}
        self.project_id = None

        self.default_model_config = DEFAULT_MODEL_CONFIG

        # Create interface last
        self.demo = self.create_interface()

    async def stream_response(self, message, word_delay=0.01):
        """Stream a response word by word with a delay"""
        words = message.split()
        partial = ""
        for word in words:
            partial += word + " "
            yield partial
            await asyncio.sleep(word_delay)

    async def run_all_phases(self, title, synopsis, manuscript, notes_text, task,
                       model_id, model_task, temperature, max_new_tokens,
                       repetition_penalty, do_sample, agent_model_configs=None, max_iterations_per_phase=5):
        """Run all phases of the storybook workflow in sequence"""
        if self.is_running:
            yield "Already running a session. Please wait for it to complete.", "", "", "", "", []
            return

        self.is_running = True
        stream_buffer = []
        phases = ["initialization", "development", "creation", "refinement", "finalization"]

        try:
            # Create model configuration
            main_model_config = {
                "model_id": model_id,
                "task": model_task,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "repetition_penalty": repetition_penalty,
                "do_sample": do_sample
            }

            # Include agent-specific model configs if provided
            if agent_model_configs and len(agent_model_configs) > 0:
                model_config = {
                    **main_model_config,
                    "agent_configs": agent_model_configs
                }
            else:
                model_config = main_model_config

            # Update storybook with model config
            try:
                self.storybook.update_model_config(model_config)
                stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Model configuration updated successfully")
                if agent_model_configs:
                    stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Agent-specific models configured for {len(agent_model_configs)} agents")
            except Exception as e:
                error_msg = f"Error updating model configuration: {str(e)}"
                stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] {error_msg}")
                logger.error(error_msg)

            # Parse notes
            notes = {}
            if notes_text:
                for line in notes_text.strip().split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        notes[key.strip()] = value.strip()

            # Initialize project
            stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Initializing new project for all phases...")
            initial_state = self.storybook.initialize_storybook_project(title, synopsis, manuscript, notes)
            self.project_id = initial_state["project"]["id"]

            # Calculate manuscript statistics
            stats = self.storybook.get_manuscript_statistics(initial_state)
            stats_text = ", ".join(f"{k}: {v}" for k, v in stats.items() if k != "error")
            stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Manuscript statistics: {stats_text}")

            stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Initialized project with ID: {self.project_id}")
            stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Using default model: {model_id}")
            self.partial_message = "\n".join(stream_buffer)
            yield self.partial_message, "initialization", "initialization", self.project_id, "", stream_buffer

            # Set initial task - focus on improving the manuscript
            if task:
                initial_state["current_input"]["task"] = task
            else:
                initial_state["current_input"]["task"] = "Review and improve the manuscript to transform it from draft to professional quality"

            # Start with state from initialization
            state = initial_state

            # Run each phase in sequence
            for phase_idx, phase in enumerate(phases):
                stream_buffer.append(f"\n[{datetime.now().strftime('%H:%M:%S')}] --- Starting Phase: {phase.upper()} ({phase_idx + 1}/{len(phases)}) ---")
                self.partial_message = "\n".join(stream_buffer)
                yield self.partial_message, phase, phase, self.project_id, "", stream_buffer

                # Update state for new phase
                state["phase"] = phase
                state["current_input"]["phase"] = phase

                # Set phase-specific task
                if phase == "initialization":
                    state["current_input"]["task"] = "Analyze manuscript and identify key areas for improvement"
                elif phase == "development":
                    state["current_input"]["task"] = "Develop story structure, characters, and world-building elements"
                elif phase == "creation":
                    state["current_input"]["task"] = "Rewrite and improve content to elevate writing quality"
                elif phase == "refinement":
                    state["current_input"]["task"] = "Polish prose and dialogue to professional quality"
                elif phase == "finalization":
                    state["current_input"]["task"] = "Finalize manuscript and prepare for publication"

                # Run this phase for multiple iterations
                for iteration in range(max_iterations_per_phase):
                    stream_buffer.append(f"\n[{datetime.now().strftime('%H:%M:%S')}] --- {phase.upper()} Phase - Iteration {iteration + 1}/{max_iterations_per_phase} ---")
                    self.partial_message = "\n".join(stream_buffer)
                    yield self.partial_message, state.get('lnode', phase), phase, self.project_id, "", stream_buffer

                    # Create a queue for progress updates
                    progress_queue = queue.Queue()
                    result_container = [None]
                    error_container = [None]

                    # Function to run in a separate thread
                    def run_in_thread():
                        try:
                            progress_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] Starting {phase} phase processing...")

                            # Define progress callback
                            def progress_cb(message):
                                progress_queue.put(f"{message}")

                            # Set config with checkpoint info
                            config = {
                                "configurable": {
                                    "thread_id": f"{self.project_id}_{phase}_{int(time.time())}",
                                    "checkpoint_id": self.project_id,
                                    "checkpoint_ns": phase,
                                }
                            }

                            # Run the phase with the config
                            result = self.storybook.run_storybook_phase(state, phase, progress_cb, config)
                            result_container[0] = result

                            progress_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] Completed {phase} phase processing")
                        except Exception as e:
                            error_container[0] = str(e)
                            progress_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR processing {phase}: {str(e)}")
                            logger.error(f"Error processing {phase}: {str(e)}")
                            logger.debug(traceback.format_exc())

                    # Start thread
                    thread = threading.Thread(target=run_in_thread)
                    thread.start()

                    # Wait for thread with timeout and updates
                    max_wait_time = 1200  # 20 minutes instead of 5
                    start_time = time.time()
                    heartbeat_counter = 0

                    while thread.is_alive():
                        # Check timeout
                        if time.time() - start_time > max_wait_time:
                            error_msg = f"Operation timed out for {phase} phase after 20 minutes"
                            stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: {error_msg}")
                            error_container[0] = error_msg
                            logger.error(error_msg)
                            break

                        # Check for progress
                        messages_added = False
                        while not progress_queue.empty():
                            message = progress_queue.get()
                            stream_buffer.append(message)
                            messages_added = True

                        # Heartbeat - much less frequent now and only if no messages were added
                        heartbeat_counter += 1
                        if heartbeat_counter >= 180 and not messages_added:  # Every 3 minutes if no activity
                            stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Process still active...")
                            heartbeat_counter = 0
                            messages_added = True

                        if messages_added:
                            self.partial_message = "\n".join(stream_buffer)
                            yield self.partial_message, state.get('lnode', phase), phase, self.project_id, "", stream_buffer

                        await asyncio.sleep(0.1)  # Check more frequently - 10 times per second

                        # Force memory cleanup periodically
                        if heartbeat_counter % 60 == 0:  # Every minute
                            memory_stats = cleanup_memory()
                            system_memory = memory_stats["system"]
                            if system_memory["percent_used"] > 90:  # If system memory usage is high
                                stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: High memory usage ({system_memory['percent_used']}%). Performing cleanup.")
                                logger.warning(f"High memory usage ({system_memory['percent_used']}%). Performing cleanup.")

                    # Get remaining updates
                    while not progress_queue.empty():
                        message = progress_queue.get()
                        stream_buffer.append(message)

                    # Check for errors
                    if error_container[0]:
                        stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Error in {phase} phase: {error_container[0]}")
                        self.partial_message = "\n".join(stream_buffer)
                        yield self.partial_message, "error", phase, self.project_id, "", stream_buffer
                        break

                    # Get the result
                    state = result_container[0]

                    if state and hasattr(state, 'get') and state.get("error"):
                        stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Error in {phase} phase: {state['error']}")
                        self.partial_message = "\n".join(stream_buffer)
                        yield self.partial_message, "error", phase, self.project_id, "", stream_buffer
                        break

                    # Extract last message
                    if state.get("messages", []):
                        last_message = state["messages"][-1]
                        agent_name = state.get("lnode", "unknown")
                        content = last_message.get('content', '')
                        content_preview = content[:300]
                        if len(content) > 300:
                            content_preview += "..."

                        stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] [{agent_name}]: {content_preview}")

                    # Update quality assessment
                    quality_assessment = state.get("project", {}).get("quality_assessment", {})
                    quality_str = "Quality Assessment:\n" + "\n".join(
                        [f"- {k}: {v:.2f}" for k, v in quality_assessment.items()]
                    )

                    # Check if completed
                    if state.get("lnode") == "END":
                        stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] {phase.upper()} phase complete. Moving to next phase.")
                        break

                    # Yield current state
                    self.partial_message = "\n".join(stream_buffer)
                    yield self.partial_message, state.get('lnode', phase), phase, self.project_id, quality_str, stream_buffer

                stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Completed {phase.upper()} phase.")

            # All phases complete
            stream_buffer.append(f"\n[{datetime.now().strftime('%H:%M:%S')}] --- All phases completed successfully! ---")
            stream_buffer.append(f"\n[{datetime.now().strftime('%H:%M:%S')}] Your draft manuscript has been transformed into a polished novel!")

            # Calculate final manuscript statistics
            final_stats = self.storybook.get_manuscript_statistics(state)
            final_stats_text = ", ".join(f"{k}: {v}" for k, v in final_stats.items() if k != "error")
            stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Final manuscript statistics: {final_stats_text}")

            # Final quality assessment
            quality_assessment = state.get("project", {}).get("quality_assessment", {})
            quality_str = "Final Quality Assessment:\n" + "\n".join(
                [f"- {k}: {v:.2f}" for k, v in quality_assessment.items()]
            )

            self.partial_message = "\n".join(stream_buffer)
            yield self.partial_message, "complete", "finalization", self.project_id, quality_str, stream_buffer

        except Exception as e:
            error_details = traceback.format_exc()
            stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Unhandled error: {str(e)}")
            stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Error details: {error_details}")
            self.partial_message = "\n".join(stream_buffer)
            logger.error(f"Unhandled error: {str(e)}")
            logger.debug(error_details)
            yield self.partial_message, "error", phases[0], self.project_id, "", stream_buffer

        finally:
            self.is_running = False

    async def run_storybook_with_model(self, title, synopsis, manuscript, notes_text, task, phase,
                                     model_id, model_task, temperature, max_new_tokens,
                                     repetition_penalty, do_sample, agent_model_configs=None, max_iterations=10):
        """Run the storybook workflow with streaming updates and custom model config"""
        if self.is_running:
            yield "Already running a session. Please wait for it to complete.", "", "", "", "", []
            return

        self.is_running = True
        iterations = 0
        state = None
        quality_str = ""
        stream_buffer = []

        try:
            # Create model configuration
            main_model_config = {
                "model_id": model_id,
                "task": model_task,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "repetition_penalty": repetition_penalty,
                "do_sample": do_sample
            }

            # Include agent-specific model configs if provided
            if agent_model_configs and len(agent_model_configs) > 0:
                model_config = {
                    **main_model_config,
                    "agent_configs": agent_model_configs
                }
            else:
                model_config = main_model_config

            # Check for empty manuscript and provide default
            if not manuscript.strip():
                manuscript = "This is a sample manuscript for testing."
                stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: Empty manuscript provided. Using sample text.")
                logger.warning("Empty manuscript provided. Using sample text.")

            # Check for empty title and provide default
            if not title.strip():
                title = "Untitled Project"
                stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: Empty title provided. Using 'Untitled Project'.")
                logger.warning("Empty title provided. Using 'Untitled Project'.")

            # Check for empty synopsis and provide default
            if not synopsis.strip():
                synopsis = "No synopsis provided."
                stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: Empty synopsis provided. Using default.")
                logger.warning("Empty synopsis provided. Using default.")

            # Update storybook with model config - with error handling
            try:
                self.storybook.update_model_config(model_config)
                stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Model configuration updated successfully")
                if agent_model_configs:
                    stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Agent-specific models configured for {len(agent_model_configs)} agents")
            except Exception as e:
                error_msg = f"Error updating model configuration: {str(e)}"
                stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] {error_msg}")
                logger.error(error_msg)
                logger.debug(traceback.format_exc())
                # Continue with existing configuration

            # Parse notes from text input
            notes = {}
            if notes_text:
                for line in notes_text.strip().split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        notes[key.strip()] = value.strip()

            # Initialize a new project
            stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Initializing new project...")
            initial_state = self.storybook.initialize_storybook_project(title, synopsis, manuscript, notes)
            self.project_id = initial_state["project"]["id"]

            # Calculate manuscript statistics
            stats = self.storybook.get_manuscript_statistics(initial_state)
            stats_text = ", ".join(f"{k}: {v}" for k, v in stats.items() if k != "error")
            stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Manuscript statistics: {stats_text}")

            stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Initialized project with ID: {self.project_id}")
            stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Using default model: {model_id}")
            self.partial_message = "\n".join(stream_buffer)
            yield self.partial_message, "initialization", phase, self.project_id, "", stream_buffer

            # Set the phase in the state
            initial_state["phase"] = phase
            initial_state["current_input"]["phase"] = phase

            # Set the task - if not provided, set a default for the phase
            if task:
                initial_state["current_input"]["task"] = task
            else:
                # Set appropriate default task based on phase
                if phase == "initialization":
                    initial_state["current_input"]["task"] = "Analyze manuscript and identify key areas for improvement"
                elif phase == "development":
                    initial_state["current_input"]["task"] = "Develop story structure, characters, and world-building elements"
                elif phase == "creation":
                    initial_state["current_input"]["task"] = "Rewrite and improve content to elevate writing quality"
                elif phase == "refinement":
                    initial_state["current_input"]["task"] = "Polish prose and dialogue to professional quality"
                elif phase == "finalization":
                    initial_state["current_input"]["task"] = "Finalize manuscript and prepare for publication"

            # Run the workflow for the specified phase
            state = initial_state

            while iterations < max_iterations:
                try:
                    # Add iteration header
                    stream_buffer.append(f"\n[{datetime.now().strftime('%H:%M:%S')}] --- Iteration {iterations + 1} in phase '{phase}' ---")
                    self.partial_message = "\n".join(stream_buffer)
                    yield self.partial_message, state.get('lnode', 'starting'), phase, self.project_id, "", stream_buffer

                    # Setup for agent processing
                    agent_name = state.get("lnode", "starting")
                    stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Working with agent: {agent_name}")
                    self.partial_message = "\n".join(stream_buffer)
                    yield self.partial_message, agent_name, phase, self.project_id, "", stream_buffer

                    # Create a thread-safe queue for progress updates
                    progress_queue = queue.Queue()
                    result_container = [None]
                    error_container = [None]

                    # Function to process in a background thread
                    def run_in_thread():
                        try:
                            # Log start
                            progress_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] Started processing with agent: {agent_name}")

                            # Define progress callback for real-time updates
                            def progress_cb(message):
                                progress_queue.put(f"{message}")

                            # Set config
                            config = {
                                "configurable": {
                                    "thread_id": f"{self.project_id}_{phase}_{int(time.time())}",
                                    "checkpoint_id": self.project_id,
                                    "checkpoint_ns": phase,
                                },
                            }

                            # Run the phase with progress updates and config
                            result = self.storybook.run_storybook_phase(state, phase, progress_cb, config)

                            # Store the result
                            result_container[0] = result

                            # Log completion
                            progress_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] Completed processing with agent: {agent_name}")
                        except Exception as e:
                            error_details = traceback.format_exc()
                            error_container[0] = str(e)
                            progress_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR during processing: {str(e)}")
                            progress_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] Error details: {error_details}")
                            logger.error(f"Error during processing with agent {agent_name}: {str(e)}")
                            logger.debug(error_details)

                    # Start the thread
                    thread = threading.Thread(target=run_in_thread)
                    thread.start()

                    # Maximum time to wait for the thread (in seconds)
                    max_wait_time = 1200  # 20 minutes instead of 5 minutes
                    start_time = time.time()

                    # Heartbeat counter for less frequent full updates
                    heartbeat_counter = 0

                    # Add an initial progress message
                    stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Processing started (this may take a while)...")
                    self.partial_message = "\n".join(stream_buffer)
                    yield self.partial_message, agent_name, phase, self.project_id, "", stream_buffer

                    # Wait for the thread with timeout and periodic updates
                    while thread.is_alive():
                        # Check for timeout
                        if time.time() - start_time > max_wait_time:
                            error_msg = "Operation timed out after 20 minutes" # Changed from 5 to 20 minutes
                            stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: {error_msg}")
                            error_container[0] = error_msg
                            logger.error(error_msg)
                            break

                        # Check for progress updates
                        messages_added = False
                        while not progress_queue.empty():
                            message = progress_queue.get()
                            stream_buffer.append(message)
                            messages_added = True

                        # Add heartbeat only if no messages have been added in a long time
                        heartbeat_counter += 1
                        if heartbeat_counter >= 300 and not messages_added:  # Every 30 seconds (300 * 0.1s)
                            stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Process still active...")
                            heartbeat_counter = 0
                            messages_added = True

                        # Only yield if we added messages
                        if messages_added:
                            self.partial_message = "\n".join(stream_buffer)
                            yield self.partial_message, agent_name, phase, self.project_id, "", stream_buffer

                        # Check more frequently (10 times per second) for better responsiveness
                        await asyncio.sleep(0.1)

                        # Force memory cleanup periodically
                        if heartbeat_counter % 60 == 0:  # Every minute
                            memory_stats = cleanup_memory()
                            system_memory = memory_stats["system"]
                            if system_memory["percent_used"] > 90:  # If system memory usage is high
                                stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: High memory usage ({system_memory['percent_used']}%). Performing cleanup.")
                                logger.warning(f"High memory usage ({system_memory['percent_used']}%). Performing cleanup.")

                    # Get any remaining progress updates
                    while not progress_queue.empty():
                        message = progress_queue.get()
                        stream_buffer.append(message)

                    # Check for errors
                    if error_container[0] is not None:
                        stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Error occurred: {error_container[0]}")
                        self.partial_message = "\n".join(stream_buffer)
                        yield self.partial_message, "error", phase, self.project_id, "", stream_buffer
                        break

                    # Get the result
                    state = result_container[0]

                    # Handle state with error attribute (set in run_storybook_phase)
                    if state and hasattr(state, 'get') and state.get("error"):
                        stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Error occurred: {state['error']}")
                        self.partial_message = "\n".join(stream_buffer)
                        yield self.partial_message, "error", phase, self.project_id, "", stream_buffer
                        break

                    # Extract the last message for display
                    if state.get("messages", []):
                        last_message = state["messages"][-1]
                        agent_name = state.get("lnode", "unknown")
                        content = last_message.get('content', '')
                        content_preview = content[:300]
                        if len(content) > 300:
                            content_preview += "..."

                        stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] [{agent_name}]: {content_preview}")

                    # Update quality assessment display
                    quality_assessment = state.get("project", {}).get("quality_assessment", {})
                    quality_str = "Quality Assessment:\n" + "\n".join(
                        [f"- {k}: {v:.2f}" for k, v in quality_assessment.items()]
                    )
                    stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Updated quality assessment")

                    # Check if workflow has ended
                    if state.get("lnode") == "END":
                        stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Workflow completed successfully.")
                        break

                    iterations += 1
                    self.partial_message = "\n".join(stream_buffer)
                    yield self.partial_message, state.get('lnode', 'working'), phase, self.project_id, quality_str, stream_buffer

                except Exception as e:
                    error_details = traceback.format_exc()
                    stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Error in iteration: {str(e)}")
                    stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Error details: {error_details}")
                    self.partial_message = "\n".join(stream_buffer)
                    logger.error(f"Error in iteration: {str(e)}")
                    logger.debug(error_details)
                    yield self.partial_message, "error", phase, self.project_id, "", stream_buffer
                    break

            # Calculate final manuscript statistics
            final_stats = self.storybook.get_manuscript_statistics(state)
            final_stats_text = ", ".join(f"{k}: {v}" for k, v in final_stats.items() if k != "error")
            stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Final manuscript statistics: {final_stats_text}")

            stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Workflow finished after {iterations} iterations.")
            self.partial_message = "\n".join(stream_buffer)
            yield self.partial_message, state.get('lnode', 'complete'), phase, self.project_id, quality_str, stream_buffer

        except Exception as e:
            error_details = traceback.format_exc()
            stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Unhandled error: {str(e)}")
            stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Error details: {error_details}")
            self.partial_message = "\n".join(stream_buffer)
            logger.error(f"Unhandled error: {str(e)}")
            logger.debug(error_details)
            yield self.partial_message, "error", phase, self.project_id, "", stream_buffer

        finally:
            self.is_running = False

    def get_checkpoints(self):
        """Get available checkpoints for loading with enhanced error handling"""
        try:
            # First check if MongoDB client is available
            if not self.storybook.mongo_client:
                logger.warning("MongoDB client not available. Setting up connection...")
                try:
                    # Try to reconnect
                    mongo_uri = MONGODB_URI
                    self.storybook.mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
                    # Test connection
                    self.storybook.mongo_client.admin.command('ping')
                    logger.info("MongoDB connection successful")
                except Exception as conn_err:
                    logger.error(f"Could not connect to MongoDB: {str(conn_err)}")
                    return ["Error: MongoDB connection failed"], []

            # Get checkpoints with detailed logging
            logger.info("Retrieving checkpoints from MongoDB...")
            checkpoints = self.storybook.get_available_checkpoints()
            logger.info(f"Retrieved {len(checkpoints)} checkpoints")

            if not checkpoints:
                # If no checkpoints, return an informative message
                return ["No checkpoints available - run a project first"], []

            checkpoint_list = [f"{cp['project_id']} - {cp['phase']} ({cp['last_modified']})" for cp in checkpoints]
            checkpoint_ids = [cp["checkpoint_id"] for cp in checkpoints]

            logger.info(f"Processed {len(checkpoint_list)} checkpoints")
            return checkpoint_list, checkpoint_ids
        except Exception as e:
            error_msg = f"Error getting checkpoints: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            return [f"Error: {error_msg}"], []

    def get_checkpoint_id_safely(self, selected_value, checkpoint_ids):
        """Safely get a checkpoint ID from various input types."""
        if selected_value is None:
            return None

        # Handle list input
        if isinstance(selected_value, list) and selected_value:
            selected_value = selected_value[0]  # Take the first element

        # Try to use it as an index
        try:
            if isinstance(selected_value, (int, float)) or (isinstance(selected_value, str) and selected_value.isdigit()):
                idx = int(selected_value)
                if 0 <= idx < len(checkpoint_ids):
                    return checkpoint_ids[idx]
        except (ValueError, IndexError):
            pass

        # If that fails, try to match by string
        if isinstance(selected_value, str) and checkpoint_ids:
            # Try to find an ID that contains the selected value
            for checkpoint_id in checkpoint_ids:
                if isinstance(checkpoint_id, str) and selected_value in checkpoint_id:
                    return checkpoint_id

        return None

    def load_project_from_checkpoint(self, checkpoint_id):
        """Load a project from a checkpoint ID"""
        if not checkpoint_id:
            return "No checkpoint selected", "", "", "", "", []

        try:
            logger.info(f"Loading checkpoint: {checkpoint_id}")
            state = self.storybook.load_checkpoint(checkpoint_id)

            # Extract project details
            project = state.get("project", {})
            self.project_id = project.get("id", "unknown")
            phase = state.get("phase", "unknown")

            # Calculate manuscript statistics
            stats = self.storybook.get_manuscript_statistics(state)
            stats_text = ", ".join(f"{k}: {v}" for k, v in stats.items() if k != "error")

            # Prepare status message
            stream_buffer = [
                f"[{datetime.now().strftime('%H:%M:%S')}] Loaded project {self.project_id} from checkpoint",
                f"[{datetime.now().strftime('%H:%M:%S')}] Title: {project.get('title', 'Untitled')}",
                f"[{datetime.now().strftime('%H:%M:%S')}] Phase: {phase}",
                f"[{datetime.now().strftime('%H:%M:%S')}] Last agent: {state.get('lnode', 'unknown')}",
                f"[{datetime.now().strftime('%H:%M:%S')}] Manuscript statistics: {stats_text}"
            ]

            # Include quality assessment
            quality_assessment = project.get("quality_assessment", {})
            quality_str = "Quality Assessment:\n" + "\n".join(
                [f"- {k}: {v:.2f}" for k, v in quality_assessment.items()]
            )

            # Return the state information to update UI
            message = "\n".join(stream_buffer)
            logger.info(f"Successfully loaded checkpoint: {checkpoint_id}")
            return message, state.get('lnode', 'unknown'), phase, self.project_id, quality_str, stream_buffer, state
        except Exception as e:
            error_message = f"Error loading checkpoint: {str(e)}"
            logger.error(error_message)
            logger.debug(traceback.format_exc())
            return error_message, "", "", "", "", [error_message], None

    async def continue_from_checkpoint(self, state, task, phase, agent_model_configs=None, max_iterations=10):
        """Continue processing from a loaded checkpoint"""
        if self.is_running:
            yield "Already running a session. Please wait for it to complete.", "", "", "", "", []
            return

        if not state:
            yield "No state loaded from checkpoint", "", "", "", "", []
            return

        if not validate_state(state):
            logger.error("Invalid state from checkpoint")
            yield "Invalid state loaded from checkpoint", "", "", "", "", []
            return

        self.is_running = True
        iterations = 0
        quality_str = ""
        stream_buffer = []

        try:
            # If agent-specific model configs are provided, update the model config
            if agent_model_configs and len(agent_model_configs) > 0:
                try:
                    current_config = self.storybook.model_config.copy()
                    model_config = {
                        **current_config,
                        "agent_configs": agent_model_configs
                    }
                    self.storybook.update_model_config(model_config)
                    stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Updated model configuration with agent-specific models")
                except Exception as e:
                    stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Error updating model config: {str(e)}")
                    logger.error(f"Error updating model config: {str(e)}")

            # Update the task and phase if provided
            if task:
                state["current_input"]["task"] = task

            if phase:
                state["phase"] = phase
                state["current_input"]["phase"] = phase

            self.project_id = state["project"]["id"]
            current_phase = state["phase"]

            # Calculate manuscript statistics
            stats = self.storybook.get_manuscript_statistics(state)
            stats_text = ", ".join(f"{k}: {v}" for k, v in stats.items() if k != "error")

            stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Continuing project with ID: {self.project_id}")
            stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Phase: {current_phase}")
            stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Task: {state['current_input']['task']}")
            stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Manuscript statistics: {stats_text}")
            self.partial_message = "\n".join(stream_buffer)
            yield self.partial_message, state.get('lnode', 'starting'), current_phase, self.project_id, "", stream_buffer

            # Run the workflow for the specified phase
            while iterations < max_iterations:
                try:
                    # Add iteration header
                    stream_buffer.append(f"\n[{datetime.now().strftime('%H:%M:%S')}] --- Iteration {iterations + 1} in phase '{current_phase}' ---")
                    self.partial_message = "\n".join(stream_buffer)
                    yield self.partial_message, state.get('lnode', 'continuing'), current_phase, self.project_id, "", stream_buffer

                    # Setup for agent processing
                    agent_name = state.get("lnode", "continuing")
                    stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Working with agent: {agent_name}")
                    self.partial_message = "\n".join(stream_buffer)
                    yield self.partial_message, agent_name, current_phase, self.project_id, "", stream_buffer

                    # Create a thread-safe queue for progress updates
                    progress_queue = queue.Queue()
                    result_container = [None]
                    error_container = [None]

                    # Function to process in a background thread
                    def run_in_thread():
                        try:
                            # Log start
                            progress_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] Started processing with agent: {agent_name}")

                            # Define progress callback for real-time updates
                            def progress_cb(message):
                                progress_queue.put(f"{message}")

                            # Set config
                            config = {
                                "configurable": {
                                    "thread_id": f"{self.project_id}_{current_phase}_{int(time.time())}",
                                    "checkpoint_id": self.project_id,
                                    "checkpoint_ns": current_phase,
                                }
                            }

                            # Run the phase with progress updates and config
                            result = self.storybook.run_storybook_phase(state, current_phase, progress_cb, config)

                            # Store the result
                            result_container[0] = result

                            # Log completion
                            progress_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] Completed processing with agent: {agent_name}")
                        except Exception as e:
                            error_details = traceback.format_exc()
                            error_container[0] = str(e)
                            progress_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR during processing: {str(e)}")
                            progress_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] Error details: {error_details}")
                            logger.error(f"Error during processing with agent {agent_name}: {str(e)}")
                            logger.debug(error_details)

                    # Start the thread
                    thread = threading.Thread(target=run_in_thread)
                    thread.start()

                    # Maximum time to wait for the thread (in seconds)
                    max_wait_time = 1200  # 20 minutes instead of 5 minutes
                    start_time = time.time()

                    # Heartbeat counter for less frequent full updates
                    heartbeat_counter = 0

                    # Add an initial progress message
                    stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Processing started (this may take a while)...")
                    self.partial_message = "\n".join(stream_buffer)
                    yield self.partial_message, agent_name, current_phase, self.project_id, "", stream_buffer

                    # Wait for the thread with timeout and periodic updates
                    while thread.is_alive():
                        # Check for timeout
                        if time.time() - start_time > max_wait_time:
                            error_msg = "Operation timed out after 20 minutes"  # Changed from 5 to 20
                            stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: {error_msg}")
                            error_container[0] = error_msg
                            logger.error(error_msg)
                            break

                        # Check for progress updates
                        messages_added = False
                        while not progress_queue.empty():
                            message = progress_queue.get()
                            stream_buffer.append(message)
                            messages_added = True

                        # Add heartbeat only if no messages have been added for a while
                        heartbeat_counter += 1
                        if heartbeat_counter >= 300 and not messages_added:  # Every 30 seconds (300 * 0.1s)
                            stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Process still active...")
                            heartbeat_counter = 0
                            messages_added = True

                        # Only yield if we added messages
                        if messages_added:
                            self.partial_message = "\n".join(stream_buffer)
                            yield self.partial_message, agent_name, current_phase, self.project_id, "", stream_buffer

                        # Check for updates more frequently (10 times per second)
                        await asyncio.sleep(0.1)

                        # Force memory cleanup periodically
                        if heartbeat_counter % 60 == 0:  # Every minute
                            memory_stats = cleanup_memory()
                            system_memory = memory_stats["system"]
                            if system_memory["percent_used"] > 90:  # If system memory usage is high
                                stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: High memory usage ({system_memory['percent_used']}%). Performing cleanup.")
                                logger.warning(f"High memory usage ({system_memory['percent_used']}%). Performing cleanup.")

                    # Get any remaining progress updates
                    while not progress_queue.empty():
                        message = progress_queue.get()
                        stream_buffer.append(message)

                    # Check for errors
                    if error_container[0] is not None:
                        stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Error occurred: {error_container[0]}")
                        self.partial_message = "\n".join(stream_buffer)
                        yield self.partial_message, "error", current_phase, self.project_id, "", stream_buffer
                        break

                    # Get the result
                    state = result_container[0]

                    # Handle state with error attribute
                    if state and hasattr(state, 'get') and state.get("error"):
                        stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Error occurred: {state['error']}")
                        self.partial_message = "\n".join(stream_buffer)
                        yield self.partial_message, "error", current_phase, self.project_id, "", stream_buffer
                        break

                    # Extract the last message for display
                    if state.get("messages", []):
                        last_message = state["messages"][-1]
                        agent_name = state.get("lnode", "unknown")
                        content = last_message.get('content', '')
                        content_preview = content[:300]
                        if len(content) > 300:
                            content_preview += "..."

                        stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] [{agent_name}]: {content_preview}")

                    # Update quality assessment display
                    quality_assessment = state.get("project", {}).get("quality_assessment", {})
                    quality_str = "Quality Assessment:\n" + "\n".join(
                        [f"- {k}: {v:.2f}" for k, v in quality_assessment.items()]
                    )
                    stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Updated quality assessment")

                    # Check if workflow has ended
                    if state.get("lnode") == "END":
                        stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Workflow completed successfully.")
                        break

                    iterations += 1
                    self.partial_message = "\n".join(stream_buffer)
                    yield self.partial_message, state.get('lnode', 'working'), current_phase, self.project_id, quality_str, stream_buffer

                except Exception as e:
                    error_details = traceback.format_exc()
                    stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Error in iteration: {str(e)}")
                    stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Error details: {error_details}")
                    self.partial_message = "\n".join(stream_buffer)
                    logger.error(f"Error in iteration: {str(e)}")
                    logger.debug(error_details)
                    yield self.partial_message, "error", current_phase, self.project_id, "", stream_buffer
                    break

            # Calculate final manuscript statistics
            final_stats = self.storybook.get_manuscript_statistics(state)
            final_stats_text = ", ".join(f"{k}: {v}" for k, v in final_stats.items() if k != "error")
            stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Final manuscript statistics: {final_stats_text}")

            stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Workflow finished after {iterations} iterations.")
            self.partial_message = "\n".join(stream_buffer)
            yield self.partial_message, state.get('lnode', 'complete'), current_phase, self.project_id, quality_str, stream_buffer

        except Exception as e:
            error_details = traceback.format_exc()
            stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Unhandled error: {str(e)}")
            stream_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Error details: {error_details}")
            self.partial_message = "\n".join(stream_buffer)
            logger.error(f"Unhandled error: {str(e)}")
            logger.debug(error_details)
            yield self.partial_message, "error", state.get("phase", "unknown"), self.project_id, "", stream_buffer

        finally:
            self.is_running = False

    # Function to export manuscript from current state
    def export_manuscript(self, state, format_type="text"):
        """Export manuscript in various formats"""
        if not state:
            return "No project loaded", None
            
        if not validate_state(state):
            logger.error("Invalid state provided to export_manuscript")
            return "Invalid state - cannot export manuscript", None

        try:
            # Use storybook's export function
            content = self.storybook.export_manuscript(state, format=format_type)

            # Get project title for filename
            title = state.get("project", {}).get("title", "Untitled")
            clean_title = "".join(c if c.isalnum() else "_" for c in title)

            # Generate appropriate filename based on format
            if format_type == "text":
                filename = f"{clean_title}.txt"
                mime = "text/plain"
            elif format_type == "markdown":
                filename = f"{clean_title}.md"
                mime = "text/markdown"
            elif format_type == "html":
                filename = f"{clean_title}.html"
                mime = "text/html"
            else:
                filename = f"{clean_title}.txt"
                mime = "text/plain"

            return content, (filename, mime)
        except Exception as e:
            error_msg = f"Error exporting manuscript: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            return error_msg, None

    # Add functions to load and save prompts from MongoDB
    def load_prompt_from_mongodb(self, agent_name):
        """Load an agent's prompt from MongoDB, with error handling"""
        if not self.storybook.mongo_client:
            logger.warning(f"MongoDB client not available. Using default prompt for {agent_name}")
            return f"MongoDB client not available. Using default prompt for {agent_name}", ""

        try:
            db = self.storybook.mongo_client["storybook"]
            prompts_collection = db["prompts"]

            # Try to find the agent's prompt
            doc = prompts_collection.find_one({"agent_name": agent_name})
            if doc and "prompt_text" in doc:
                logger.info(f"Loaded prompt for {agent_name} from MongoDB")
                return f"Loaded prompt for {agent_name} from MongoDB", doc["prompt_text"]
            else:
                logger.info(f"No prompt found for {agent_name} in MongoDB, using default")
                return f"No prompt found for {agent_name} in MongoDB, using default", ""

        except Exception as e:
            logger.error(f"Error loading prompt from MongoDB: {str(e)}")
            return f"Error loading prompt from MongoDB: {str(e)}", ""

    def save_prompt_to_mongodb(self, agent_name, prompt_text):
        """Save an agent's prompt to MongoDB, with error handling"""
        if not self.storybook.mongo_client:
            logger.warning(f"MongoDB client not available. Cannot save prompt for {agent_name}")
            return f"MongoDB client not available. Cannot save prompt for {agent_name}."

        if not agent_name or not prompt_text.strip():
            logger.warning("Agent name and prompt text are required")
            return "Agent name and prompt text are required."

        try:
            db = self.storybook.mongo_client["storybook"]
            prompts_collection = db["prompts"]

            # Update if exists, insert if not
            result = prompts_collection.update_one(
                {"agent_name": agent_name},
                {"$set": {"prompt_text": prompt_text, "updated_at": datetime.now()}},
                upsert=True
            )

            if result.matched_count > 0:
                logger.info(f"Updated prompt for {agent_name} in MongoDB")
                return f"Updated prompt for {agent_name} in MongoDB"
            else:
                logger.info(f"Inserted new prompt for {agent_name} in MongoDB")
                return f"Inserted new prompt for {agent_name} in MongoDB"

        except Exception as e:
            logger.error(f"Error saving prompt to MongoDB: {str(e)}")
            return f"Error saving prompt to MongoDB: {str(e)}"

    def auto_save_progress(self, state):
        """Automatic progress saving during long operations"""
        if not validate_state(state):
            logger.error("Invalid state provided to auto_save_progress")
            return False
            
        try:
            checkpoint_id = f"auto_{state['project']['id']}_{int(time.time())}"
            self.storybook.save_checkpoint(state, checkpoint_id)
            logger.info(f"Auto-saved checkpoint: {checkpoint_id}")
            return True
        except Exception as e:
            logger.error(f"Auto-save failed: {e}")
            return False

    def create_interface(self):
        theme = gr.themes.Ocean(
            primary_hue="amber",
            secondary_hue="fuchsia",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont('Work Sans'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
        )

        # Define all available agents for model selection
        all_agents = ALL_AGENTS

        # Group agents by team
        agent_teams = AGENT_TEAMS

        # Function for system stats display
        def get_system_stats():
            memory_stats = get_memory_usage()
            system_memory = memory_stats["system"]
            gpu_memory = memory_stats["gpu"]

            stats = [f"System Memory: {system_memory['percent_used']}% used ({system_memory['available'] / 1e9:.1f} GB free)"]

            if gpu_memory:
                for device_id, device_data in gpu_memory.items():
                    if isinstance(device_id, int):  # Skip error key if present
                        used_gb = device_data["allocated"] / 1e9
                        total_gb = device_data["total"] / 1e9
                        percent = (device_data["allocated"] / device_data["total"]) * 100 if device_data["total"] > 0 else 0
                        stats.append(f"GPU {device_id}: {percent:.1f}% used ({used_gb:.1f}/{total_gb:.1f} GB)")

            return "\n".join(stats)

        with gr.Blocks(
            theme=theme,
            analytics_enabled=False
        ) as demo:
            gr.Markdown("# Storybook Writer")

            # First Row - Status Panel
            with gr.Row():
                sb_lnode_bx = gr.Textbox(label="Current Agent", scale=1)
                sb_phase_bx = gr.Textbox(label="Current Phase", scale=1)
                sb_thread_bx = gr.Textbox(label="Project ID", scale=1)
                sb_quality_bx = gr.Textbox(label="Quality Assessment", scale=2)
                sb_system_stats = gr.Textbox(label="System Stats", scale=1, value=get_system_stats())

            # Hidden state for loaded checkpoints
            loaded_state = gr.State(None)

            # Stream output pane that will show real-time updates
            stream_output = gr.State([])

            # Dictionary to store agent model configs
            agent_models = gr.State({})

            with gr.Tabs() as tabs:
                # Project Content Tab (First for emphasis)
                with gr.Tab("1. Manuscript Input"):
                    gr.Markdown("### Enter your draft manuscript and project details")
                    gr.Markdown("The system will transform your draft into a polished novel-quality manuscript using specialized agents.")
                    with gr.Row():
                        with gr.Column():
                            title_bx = gr.Textbox(
                                label="Title (required)",
                                value="The Hidden Quest",
                                lines=1
                            )
                            synopsis_bx = gr.Textbox(
                                label="Synopsis (required)",
                                value="A young adventurer discovers a mysterious artifact that leads to an epic journey through dangerous lands and magical encounters.",
                                lines=3
                            )
                            notes_bx = gr.Textbox(
                                label="Notes (optional - key: value format)",
                                value="target_audience: Young Adult\ngenre: Fantasy\nwriting_style: Descriptive\ntone: Adventure, Mystery",
                                lines=4
                            )

                    gr.Markdown("### Enter your draft manuscript below:")
                    with gr.Row():
                        manuscript_bx = gr.Textbox(
                            label="Manuscript (required)",
                            value="Chapter 1: The Discovery\n\nThe sun was setting over the village of Elmwood as James finished his chores. He wiped sweat from his brow and looked at the distant mountains. Something was calling to him from beyond the fields he knew so well.\n\n\"James! Dinner's ready,\" his mother called from their cottage.\n\nHe sighed and turned back. Another day, same as always. But tomorrow would be different. Tomorrow he would explore the old ruins his friend Thomas had told him about.\n\nAs he ate dinner, he couldn't help but think about what he might find there. Legend said the ruins held powerful artifacts from the ancient kingdom that once ruled these lands.\n\nThe next morning, James woke early and packed his bag with food, water, and a small knife. He snuck out before his parents woke up.\n\nThe ruins were a three-hour walk from the village, deep in the forest that everyone avoided. People said the forest was haunted, but James didn't believe in such nonsense.\n\nWhen he arrived at the ruins, he was surprised by how intact they still were. Stone pillars rose from the ground, covered in vines and moss. There was an entrance that led underground.\n\n\"Well, here goes nothing,\" he said to himself as he lit his torch and descended into the darkness.\n\nThe air was cool and damp. The walls were decorated with strange symbols he couldn't understand. He walked carefully, listening for any sounds that might mean danger.\n\nAfter what seemed like hours, he entered a large chamber. In the center, on a stone pedestal, was a small crystal orb that glowed with an inner light.\n\n\"Wow, what is this?\" he whispered as he approached. When he touched it, visions flooded his mind - distant lands, strange creatures, and a looming darkness threatening to consume everything.\n\nHe pulled his hand back in shock, but knew immediately what he had to do. This artifact was important, and somehow, he was now connected to its purpose.\n\nJames carefully placed the orb in his bag and made his way back to the surface, unaware that his discovery had already set ancient powers in motion.\n\nAs he walked home, the weight of the orb seemed to grow heavier with each step. His simple life was about to change forever.",
                            lines=15
                        )

                    # Add manuscript import/export options
                    with gr.Row():
                        manuscript_file = gr.File(label="Import manuscript from file", file_types=["text", ".txt", ".md"])
                        export_format = gr.Dropdown(
                            label="Export Format",
                            choices=["text", "markdown", "html"],
                            value="text"
                        )
                        export_btn = gr.Button("Export Manuscript")

                    # Main button for full manuscript transformation
                    with gr.Row():
                        transform_btn = gr.Button("Run Editing Workflow (All Phases)", variant="primary", size="lg")
                        memory_cleanup_btn = gr.Button("Clean Memory", size="sm")

                # Task Control Tab
                with gr.Tab("2. Process Control"):
                    with gr.Row():
                        task_bx = gr.Textbox(
                            label="Specific Task (optional)",
                            value="Review the manuscript and improve character development, dialogue, and descriptive elements",
                            scale=2
                        )
                        phase_select = gr.Dropdown(
                            choices=["initialization", "development", "creation", "refinement", "finalization"],
                            value="creation",
                            label="Phase",
                            scale=1
                        )

                    with gr.Row():
                        run_phase_btn = gr.Button("Run Single Phase", variant="secondary")
                        clear_btn = gr.Button("Clear Output")

                    # Add more focused task buttons
                    with gr.Row():
                        gr.Markdown("### Quick Task Buttons")
                    with gr.Row():
                        improve_prose_btn = gr.Button("Improve Prose Quality", variant="secondary", size="sm")
                        enhance_dialogue_btn = gr.Button("Enhance Dialogue", variant="secondary", size="sm")
                        develop_characters_btn = gr.Button("Develop Characters", variant="secondary", size="sm")
                        structure_plot_btn = gr.Button("Structure Plot", variant="secondary", size="sm")

                    # Add agent workflow visualization
                    with gr.Row():
                        show_graph_btn = gr.Button("Show Agent Workflow Graph", variant="secondary", size="sm")

                    gr.Markdown("### Phase Descriptions:")
                    gr.Markdown("""
                    - **Initialization**: Executive Director assesses the manuscript and identifies areas for improvement
                    - **Development**: Story structure, characters, and world-building are enhanced
                    - **Creation**: Chapter Drafters and Dialogue Crafters rewrite sections to improve content
                    - **Refinement**: Prose Enhancement and Dialogue Refinement experts polish the writing
                    - **Finalization**: The manuscript is given final checks and prepared for publication

                    For best results with draft improvement, use the **Creation** and **Refinement** phases.
                    """)

                # Model Configuration Tab
                with gr.Tab("3. Model Configuration"):
                    gr.Markdown("### Default Model Configuration")
                    gr.Markdown("This configuration applies to all agents unless overridden in the Agent-Specific section below.")

                    with gr.Row():
                        model_id = gr.Dropdown(
                            label="Default Model",
                            choices=self.model_choices,
                            value=self.default_model_config["model_id"]
                        )
                        model_task = gr.Dropdown(
                            label="Task",
                            choices=["text-generation", "text2text-generation", "summarization"],
                            value=self.default_model_config["task"]
                        )

                    with gr.Row():
                        temperature = gr.Slider(
                            label="Temperature",
                            minimum=0.0,
                            maximum=1.0,
                            value=self.default_model_config["temperature"],
                            step=0.01
                        )
                        max_new_tokens = gr.Slider(
                            label="Max New Tokens",
                            minimum=10,
                            maximum=1024,
                            value=self.default_model_config["max_new_tokens"],
                            step=1
                        )

                    with gr.Row():
                        repetition_penalty = gr.Slider(
                            label="Repetition Penalty",
                            minimum=1.0,
                            maximum=2.0,
                            value=self.default_model_config["repetition_penalty"],
                            step=0.01
                        )
                        do_sample = gr.Checkbox(
                            label="Do Sample",
                            value=self.default_model_config["do_sample"]
                        )

                    # Add preset configurations for common use cases
                    with gr.Row():
                        gr.Markdown("### Quick Presets")
                    with gr.Row():
                        speed_preset_btn = gr.Button("Speed-Optimized", variant="secondary", size="sm")
                        quality_preset_btn = gr.Button("Quality-Optimized", variant="secondary", size="sm")
                        balanced_preset_btn = gr.Button("Balanced", variant="secondary", size="sm")
                        creative_preset_btn = gr.Button("Creative Mode", variant="secondary", size="sm")

                    gr.Markdown("### Agent-Specific Model Configuration")
                    gr.Markdown("Select different models for specific agents. If not specified, the default model will be used.")

                    # Create lists to store input component references
                    agent_enables = []
                    agent_model_selects = []
                    agent_temperatures = []
                    agent_tokens_list = []

                    # Create nested accordions for each team
                    for team_name, team_agents in agent_teams.items():
                        with gr.Accordion(f"{team_name}", open=False):
                            for agent in team_agents:
                                agent_display = agent.replace("_", " ").title()
                                with gr.Accordion(f"{agent_display}", open=False):
                                    with gr.Row():
                                        enable = gr.Checkbox(label=f"Use Custom Model", value=False)
                                        model = gr.Dropdown(
                                            label="Model",
                                            choices=self.model_choices,
                                            value=self.default_model_config["model_id"],
                                            interactive=True
                                        )
                                        # Store component references
                                        agent_enables.append(enable)
                                        agent_model_selects.append(model)
                                    with gr.Row():
                                        temp = gr.Slider(
                                            label="Temperature",
                                            minimum=0.0,
                                            maximum=1.0,
                                            value=self.default_model_config["temperature"],
                                            step=0.01
                                        )
                                        token = gr.Slider(
                                            label="Max New Tokens",
                                            minimum=10,
                                            maximum=1024,
                                            value=self.default_model_config["max_new_tokens"],
                                            step=1
                                        )
                                        # Store component references
                                        agent_temperatures.append(temp)
                                        agent_tokens_list.append(token)

                    # Function to collect agent model configs
                    def collect_agent_configs(*values):
                        num_agents = len(all_agents)
                        enables = values[:num_agents]
                        models = values[num_agents:num_agents*2]
                        temperatures = values[num_agents*2:num_agents*3]
                        tokens_values = values[num_agents*3:num_agents*4]

                        configs = {}
                        for i, agent in enumerate(all_agents):
                            if enables[i]:  # If this agent's custom model is enabled
                                configs[agent] = {
                                    "model_id": models[i],
                                    "task": "text-generation",
                                    "temperature": temperatures[i],
                                    "max_new_tokens": tokens_values[i],
                                    "do_sample": True if temperatures[i] > 0.01 else False,
                                    "repetition_penalty": 1.03
                                }
                        return configs

                    # Button to apply model configurations
                    apply_models_btn = gr.Button("Apply Model Configuration", variant="secondary")
                    apply_models_btn.click(
                        fn=collect_agent_configs,
                        inputs=agent_enables + agent_model_selects + agent_temperatures + agent_tokens_list,
                        outputs=[agent_models]
                    )

                # Checkpoint Tab
                with gr.Tab("4. Checkpoints"):
                    with gr.Row():
                        refresh_checkpoints_btn = gr.Button("Refresh Checkpoints")
                        checkpoint_list = gr.Dropdown(
                            label="Available Checkpoints",
                            choices=[],
                            value=None,
                            interactive=True,
                            allow_custom_value=True
                        )
                        checkpoint_ids = gr.State([])

                    with gr.Row():
                        load_checkpoint_btn = gr.Button("Load Selected Checkpoint")
                        continue_checkpoint_btn = gr.Button("Continue from Checkpoint")

                # Prompt Editor Tab (New)
                with gr.Tab("5. Prompt Editor"):
                    gr.Markdown("### Edit Agent Prompts")
                    gr.Markdown("Customize the system prompts used by each agent. Changes will be saved to MongoDB.")

                    # Agent selection and prompt editing
                    with gr.Row():
                        prompt_agent_select = gr.Dropdown(
                            label="Select Agent",
                            choices=[agent.replace("_", " ").title() for agent in all_agents],
                            value="Executive Director"
                        )
                        refresh_prompt_btn = gr.Button("Load Prompt", variant="secondary")

                    # Status message for prompt loading/saving
                    prompt_status = gr.Textbox(label="Status", value="Select an agent and click Load Prompt", interactive=False)

                    # Prompt editing area
                    prompt_text = gr.Textbox(
                        label="Agent Prompt",
                        value="",
                        lines=10
                    )

                    # Save button for prompts
                    save_prompt_btn = gr.Button("Save Prompt to MongoDB", variant="primary")

                # Results & Analysis Tab (New)
                with gr.Tab("6. Results & Analysis"):
                    with gr.Row():
                        gr.Markdown("### Manuscript Analysis")

                    with gr.Row():
                        analyze_btn = gr.Button("Analyze Current Manuscript", variant="secondary")

                    with gr.Row():
                        stats_display = gr.JSON(label="Manuscript Statistics")

                    # Side-by-side comparison for before/after
                    with gr.Row():
                        gr.Markdown("### Before/After Comparison")

                    with gr.Row():
                        original_text = gr.Textbox(label="Original Text", lines=10)
                        improved_text = gr.Textbox(label="Improved Text", lines=10)

                    with gr.Row():
                        compare_btn = gr.Button("Compare Sections", variant="secondary")

                    # Visualization of quality metrics
                    with gr.Row():
                        gr.Markdown("### Quality Assessment Over Time")

                    with gr.Row():
                        quality_chart = gr.Plot(label="Quality Metrics")

            # Streaming Output area with real-time updates
            gr.Markdown("### Process Log")
            sb_live = gr.Textbox(label="Output Log", lines=20, autoscroll=True)

            # Visual workflow graph display
            agent_flow_img = gr.Image(label="Agent Workflow Diagram", visible=False)

            # Output area for exported manuscript
            exported_text = gr.Textbox(label="Exported Manuscript", visible=False, lines=20)
            export_file = gr.File(label="Download Exported File", visible=False)

            # Event handlers
            # Refresh checkpoints
            refresh_checkpoints_btn.click(
                fn=self.get_checkpoints,
                inputs=[],
                outputs=[checkpoint_list, checkpoint_ids]
            )

            # Load checkpoint
            load_checkpoint_btn.click(
                fn=lambda checkpoint_list, checkpoint_ids: self.get_checkpoint_id_safely(checkpoint_list, checkpoint_ids),
                inputs=[checkpoint_list, checkpoint_ids],
                outputs=[gr.State(None)]
            ).then(
                fn=lambda checkpoint_id_state: self.load_project_from_checkpoint(checkpoint_id_state),
                inputs=[gr.State(None)],
                outputs=[sb_live, sb_lnode_bx, sb_phase_bx, sb_thread_bx, sb_quality_bx, stream_output, loaded_state]
            )

            # Continue from checkpoint
            continue_checkpoint_btn.click(
                fn=lambda: (
                    gr.update(value="Processing...", interactive=False),
                    gr.update(value="Continuing from checkpoint...")
                ),
                inputs=None,
                outputs=[continue_checkpoint_btn, sb_live]
            ).then(
                fn=self.continue_from_checkpoint,
                inputs=[loaded_state, task_bx, phase_select, agent_models],
                outputs=[sb_live, sb_lnode_bx, sb_phase_bx, sb_thread_bx, sb_quality_bx, stream_output],
                show_progress=True
            ).then(
                fn=lambda: gr.update(value="Continue from Checkpoint", interactive=True),
                inputs=None,
                outputs=continue_checkpoint_btn
            )

            # Run single phase
            run_phase_btn.click(
                fn=lambda: (
                    gr.update(value="Processing...", interactive=False),
                    gr.update(value="Starting single phase workflow...")
                ),
                inputs=None,
                outputs=[run_phase_btn, sb_live]
            ).then(
                fn=self.run_storybook_with_model,
                inputs=[
                    title_bx, synopsis_bx, manuscript_bx, notes_bx, task_bx, phase_select,
                    model_id, model_task, temperature, max_new_tokens, repetition_penalty, do_sample, agent_models
                ],
                outputs=[sb_live, sb_lnode_bx, sb_phase_bx, sb_thread_bx, sb_quality_bx, stream_output],
                show_progress=True
            ).then(
                fn=lambda: gr.update(value="Run Single Phase", variant="secondary", interactive=True),
                inputs=None,
                outputs=run_phase_btn
            )

            transform_btn.click(
                fn=lambda: (
                    gr.update(value="Processing All Phases...", interactive=False),
                    gr.update(value="Starting Manuscript Revision...")
                ),
                inputs=None,
                outputs=[transform_btn, sb_live]
            ).then(
                fn=self.run_all_phases,
                inputs=[
                    title_bx, synopsis_bx, manuscript_bx, notes_bx, task_bx,
                    model_id, model_task, temperature, max_new_tokens, repetition_penalty, do_sample, agent_models
                ],
                outputs=[sb_live, sb_lnode_bx, sb_phase_bx, sb_thread_bx, sb_quality_bx, stream_output],
                show_progress=True
            ).then(
                fn=lambda: gr.update(value="Run Editing Workflow (All Phases)", variant="primary", interactive=True),
                inputs=None,
                outputs=transform_btn
            )

            # Clear output
            clear_btn.click(
                fn=lambda: (
                    "",
                    "",
                    "",
                    "",
                    "",
                    []
                ),
                inputs=[],
                outputs=[sb_live, sb_lnode_bx, sb_phase_bx, sb_thread_bx, sb_quality_bx, stream_output]
            )

            # Export manuscript
            export_btn.click(
                fn=self.export_manuscript,
                inputs=[loaded_state, export_format],
                outputs=[exported_text, export_file]
            ).then(
                fn=lambda: (gr.update(visible=True), gr.update(visible=True)),
                inputs=None,
                outputs=[exported_text, export_file]
            )

            # Analyze manuscript
            analyze_btn.click(
                fn=lambda state: self.storybook.get_manuscript_statistics(state) if state and validate_state(state) else {"error": "No manuscript loaded or invalid state"},
                inputs=[loaded_state],
                outputs=[stats_display]
            )

            # File upload handler for manuscript import
            def import_manuscript_file(file):
                if file is None:
                    return "No file uploaded"
                try:
                    content = file.decode('utf-8')
                    return content
                except Exception as e:
                    logger.error(f"Error importing file: {str(e)}")
                    return f"Error importing file: {str(e)}"

            manuscript_file.change(
                fn=import_manuscript_file,
                inputs=[manuscript_file],
                outputs=[manuscript_bx]
            )

            # Task preset buttons
            improve_prose_btn.click(
                fn=lambda: ("Analyze the manuscript and improve the prose quality, focusing on descriptive language, imagery, and sentence flow", "refinement"),
                inputs=None,
                outputs=[task_bx, phase_select]
            )

            enhance_dialogue_btn.click(
                fn=lambda: ("Review and enhance dialogue to make it more natural, character-specific, and engaging", "creation"),
                inputs=None,
                outputs=[task_bx, phase_select]
            )

            develop_characters_btn.click(
                fn=lambda: ("Analyze character development and improve character depth, motivations, and relationships", "development"),
                inputs=None,
                outputs=[task_bx, phase_select]
            )

            structure_plot_btn.click(
                fn=lambda: ("Review plot structure for pacing, tension, and coherence, and suggest improvements", "development"),
                inputs=None,
                outputs=[task_bx, phase_select]
            )

            # Model config presets
            def update_model_preset(preset_type):
                if preset_type == "speed":
                    return "microsoft/phi-2", "text-generation", 0.1, 256, 1.03, True
                elif preset_type == "quality":
                    return "mistralai/Mixtral-8x7B-Instruct-v0.1", "text-generation", 0.1, 512, 1.03, True
                elif preset_type == "balanced":
                    return "mistralai/Mistral-7B-Instruct-v0.2", "text-generation", 0.1, 384, 1.03, True
                elif preset_type == "creative":
                    return "meta-llama/Llama-2-7b-chat-hf", "text-generation", 0.7, 512, 1.03, True
                else:
                    return "HuggingFaceH4/zephyr-7b-beta", "text-generation", 0.1, 512, 1.03, True

            speed_preset_btn.click(
                fn=lambda: update_model_preset("speed"),
                inputs=None,
                outputs=[model_id, model_task, temperature, max_new_tokens, repetition_penalty, do_sample]
            )

            quality_preset_btn.click(
                fn=lambda: update_model_preset("quality"),
                inputs=None,
                outputs=[model_id, model_task, temperature, max_new_tokens, repetition_penalty, do_sample]
            )

            balanced_preset_btn.click(
                fn=lambda: update_model_preset("balanced"),
                inputs=None,
                outputs=[model_id, model_task, temperature, max_new_tokens, repetition_penalty, do_sample]
            )

            creative_preset_btn.click(
                fn=lambda: update_model_preset("creative"),
                inputs=None,
                outputs=[model_id, model_task, temperature, max_new_tokens, repetition_penalty, do_sample]
            )

            # Display agent workflow graph
            def create_agent_workflow_image():
                try:
                    plt = visualize_storybook_graph(self.storybook.storybook_graph)
                    # Save to a temporary file
                    temp_file = "agent_workflow_graph.png"
                    plt.savefig(temp_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    return temp_file
                except Exception as e:
                    logger.error(f"Error creating workflow graph: {str(e)}")
                    logger.debug(traceback.format_exc())
                    # Return a simple error message as text
                    return None

            show_graph_btn.click(
                fn=create_agent_workflow_image,
                inputs=None,
                outputs=[agent_flow_img]
            ).then(
                fn=lambda: gr.update(visible=True),
                inputs=None,
                outputs=[agent_flow_img]
            )

            # Memory cleanup button
            memory_cleanup_btn.click(
                fn=lambda: f"Memory cleanup performed: {cleanup_memory()['system']['percent_used']}% system memory in use",
                inputs=None,
                outputs=[sb_live]
            ).then(
                fn=get_system_stats,
                inputs=None,
                outputs=[sb_system_stats]
            )

            # Add prompt editor handlers
            def prepare_agent_name_for_lookup(display_name):
                """Convert display name like 'Executive Director' to lookup key 'executive_director'"""
                return display_name.lower().replace(" ", "_")

            # Load prompt button
            refresh_prompt_btn.click(
                fn=lambda agent_display: self.load_prompt_from_mongodb(prepare_agent_name_for_lookup(agent_display)),
                inputs=[prompt_agent_select],
                outputs=[prompt_status, prompt_text]
            )

            # Save prompt button
            save_prompt_btn.click(
                fn=lambda agent_display, prompt_text: self.save_prompt_to_mongodb(
                    prepare_agent_name_for_lookup(agent_display),
                    prompt_text
                ),
                inputs=[prompt_agent_select, prompt_text],
                outputs=[prompt_status]
            )

            # Help Section
            with gr.Accordion("Help & Information", open=False):
                gr.Markdown(
                    "# How to Use Storybook Writer\n\n"
                    "## Quick Start\n"
                    "1. Enter your manuscript in the 'Manuscript Input' tab or import from a text file\n"
                    "2. Click 'Run Editing Workflow' to run all phases or focus on a specific phase\n"
                    "3. Watch as the system improves your draft with specialized agents\n"
                    "4. When complete, export your manuscript in your preferred format\n\n"

                    "## The Editorial Process\n"
                    "1. **Executive Director** reviews your manuscript and identifies areas for improvement\n"
                    "2. **Content specialists** (Chapter Drafters, Dialogue Crafters) rewrite sections\n"
                    "3. **Editorial specialists** (Prose Enhancement, Dialogue Refinement) polish the writing\n"
                    "4. This process repeats until your manuscript reaches professional quality\n\n"

                    "## Advanced Options\n"
                    "- **Process Control**: Run specific phases or focus on particular tasks\n"
                    "- **Model Configuration**: Set different models for each agent type\n"
                    "- **Checkpoints**: Save progress and continue later\n"
                    "- **Prompt Editor**: Customize the system prompts for each agent\n"
                    "- **Results & Analysis**: View statistics and compare before/after improvements\n\n"

                    "## Model Presets\n"
                    "- **Speed-Optimized**: Faster processing with smaller models (best for limited hardware)\n"
                    "- **Quality-Optimized**: Best output quality but slower processing\n"
                    "- **Balanced**: Good compromise between speed and quality\n"
                    "- **Creative Mode**: More innovative and varied outputs but less predictable\n\n"

                    "## Task Presets\n"
                    "- **Improve Prose Quality**: Enhance descriptive language and flow\n"
                    "- **Enhance Dialogue**: Make conversations more natural and character-specific\n"
                    "- **Develop Characters**: Deepen character motivations and relationships\n"
                    "- **Structure Plot**: Improve story structure and pacing\n\n"

                    "## Best Practices\n"
                    "- Provide a complete draft manuscript for best results\n"
                    "- Include notes about genre, audience, and style\n"
                    "- Use checkpoint system to save progress between sessions\n"
                    "- If you encounter memory issues, use the 'Clean Memory' button\n"
                    "- Export your work regularly in your preferred format\n"
                    "- Customize agent prompts to focus on specific aspects of writing\n"
                )

            # Move the checkpoint initialization inside the Blocks context
            demo.load(
                fn=self.get_checkpoints,
                inputs=None,
                outputs=[checkpoint_list, checkpoint_ids]
            )

            # Regular refresh of system stats
            def update_stats_timer():
                while True:
                    time.sleep(30)  # Update every 30 seconds
                    yield get_system_stats()

            demo.load(
                fn=get_system_stats,
                inputs=None,
                outputs=[sb_system_stats]
            )

            # Add Results & Analysis tab handlers
            compare_btn.click(
                fn=compare_sections,
                inputs=[original_text, improved_text],
                outputs=[stats_display]
            )

            # Update quality metrics visualization
            analyze_btn.click(
                fn=lambda state: [
                    generate_quality_chart(state),
                    self.storybook.get_manuscript_statistics(state) if state and validate_state(state) else {"error": "No manuscript loaded or invalid state"}
                ],
                inputs=[loaded_state],
                outputs=[quality_chart, stats_display]
            )

            # Improve system stats updates
            def update_stats():
                while True:
                    try:
                        stats = get_system_stats()
                        yield stats
                    except Exception as e:
                        logger.error(f"Error updating stats: {e}")
                        yield "Error updating system stats"
                    time.sleep(30)

            demo.queue()
            gr.update(every=30, outputs=[sb_system_stats], function=update_stats)

        return demo

    def launch(self):
        # Use queue() for better streaming support
        self.demo.queue().launch(share=False)  # Set share to False to avoid sharing issues

def compare_sections(original_text, improved_text):
    """Compare original and improved text sections."""
    if not original_text or not improved_text:
        return "Please load a manuscript to compare sections."

    try:
        # Calculate basic diff statistics
        original_words = set(original_text.split())
        improved_words = set(improved_text.split())
        words_added = improved_words - original_words
        words_removed = original_words - improved_words

        result = {
            "original": original_text,
            "improved": improved_text,
            "diff_stats": {
                "words_added": len(words_added),
                "words_removed": len(words_removed),
                "length_diff": len(improved_text) - len(original_text),
                "word_count_diff": len(improved_text.split()) - len(original_text.split())
            },
            "summary": {
                "notable_changes": list(words_added)[:10],  # Show first 10 new words
                "removed_words": list(words_removed)[:10]   # Show first 10 removed words
            }
        }
        return result
    except Exception as e:
        logger.error(f"Error comparing sections: {str(e)}")
        return {
            "error": f"Error comparing sections: {str(e)}",
            "original": original_text,
            "improved": improved_text
        }

def generate_quality_chart(state):
    """Generate quality metrics visualization with enhanced features."""
    if not state or not state.get("quality_history"):
        return None

    try:
        quality_history = state["quality_history"]
        metrics = ["clarity", "coherence", "engagement", "style", "overall"]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        # Plot individual metrics
        for metric in metrics:
            values = [q.get(metric, 0) for q in quality_history]
            ax1.plot(values, label=metric, marker='o')

        ax1.set_xlabel("Iterations")
        ax1.set_ylabel("Score")
        ax1.set_title("Quality Metrics Over Time")
        ax1.legend()
        ax1.grid(True)

        # Plot aggregate improvement
        overall_improvement = [sum(q.values())/len(q) for q in quality_history]
        ax2.plot(overall_improvement, color='green', label='Overall Quality', linewidth=2)
        ax2.fill_between(range(len(overall_improvement)), 0, overall_improvement, alpha=0.2)

        ax2.set_xlabel("Iterations")
        ax2.set_ylabel("Average Score")
        ax2.set_title("Overall Quality Progression")
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        return fig
    except Exception as e:
        logger.error(f"Error generating quality chart: {str(e)}")
        logger.debug(traceback.format_exc())
        return None

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('storybook.log')
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Initialize CUDA handling
    init_cuda()

    # Suppress NVML warnings
    warnings.filterwarnings("ignore", ".*Can't initialize NVML.*")
    logger.info("Starting Storybook Writer application")

    try:
        # Create default model config with agent-specific configs
        default_model_config = {
            "model_id": "HuggingFaceH4/zephyr-7b-beta",  # Default model
            "task": "text-generation",
            "temperature": 0.1,
            "max_new_tokens": 512,
            "do_sample": True,  # Changed from False to True
            "repetition_penalty": 1.03,
            "agent_configs": create_agent_model_configs()  # Add agent-specific configs
        }

        # Create the main storybook instance with agent-specific model configs
        sb_instance = storybook(default_model_config)
        logger.info("Storybook instance created successfully")

        # Create the GUI
        app = writer_gui(sb_instance)
        logger.info("GUI initialized, launching application")
        app.launch()
        
    except Exception as e:
        logger.critical(f"Failed to start application: {str(e)}")
        logger.debug(traceback.format_exc())
        print(f"Critical error: {str(e)}")

if __name__ == "__main__":
    main()