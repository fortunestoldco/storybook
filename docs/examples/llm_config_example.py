from storybook.config import LLMProvider

# Example agent configurations
research_config = {
    "provider": LLMProvider.ANTHROPIC,
    "config": {
        "model_name": "claude-3-sonnet",
        "temperature": 0.7,
        "max_tokens": 4000
    }
}

writing_config = {
    "provider": LLMProvider.OPENAI,
    "config": {
        "model_name": "gpt-4",
        "temperature": 0.9,
        "max_tokens": 4000
    }
}

editorial_config = {
    "provider": LLMProvider.LLAMACPP,
    "config": {
        "model_path": "./models/llama-2-7b.Q4_K_M.gguf",
        "temperature": 0.3,
        "n_gpu_layers": 1
    }
}

# Example usage with an agent
def example_usage():
    from storybook.agents import ContentAnalyzer
    
    # Initialize with specific configuration
    analyzer = ContentAnalyzer(llm_config=research_config)
    
    # Or switch configuration at runtime
    result = analyzer.analyze_content(
        manuscript_id="123",
        llm_config=writing_config
    )
    
    # Use different configurations for different tasks
    editorial_result = analyzer.analyze_progress(
        manuscript_id="123",
        previous_analysis={},
        stage="final_review",
        llm_config=editorial_config
    )

if __name__ == "__main__":
    example_usage()