# In main.py, update the server mounting section
# This makes sure we properly mount all required API endpoints

# Mount LangGraph API with correct prefix - this is critical
app.mount("/api/v1", server.app)

# Add an assistant API directly to the FastAPI app for redundancy
@app.get("/api/v1/assistants")
async def get_assistants():
    """Get available assistants information for UI."""
    assistants = [
        {
            "id": "exec_director",
            "name": "Executive Director",
            "description": "Overall project manager for the novel writing process",
            "model": agent_factory.backend_config.provider
        },
        {
            "id": "creative_director",
            "name": "Creative Director",
            "description": "Manages creative aspects of the novel",
            "model": agent_factory.backend_config.provider
        },
        {
            "id": "content_director",
            "name": "Content Development Director",
            "description": "Manages content creation and drafting",
            "model": agent_factory.backend_config.provider
        },
        {
            "id": "editorial_director",
            "name": "Editorial Director",
            "description": "Manages editing and refinement",
            "model": agent_factory.backend_config.provider
        },
        {
            "id": "market_director",
            "name": "Market Alignment Director",
            "description": "Manages market positioning and audience targeting",
            "model": agent_factory.backend_config.provider
        }
    ]
    return assistants