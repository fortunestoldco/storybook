FROM langchain/langgraph-api:3.11

ENV PYTHONPATH=/app/src:$PYTHONPATH

RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt langchain-core>=0.1.17 langchain-openai>=0.0.5 langgraph>=0.0.15 python-dotenv>=1.0.0

# -- Adding local package . --
ADD . /deps/storybook
# -- End of local package . --

# -- Installing all local dependencies --
RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*
# -- End of local dependencies install --
ENV LANGSERVE_GRAPHS='{"storybook": "/deps/storybook/src/storybook/graph.py:graph"}'

WORKDIR /deps/storybook