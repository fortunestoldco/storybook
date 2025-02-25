# Use Node.js 20 as base image
FROM node:22-slim

# Install Python and required packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Set up Python virtual environment and install langgraph-cli
RUN python3 -m venv .venv && \
    . .venv/bin/activate && \
    pip install langgraph-cli[inmem]

# Set up frontend
WORKDIR /app/frontend
RUN corepack enable && \
    yarn install &&  \
     yarn run build

# Set environment variables
ENV PORT=8080
ENV NODE_ENV=production
ENV PATH="/app/.venv/bin:$PATH"

# Expose port 8080 for Next.js
EXPOSE 8080

# Create start script
RUN echo '#!/bin/bash\n\
source /app/.venv/bin/activate\n\
cd /app && langgraph up & \n\
cd /app/frontend && yarn start -p 8080\n\
wait' > /app/start.sh && \
chmod +x /app/start.sh

# Start both services
CMD ["/app/start.sh"]
