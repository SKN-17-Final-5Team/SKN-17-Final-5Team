"""
Django Backend Example for Contract Editor AI Integration

This file shows how to implement the streaming chat endpoint in your Django backend
that proxies to your LLM Agent service.

Required packages:
- djangorestframework
- django-cors-headers (for CORS)
"""

# =============================================================================
# views.py - Streaming Chat Endpoint
# =============================================================================

import json
from django.http import StreamingHttpResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import requests


class ChatStreamView(APIView):
    """
    Streaming chat endpoint that proxies to your LLM Agent service.

    Frontend calls: POST /api/chat/
    This view forwards to your LLM Agent and streams the response back.
    """

    def post(self, request):
        message = request.data.get('message', '')
        document_content = request.data.get('documentContent', '')
        conversation_id = request.data.get('conversationId')

        # Prepare request for your LLM Agent
        agent_payload = {
            'message': message,
            'document_content': document_content,
            'conversation_id': conversation_id,
            'context': 'contract_editing',  # Tell agent this is for contract editing
        }

        # Stream response from your LLM Agent
        def event_stream():
            try:
                # Call your LLM Agent service (adjust URL as needed)
                agent_url = 'http://localhost:8001/api/agent/chat/'  # Your LLM Agent URL

                with requests.post(
                    agent_url,
                    json=agent_payload,
                    stream=True,
                    timeout=120
                ) as response:
                    response.raise_for_status()

                    for line in response.iter_lines():
                        if line:
                            # Forward the SSE data to frontend
                            yield f"data: {line.decode('utf-8')}\n\n"

                    yield "data: [DONE]\n\n"

            except requests.exceptions.RequestException as e:
                error_data = json.dumps({
                    'type': 'error',
                    'data': str(e)
                })
                yield f"data: {error_data}\n\n"

        response = StreamingHttpResponse(
            event_stream(),
            content_type='text/event-stream'
        )
        response['Cache-Control'] = 'no-cache'
        response['X-Accel-Buffering'] = 'no'  # Disable NGINX buffering
        return response


class ChatSyncView(APIView):
    """
    Non-streaming chat endpoint for simpler use cases.
    """

    def post(self, request):
        message = request.data.get('message', '')
        document_content = request.data.get('documentContent', '')

        # Call your LLM Agent synchronously
        agent_url = 'http://localhost:8001/api/agent/chat/sync/'

        try:
            response = requests.post(
                agent_url,
                json={
                    'message': message,
                    'document_content': document_content,
                },
                timeout=60
            )
            response.raise_for_status()
            return Response(response.json())

        except requests.exceptions.RequestException as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


# =============================================================================
# urls.py
# =============================================================================

"""
from django.urls import path
from .views import ChatStreamView, ChatSyncView

urlpatterns = [
    path('api/chat/', ChatStreamView.as_view(), name='chat-stream'),
    path('api/chat/sync/', ChatSyncView.as_view(), name='chat-sync'),
]
"""


# =============================================================================
# NGINX Configuration (for streaming to work properly)
# =============================================================================

"""
Add to your NGINX configuration:

location /api/chat/ {
    proxy_pass http://django_backend;
    proxy_http_version 1.1;
    proxy_set_header Connection "";
    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 120s;

    # For SSE
    proxy_set_header X-Accel-Buffering no;
}
"""


# =============================================================================
# LLM Agent Service Example (your separate Gunicorn service)
# =============================================================================

"""
This is an example of what your LLM Agent service might look like.
Adjust based on your actual implementation.

# agent_service/views.py

from django.http import StreamingHttpResponse
from rest_framework.views import APIView
import json

class AgentChatView(APIView):
    def post(self, request):
        message = request.data.get('message', '')
        document_content = request.data.get('document_content', '')

        def generate_response():
            # Initialize your LLM agent
            agent = YourLLMAgent()

            # Stream response
            for chunk in agent.stream_response(
                message=message,
                document_context=document_content,
                tools=['document_generation', 'rag_search', 'web_search']
            ):
                # Format chunk based on type
                if chunk.type == 'text':
                    data = json.dumps({
                        'type': 'content',
                        'data': chunk.content
                    })
                elif chunk.type == 'action':
                    data = json.dumps({
                        'type': 'action',
                        'data': json.dumps({
                            'type': chunk.action_type,  # 'insert', 'replace', etc.
                            'content': chunk.action_content
                        })
                    })
                else:
                    continue

                yield f"{data}\n"

        response = StreamingHttpResponse(
            generate_response(),
            content_type='text/plain'
        )
        return response
"""


# =============================================================================
# Document Actions the LLM Agent Can Return
# =============================================================================

"""
Your LLM Agent can return actions to modify the document.
The frontend handles these action types:

1. Insert content at cursor:
   {
       "type": "action",
       "data": {
           "type": "insert",
           "content": "<p>New paragraph to insert</p>"
       }
   }

2. Replace entire document:
   {
       "type": "action",
       "data": {
           "type": "replace",
           "content": "<h1>New Document</h1><p>...</p>"
       }
   }

3. Replace selection/range:
   {
       "type": "action",
       "data": {
           "type": "replace_range",
           "from": 100,
           "to": 200,
           "content": "replacement text"
       }
   }
"""


# =============================================================================
# CORS Settings (settings.py)
# =============================================================================

"""
INSTALLED_APPS = [
    ...
    'corsheaders',
    'rest_framework',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',  # Must be high in the list
    ...
]

# Allow your frontend origin
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",  # Next.js dev server
    "https://your-production-domain.com",
]

# Or for development:
CORS_ALLOW_ALL_ORIGINS = True  # Don't use in production!

# Allow credentials if needed
CORS_ALLOW_CREDENTIALS = True
"""
