import { ChatRequest, StreamingChunk } from '@/types'

// Configure your Django backend URL
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

/**
 * Send a chat message to the AI agent and receive streaming response
 * This connects to your Django backend which proxies to your LLM Agent
 */
export async function sendChatMessage(
  request: ChatRequest,
  onChunk: (chunk: StreamingChunk) => void,
  onComplete: () => void,
  onError: (error: Error) => void
): Promise<void> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/chat/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        // Add auth header if needed
        // 'Authorization': `Bearer ${token}`,
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    if (!response.body) {
      throw new Error('No response body')
    }

    // Handle streaming response (SSE)
    const reader = response.body.getReader()
    const decoder = new TextDecoder()

    while (true) {
      const { done, value } = await reader.read()

      if (done) {
        onComplete()
        break
      }

      const text = decoder.decode(value, { stream: true })

      // Parse SSE format: data: {...}\n\n
      const lines = text.split('\n')

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6)

          if (data === '[DONE]') {
            onComplete()
            return
          }

          try {
            const parsed = JSON.parse(data) as StreamingChunk
            onChunk(parsed)
          } catch {
            // If not JSON, treat as plain text content
            onChunk({ type: 'content', data })
          }
        }
      }
    }
  } catch (error) {
    onError(error instanceof Error ? error : new Error('Unknown error'))
  }
}

/**
 * Alternative: Non-streaming chat for simpler implementations
 */
export async function sendChatMessageSync(request: ChatRequest): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/api/chat/sync/`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  })

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`)
  }

  const data = await response.json()
  return data.response
}

/**
 * Save document to Django backend
 */
export async function saveDocument(
  documentId: string,
  content: string
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/documents/${documentId}/`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ content }),
  })

  if (!response.ok) {
    throw new Error(`Failed to save document: ${response.status}`)
  }
}

/**
 * Load document from Django backend
 */
export async function loadDocument(documentId: string): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/api/documents/${documentId}/`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
  })

  if (!response.ok) {
    throw new Error(`Failed to load document: ${response.status}`)
  }

  const data = await response.json()
  return data.content
}
