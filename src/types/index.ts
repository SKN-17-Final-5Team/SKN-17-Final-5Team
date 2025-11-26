// Chat message types
export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  isStreaming?: boolean
}

// AI Agent response types
export interface AgentResponse {
  type: 'text' | 'action'
  content: string
  action?: DocumentAction
}

export interface DocumentAction {
  type: 'insert' | 'replace' | 'delete'
  position?: number
  from?: number
  to?: number
  content?: string
}

// Contract field types
export interface ContractField {
  key: string
  label: string
  section: string
  value?: string
}

// API types for Django integration
export interface ChatRequest {
  message: string
  documentContent: string
  conversationId?: string
}

export interface StreamingChunk {
  type: 'content' | 'action' | 'done' | 'error'
  data: string
}

// Editor state
export interface EditorState {
  content: string
  isDirty: boolean
  lastSaved?: Date
}
