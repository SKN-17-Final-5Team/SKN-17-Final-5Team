'use client'

import { useRef, useState, useEffect } from 'react'
import { MessageSquare, Save, Download, Loader2, FileText, ChevronDown } from 'lucide-react'
import ContractEditor, { ContractEditorRef } from '@/components/editor/ContractEditor'
import AISidebar from '@/components/chat/AISidebar'
import { documents, DocumentType, getTemplateByType } from '@/templates'
import '@/components/editor/editor.css'

const DOC_CONTENT_KEY = 'contract-editor-documents'
const DOC_TYPE_KEY = 'contract-editor-doc-type'
const SHARED_FIELDS_KEY = 'contract-editor-shared-fields'

// Extract shared field values from HTML content
function extractSharedFields(html: string): Record<string, string> {
  const fields: Record<string, string> = {}
  const regex = /<span[^>]*class="shared-field"[^>]*data-field="([^"]+)"[^>]*>([^<]*)<\/span>/g
  let match
  while ((match = regex.exec(html)) !== null) {
    const fieldName = match[1]
    const fieldValue = match[2]
    // Only store if not a placeholder like [FIELD_NAME]
    if (fieldValue && !fieldValue.startsWith('[')) {
      fields[fieldName] = fieldValue
    }
  }
  return fields
}

// Apply shared field values to HTML content
function applySharedFields(html: string, fields: Record<string, string>): string {
  let result = html
  for (const [fieldName, fieldValue] of Object.entries(fields)) {
    // Replace all instances of this shared field
    const regex = new RegExp(
      `(<span[^>]*class="shared-field"[^>]*data-field="${fieldName}"[^>]*>)([^<]*)(<\\/span>)`,
      'g'
    )
    result = result.replace(regex, `$1${fieldValue}$3`)
  }
  return result
}

// Load shared field values from localStorage
function loadSharedFields(): Record<string, string> {
  if (typeof window === 'undefined') return {}
  try {
    const saved = localStorage.getItem(SHARED_FIELDS_KEY)
    return saved ? JSON.parse(saved) : {}
  } catch {
    return {}
  }
}

// Save shared field values to localStorage
function saveSharedFields(fields: Record<string, string>) {
  if (typeof window === 'undefined') return
  try {
    const existing = loadSharedFields()
    const merged = { ...existing, ...fields }
    localStorage.setItem(SHARED_FIELDS_KEY, JSON.stringify(merged))
  } catch {
    console.error('Failed to save shared fields to localStorage')
  }
}

// Load all saved document contents
function loadSavedDocuments(): Record<string, string> {
  if (typeof window === 'undefined') return {}
  try {
    const saved = localStorage.getItem(DOC_CONTENT_KEY)
    return saved ? JSON.parse(saved) : {}
  } catch {
    return {}
  }
}

// Save document content for a specific document type
function saveDocumentContent(docType: string, content: string) {
  if (typeof window === 'undefined') return
  try {
    const existing = loadSavedDocuments()
    existing[docType] = content
    localStorage.setItem(DOC_CONTENT_KEY, JSON.stringify(existing))
  } catch {
    console.error('Failed to save document to localStorage')
  }
}

// Load saved document type
function loadSavedDocType(): DocumentType {
  if (typeof window === 'undefined') return 'saleContract'
  try {
    const saved = localStorage.getItem(DOC_TYPE_KEY)
    return (saved as DocumentType) || 'saleContract'
  } catch {
    return 'saleContract'
  }
}

// Save current document type
function saveDocType(docType: DocumentType) {
  if (typeof window === 'undefined') return
  localStorage.setItem(DOC_TYPE_KEY, docType)
}

export default function Home() {
  const editorRef = useRef<ContractEditorRef>(null)
  const [isSidebarOpen, setIsSidebarOpen] = useState(true)
  const [lastSaved, setLastSaved] = useState<Date | null>(null)
  const [isExporting, setIsExporting] = useState(false)
  const [selectedDocType, setSelectedDocType] = useState<DocumentType>('saleContract')
  const [isDocSelectorOpen, setIsDocSelectorOpen] = useState(false)
  const [isInitialized, setIsInitialized] = useState(false)
  const isSwitchingDocRef = useRef(false)

  // Load saved state on mount
  useEffect(() => {
    const savedDocType = loadSavedDocType()
    const savedDocs = loadSavedDocuments()
    const sharedFields = loadSharedFields()

    setSelectedDocType(savedDocType)

    // Apply saved content or template
    const applyContent = () => {
      if (!editorRef.current) {
        setTimeout(applyContent, 100)
        return
      }

      // Use saved content if exists, otherwise use template
      let content = savedDocs[savedDocType] || getTemplateByType(savedDocType)

      // Apply shared field values to the content
      content = applySharedFields(content, sharedFields)

      editorRef.current.setContent(content)
      setIsInitialized(true)
    }

    setTimeout(applyContent, 200)
  }, [])

  // Auto-save current document content when it changes
  const handleContentChange = (content: string) => {
    if (!isInitialized || isSwitchingDocRef.current) return
    saveDocumentContent(selectedDocType, content)

    // Extract and save shared field values
    const fields = extractSharedFields(content)
    if (Object.keys(fields).length > 0) {
      saveSharedFields(fields)
    }
  }

  const handleSave = () => {
    const content = editorRef.current?.getContent()
    if (content) {
      saveDocumentContent(selectedDocType, content)

      // Extract and save shared field values
      const fields = extractSharedFields(content)
      if (Object.keys(fields).length > 0) {
        saveSharedFields(fields)
      }

      setLastSaved(new Date())
    }
  }

  const handleDocTypeChange = (docType: DocumentType) => {
    // Prevent auto-save during document switching (use ref for synchronous update)
    isSwitchingDocRef.current = true

    // Save current document content and extract shared fields first
    const currentContent = editorRef.current?.getContent() || ''
    if (currentContent) {
      saveDocumentContent(selectedDocType, currentContent)
      const fields = extractSharedFields(currentContent)
      if (Object.keys(fields).length > 0) {
        saveSharedFields(fields)
      }
    }

    // Load the target document (saved content or template)
    const savedDocs = loadSavedDocuments()
    const sharedFields = loadSharedFields()
    let targetContent = savedDocs[docType] || getTemplateByType(docType)

    // Apply shared field values to the new document
    targetContent = applySharedFields(targetContent, sharedFields)

    editorRef.current?.setContent(targetContent)
    setSelectedDocType(docType)
    setIsDocSelectorOpen(false)
    saveDocType(docType)

    // Re-enable auto-save after state updates
    setTimeout(() => {
      isSwitchingDocRef.current = false
    }, 100)
  }

  const selectedDoc = documents.find(d => d.id === selectedDocType)

  const handleExport = async () => {
    const content = editorRef.current?.getContent()
    if (!content) return

    setIsExporting(true)

    try {
      const html2pdf = (await import('html2pdf.js')).default

      const container = document.createElement('div')
      container.id = 'pdf-export-container'
      container.innerHTML = `
        <div style="font-family: Arial, sans-serif; padding: 30px; width: 700px; background: white;">
          <style>
            #pdf-export-container table { border-collapse: collapse; width: 100%; margin: 12px 0; }
            #pdf-export-container th, #pdf-export-container td { border: 1px solid #666; padding: 8px; text-align: left; font-size: 10px; }
            #pdf-export-container th { background-color: #eee; }
            #pdf-export-container h1 { text-align: center; font-size: 20px; margin-bottom: 20px; }
            #pdf-export-container h2 { font-size: 14px; border-bottom: 1px solid #666; padding-bottom: 6px; margin-top: 24px; }
            #pdf-export-container h3 { font-size: 12px; margin-top: 16px; }
            #pdf-export-container p { font-size: 10px; line-height: 1.6; margin: 6px 0; }
            #pdf-export-container ul, #pdf-export-container ol { font-size: 10px; line-height: 1.6; padding-left: 20px; }
            #pdf-export-container mark { background-color: #ffeb3b; padding: 1px 3px; }
            #pdf-export-container hr { margin: 20px 0; border: none; border-top: 1px solid #666; }
          </style>
          ${content}
        </div>
      `
      document.body.appendChild(container)

      const options = {
        margin: 10,
        filename: `${selectedDoc?.name || 'Document'}.pdf`,
        image: { type: 'jpeg' as const, quality: 0.98 },
        html2canvas: { scale: 2 },
        jsPDF: { unit: 'mm' as const, format: 'a4' as const, orientation: 'portrait' as const },
      }

      await html2pdf().set(options).from(container).save()
      document.body.removeChild(container)
    } catch (error) {
      console.error('PDF export error:', error)
      alert('PDF 내보내기 중 오류가 발생했습니다.')
    } finally {
      setIsExporting(false)
    }
  }

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="bg-white border-b border-gray-200 px-4 py-2">
          <div className="flex items-center justify-between">
            {/* Document Selector */}
            <div className="relative">
              <button
                onClick={() => setIsDocSelectorOpen(!isDocSelectorOpen)}
                className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
              >
                <FileText className="w-4 h-4" />
                {selectedDoc?.nameKo || '서류 선택'}
                <ChevronDown className={`w-4 h-4 transition-transform ${isDocSelectorOpen ? 'rotate-180' : ''}`} />
              </button>

              {isDocSelectorOpen && (
                <div className="absolute top-full left-0 mt-1 w-48 bg-white rounded-lg shadow-lg border border-gray-200 py-1 z-50">
                  {documents.map((doc) => (
                    <button
                      key={doc.id}
                      onClick={() => handleDocTypeChange(doc.id)}
                      className={`w-full text-left px-4 py-2 text-sm hover:bg-gray-100 transition-colors ${
                        selectedDocType === doc.id ? 'bg-blue-50 text-blue-600' : 'text-gray-700'
                      }`}
                    >
                      {doc.nameKo}
                      <span className="text-xs text-gray-400 ml-2">({doc.name})</span>
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Actions */}
            <div className="flex items-center gap-2">
              <span className="text-xs text-gray-400">
                {lastSaved ? `저장됨: ${lastSaved.toLocaleTimeString('ko-KR')}` : ''}
              </span>
              <button
                onClick={handleSave}
                className="flex items-center gap-1.5 px-3 py-1.5 text-sm text-gray-600 hover:bg-gray-100 rounded transition-colors"
              >
                <Save className="w-4 h-4" />
                저장
              </button>
              <button
                onClick={handleExport}
                disabled={isExporting}
                className="flex items-center gap-1.5 px-3 py-1.5 text-sm text-gray-600 hover:bg-gray-100 rounded transition-colors disabled:opacity-50"
              >
                {isExporting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
                {isExporting ? 'PDF 생성 중...' : 'PDF 내보내기'}
              </button>
              <div className="w-px h-5 bg-gray-200 mx-1" />
              <button
                onClick={() => setIsSidebarOpen(!isSidebarOpen)}
                className={`flex items-center gap-1.5 px-3 py-1.5 text-sm rounded transition-colors ${
                  isSidebarOpen
                    ? 'bg-blue-500 text-white'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                <MessageSquare className="w-4 h-4" />
                AI 도우미
              </button>
            </div>
          </div>
        </header>

        {/* Editor */}
        <main className="flex-1 overflow-auto p-4">
          <div className="max-w-5xl mx-auto">
            <ContractEditor ref={editorRef} initialDocType={selectedDocType} onChange={handleContentChange} />
          </div>
        </main>
      </div>

      {/* AI Sidebar */}
      <AISidebar
        editorRef={editorRef}
        isOpen={isSidebarOpen}
        onToggle={() => setIsSidebarOpen(false)}
      />
    </div>
  )
}
