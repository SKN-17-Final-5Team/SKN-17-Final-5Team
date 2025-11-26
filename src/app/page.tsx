'use client'

import { useRef, useState } from 'react'
import { MessageSquare, Save, Download, Loader2 } from 'lucide-react'
import ContractEditor, { ContractEditorRef } from '@/components/editor/ContractEditor'
import AISidebar from '@/components/chat/AISidebar'
import '@/components/editor/editor.css'

export default function Home() {
  const editorRef = useRef<ContractEditorRef>(null)
  const [isSidebarOpen, setIsSidebarOpen] = useState(true)
  const [lastSaved, setLastSaved] = useState<Date | null>(null)
  const [isExporting, setIsExporting] = useState(false)

  const handleSave = () => {
    const content = editorRef.current?.getContent()
    if (content) {
      console.log('Saving document...', content)
      setLastSaved(new Date())
    }
  }

  const handleExport = async () => {
    const content = editorRef.current?.getContent()
    if (!content) return

    setIsExporting(true)

    try {
      // Dynamic import for html2pdf.js (client-side only)
      const html2pdf = (await import('html2pdf.js')).default

      // Create a temporary container for PDF generation
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
        filename: 'Sale_Contract.pdf',
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
        {/* Header - Simplified */}
        <header className="bg-white border-b border-gray-200 px-4 py-2">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-500">
              {lastSaved ? `저장됨: ${lastSaved.toLocaleTimeString('ko-KR')}` : ''}
            </span>
            <div className="flex items-center gap-2">
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
            <ContractEditor ref={editorRef} />
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
