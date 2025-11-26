'use client'

import { useEditor, EditorContent } from '@tiptap/react'
import StarterKit from '@tiptap/starter-kit'
import { Table, TableRow, TableCell, TableHeader } from '@tiptap/extension-table'
import Placeholder from '@tiptap/extension-placeholder'
import Underline from '@tiptap/extension-underline'
import TextAlign from '@tiptap/extension-text-align'
import Highlight from '@tiptap/extension-highlight'
import { useCallback, useEffect, forwardRef, useImperativeHandle } from 'react'
import { saleContractTemplateHTML } from '@/templates/saleContract'
import EditorToolbar from './EditorToolbar'

export interface ContractEditorRef {
  getContent: () => string
  getJSON: () => object
  setContent: (content: string) => void
  insertContent: (content: string) => void
  replaceSelection: (content: string) => void
}

interface ContractEditorProps {
  initialContent?: string
  onChange?: (content: string) => void
  className?: string
}

const ContractEditor = forwardRef<ContractEditorRef, ContractEditorProps>(
  ({ initialContent, onChange, className }, ref) => {
    const editor = useEditor({
      extensions: [
        StarterKit.configure({
          heading: {
            levels: [1, 2, 3, 4],
          },
        }),
        Table.configure({
          resizable: true,
          HTMLAttributes: {
            class: 'contract-table',
          },
        }),
        TableRow,
        TableCell,
        TableHeader,
        Placeholder.configure({
          placeholder: 'Start typing or ask AI for help...',
        }),
        Underline,
        TextAlign.configure({
          types: ['heading', 'paragraph'],
        }),
        Highlight.configure({
          multicolor: true,
        }),
      ],
      content: initialContent || saleContractTemplateHTML,
      editorProps: {
        attributes: {
          class: 'prose prose-sm sm:prose lg:prose-lg max-w-none focus:outline-none min-h-[500px] p-4',
        },
      },
      onUpdate: ({ editor }) => {
        onChange?.(editor.getHTML())
      },
      immediatelyRender: false,
    })

    // Expose methods to parent components via ref
    useImperativeHandle(ref, () => ({
      getContent: () => editor?.getHTML() || '',
      getJSON: () => editor?.getJSON() || {},
      setContent: (content: string) => {
        editor?.commands.setContent(content)
      },
      insertContent: (content: string) => {
        editor?.commands.insertContent(content)
      },
      replaceSelection: (content: string) => {
        editor?.commands.insertContent(content)
      },
    }))

    // Handle keyboard shortcuts
    const handleKeyDown = useCallback((event: KeyboardEvent) => {
      // Cmd/Ctrl + S to save
      if ((event.metaKey || event.ctrlKey) && event.key === 's') {
        event.preventDefault()
        // Trigger save - implement your save logic here
        console.log('Save triggered')
      }
    }, [])

    useEffect(() => {
      document.addEventListener('keydown', handleKeyDown)
      return () => {
        document.removeEventListener('keydown', handleKeyDown)
      }
    }, [handleKeyDown])

    if (!editor) {
      return (
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
        </div>
      )
    }

    return (
      <div className={`contract-editor ${className || ''}`}>
        <EditorToolbar editor={editor} />
        <div className="border border-gray-200 rounded-b-lg bg-white overflow-auto max-h-[calc(100vh-200px)]">
          <EditorContent editor={editor} />
        </div>
      </div>
    )
  }
)

ContractEditor.displayName = 'ContractEditor'

export default ContractEditor
