import { Mark, mergeAttributes } from '@tiptap/core'

export interface SharedFieldMarkOptions {
  HTMLAttributes: Record<string, unknown>
}

declare module '@tiptap/core' {
  interface Commands<ReturnType> {
    sharedFieldMark: {
      setSharedField: (field: string) => ReturnType
      unsetSharedField: () => ReturnType
    }
  }
}

export const SharedFieldMark = Mark.create<SharedFieldMarkOptions>({
  name: 'sharedFieldMark',

  // Higher priority than Highlight (default is 1000)
  priority: 1001,

  addOptions() {
    return {
      HTMLAttributes: {
        class: 'shared-field',
      },
    }
  },

  addAttributes() {
    return {
      'data-field': {
        default: null,
        parseHTML: (element) => element.getAttribute('data-field'),
        renderHTML: (attributes) => {
          if (!attributes['data-field']) {
            return {}
          }
          return { 'data-field': attributes['data-field'] }
        },
      },
    }
  },

  parseHTML() {
    return [
      {
        // Match span with shared-field class
        tag: 'span.shared-field[data-field]',
        getAttrs: (element) => {
          const el = element as HTMLElement
          return {
            'data-field': el.getAttribute('data-field'),
          }
        },
      },
      {
        // Also match mark with data-field for backwards compatibility
        tag: 'mark[data-field]',
        getAttrs: (element) => {
          const el = element as HTMLElement
          return {
            'data-field': el.getAttribute('data-field'),
          }
        },
      },
    ]
  },

  renderHTML({ HTMLAttributes }) {
    // Use span instead of mark to avoid conflict with Highlight extension
    return ['span', mergeAttributes(this.options.HTMLAttributes, HTMLAttributes), 0]
  },

  addCommands() {
    return {
      setSharedField:
        (field: string) =>
        ({ commands }) => {
          return commands.setMark(this.name, { 'data-field': field })
        },
      unsetSharedField:
        () =>
        ({ commands }) => {
          return commands.unsetMark(this.name)
        },
    }
  },
})

export default SharedFieldMark
