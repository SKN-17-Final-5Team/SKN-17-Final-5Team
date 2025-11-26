'use client'

import { createContext, useContext, useState, ReactNode } from 'react'

// Define all shared fields with their default values
interface SharedFields {
  SELLER_NAME: string
  SELLER_ADDRESS: string
  BUYER_NAME: string
  BUYER_ADDRESS: string
  DATE: string
  PORT_OF_LOADING: string
  PORT_OF_DESTINATION: string
  FINAL_DESTINATION: string
  DELIVERY_TERMS: string
  PAYMENT: string
  CARRIER: string
  SAILING_DATE: string
  HS_CODE: string
  [key: string]: string
}

interface SharedFieldsContextType {
  fields: SharedFields
  updateField: (key: string, value: string) => void
  updateFields: (updates: Partial<SharedFields>) => void
  applyFieldsToTemplate: (template: string) => string
}

const defaultFields: SharedFields = {
  SELLER_NAME: '[Seller Name]',
  SELLER_ADDRESS: '[Seller Address]',
  BUYER_NAME: '[Buyer Name]',
  BUYER_ADDRESS: '[Buyer Address]',
  DATE: '[Date]',
  PORT_OF_LOADING: '[Port of Loading]',
  PORT_OF_DESTINATION: '[Port of Destination]',
  FINAL_DESTINATION: '[Final Destination]',
  DELIVERY_TERMS: '[Delivery Terms]',
  PAYMENT: '[Payment]',
  CARRIER: '[Carrier]',
  SAILING_DATE: '[Sailing Date]',
  HS_CODE: '[HS Code]',
}

const SharedFieldsContext = createContext<SharedFieldsContextType | undefined>(undefined)

export function SharedFieldsProvider({ children }: { children: ReactNode }) {
  const [fields, setFields] = useState<SharedFields>(defaultFields)

  const updateField = (key: string, value: string) => {
    setFields(prev => ({ ...prev, [key]: value }))
  }

  const updateFields = (updates: Partial<SharedFields>) => {
    setFields(prev => {
      const filtered = Object.fromEntries(
        Object.entries(updates).filter(([_, v]) => v !== undefined)
      ) as Partial<SharedFields>
      return { ...prev, ...filtered } as SharedFields
    })
  }

  // Replace all shared field markers in template with actual values
  const applyFieldsToTemplate = (template: string): string => {
    let result = template
    Object.entries(fields).forEach(([key, value]) => {
      // Replace both [KEY] and <mark>[KEY]</mark> patterns
      const regex = new RegExp(`\\[${key}\\]`, 'g')
      result = result.replace(regex, value)
    })
    return result
  }

  return (
    <SharedFieldsContext.Provider value={{ fields, updateField, updateFields, applyFieldsToTemplate }}>
      {children}
    </SharedFieldsContext.Provider>
  )
}

export function useSharedFields() {
  const context = useContext(SharedFieldsContext)
  if (context === undefined) {
    throw new Error('useSharedFields must be used within a SharedFieldsProvider')
  }
  return context
}
