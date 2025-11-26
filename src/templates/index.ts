// Export all templates
export { offerSheetHTML } from './offerSheet'
export { packingListHTML } from './packingList'
export { commercialInvoiceHTML } from './commercialInvoice'
export { saleContractHTML } from './saleContract'

// Document types
export type DocumentType = 'offer' | 'packingList' | 'commercialInvoice' | 'saleContract'

export interface DocumentInfo {
  id: DocumentType
  name: string
  nameKo: string
  template: string
}

import { offerSheetHTML } from './offerSheet'
import { packingListHTML } from './packingList'
import { commercialInvoiceHTML } from './commercialInvoice'
import { saleContractHTML } from './saleContract'

export const documents: DocumentInfo[] = [
  {
    id: 'offer',
    name: 'Offer Sheet',
    nameKo: '오퍼 시트',
    template: offerSheetHTML,
  },
  {
    id: 'packingList',
    name: 'Packing List',
    nameKo: '패킹 리스트',
    template: packingListHTML,
  },
  {
    id: 'commercialInvoice',
    name: 'Commercial Invoice',
    nameKo: '상업 송장',
    template: commercialInvoiceHTML,
  },
  {
    id: 'saleContract',
    name: 'Sale Contract',
    nameKo: '판매 계약서',
    template: saleContractHTML,
  },
]

export const getTemplateByType = (type: DocumentType): string => {
  const doc = documents.find(d => d.id === type)
  return doc?.template || ''
}

// Shared fields that are common across all documents
export const sharedFieldKeys = [
  'SELLER_NAME',
  'SELLER_ADDRESS',
  'BUYER_NAME',
  'BUYER_ADDRESS',
  'DATE',
  'PORT_OF_LOADING',
  'PORT_OF_DESTINATION',
  'FINAL_DESTINATION',
  'DELIVERY_TERMS',
  'PAYMENT',
  'CARRIER',
  'SAILING_DATE',
  'HS_CODE',
] as const

export type SharedFieldKey = typeof sharedFieldKeys[number]
