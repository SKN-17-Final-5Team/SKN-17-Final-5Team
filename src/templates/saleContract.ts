// Sale Contract Template as HTML string
// Using HTML is more reliable than JSON for complex documents with tables
export const saleContractTemplateHTML = `
<h1 style="text-align: center">Sale Contract</h1>

<h2 style="text-align: center">Specific Conditions</h2>

<p>The purpose of this contract is to stipulate the rights, obligations, and all matters of both parties necessary for <mark>[SUPPLIER_NAME]</mark> (hereinafter referred to as "Supplier") and <mark>[BUYER_NAME]</mark> (hereinafter referred to as "Buyer") to sign the Overseas distribution contract for the distribution of the following goods. 'Supplier' and 'Buyer' are hereinafter referred to as "Parties to the Contract" or "Parties".</p>

<h3>Products</h3>

<table>
  <tr>
    <th>Item No.</th>
    <th>Commodity & Description</th>
    <th>Quantity</th>
    <th>Unit Price</th>
    <th>Total</th>
    <th>Notice</th>
  </tr>
  <tr>
    <td>1</td>
    <td><mark>[PRODUCT_1]</mark></td>
    <td><mark>[QTY_1]</mark></td>
    <td><mark>[PRICE_1]</mark></td>
    <td><mark>[TOTAL_1]</mark></td>
    <td></td>
  </tr>
  <tr>
    <td>2</td>
    <td><mark>[PRODUCT_2]</mark></td>
    <td><mark>[QTY_2]</mark></td>
    <td><mark>[PRICE_2]</mark></td>
    <td><mark>[TOTAL_2]</mark></td>
    <td></td>
  </tr>
  <tr>
    <td>3</td>
    <td><mark>[PRODUCT_3]</mark></td>
    <td><mark>[QTY_3]</mark></td>
    <td><mark>[PRICE_3]</mark></td>
    <td><mark>[TOTAL_3]</mark></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="4"><strong>Total</strong></td>
    <td><strong><mark>[GRAND_TOTAL]</mark></strong></td>
    <td></td>
  </tr>
</table>

<h3>Shipment Details</h3>

<ul>
  <li><strong>Time of Shipment:</strong> By <mark>[SHIPMENT_DATE]</mark></li>
  <li><strong>Cancellation Date for Late Shipment:</strong> <mark>[CANCELLATION_DATE]</mark></li>
  <li><strong>Port of Shipment:</strong> <mark>[PORT_OF_SHIPMENT]</mark></li>
  <li><strong>Port of Destination:</strong> <mark>[PORT_OF_DESTINATION]</mark></li>
  <li><strong>Partial Shipment:</strong> <mark>[PARTIAL_SHIPMENT]</mark></li>
  <li><strong>Transhipment:</strong> <mark>[TRANSHIPMENT]</mark></li>
  <li><strong>Delivery Terms:</strong> <mark>[DELIVERY_TERMS]</mark> Incoterms 2020</li>
</ul>

<h3>Payment</h3>

<p><strong>Payment Method:</strong> <mark>[PAYMENT_METHOD]</mark></p>
<p><strong>Payment Terms:</strong> <mark>[PAYMENT_TERMS]</mark></p>

<h3>Insurance & Packing</h3>

<ul>
  <li><strong>Insurance:</strong> Under CIF (or CIP), the Seller shall arrange cargo insurance in accordance with CIF (or CIP) of the latest Incoterms of the International Chamber of Commerce.</li>
  <li><strong>Packing:</strong> <mark>[PACKING]</mark></li>
  <li><strong>Marking:</strong> <mark>[MARKING]</mark></li>
</ul>

<h3>Documents Required</h3>

<ul>
  <li>Full set of Clean on Board Bills of Lading</li>
  <li>Signed Commercial Invoices in 3 Originals</li>
  <li>Packing Lists in 3 Originals</li>
  <li>Certificate of Origin in 1 Original plus 1 copy</li>
  <li>Inspection Certificate in 2 Originals</li>
</ul>

<hr>

<h2 style="text-align: center">General Terms and Conditions</h2>

<p>Other detailed terms and conditions of this contract are specified in the following 'General Terms and Conditions'. 'Supplier' and 'Purchaser' agree to the main contracts and "General Terms and Conditions", and to prove the establishment of this contract, two copies of the contract shall be prepared, mutually signed, and each part shall keep one copy of contract.</p>

<h3>1. General</h3>

<p>These General Terms and Conditions are intended to be applied together with the Specific Conditions. In case of contradiction between these General Terms and Conditions and the Specific Conditions agreed between the parties, the Specific Conditions shall prevail.</p>

<h3>2. Sales Territory, Channel</h3>

<ol>
  <li>'Supplier' authorizes the sales right for distribution channel in <mark>[SALES_COUNTRY]</mark> to 'Buyer'.</li>
  <li>The sales permission area is designated as <mark>[SALES_REGION]</mark> (hereinafter referred to as "region"), and 'Buyer' has the right to distribute and sell within the contract period.</li>
  <li>'Buyer' can conduct general sales within the scope of authorized region when its sales.</li>
  <li>The buyer's sales channels are permitted only for this contract are as follows: <mark>[SALES_CHANNELS]</mark></li>
</ol>

<h3>3. Shipment</h3>

<ol>
  <li>The date of issuance of the bill of lading will be deemed to be the date of shipment unless the bill of lading contains an on board notation indicating the date of shipment, in which case the date stated in the on board notation will be deemed to be the date of shipment.</li>
  <li>Partial shipment and/or transshipment shall be permitted unless otherwise stated in this Sales Contract.</li>
  <li>The Seller shall not be responsible for any delay in shipment due to the Buyer's failure to provide timely a documentary credit, as the case may be, in conformity with this Sales Contract.</li>
  <li>If the parties agreed upon a cancellation date for late shipment in the Specific Conditions, the Buyer may avoid the Sales Contract by notification to the Seller in case the shipment has not occurred by the cancellation date.</li>
</ol>

<h3>4. Packing and Marking</h3>

<ol>
  <li>Packing shall be performed at the Seller's option unless otherwise stated in this Sales Contract. In case special instructions are necessary, the Buyer should provide the Seller with such instructions in a timely manner. All the additional costs thereby incurred shall be borne by the Buyer.</li>
  <li>Shipping Mark shall be made as shown in the Specific Conditions, if any.</li>
</ol>

<h3>5. Insurance</h3>

<ol>
  <li>In case of CIF, 110% of the invoice amount shall be insured with insurance cover complying with the Institute Cargo Clauses (C) or similar clause.</li>
  <li>In case of CIP, 110% of the invoice amount shall be insured with insurance cover complying with the Institute Cargo Clauses (A) or similar clause.</li>
</ol>

<h3>6. Buyer's Obligation</h3>

<ol>
  <li>'Buyer' shall be responsible for sales account, advertisements, and sales promotions, and make best efforts to maximize sales in region.</li>
  <li>'Buyer' must obtain the permission from 'Supplier' when distributing articles related to permitted product to use and produce promotional materials, regardless of form as online and offline banners, catalogs, and pamphlets, etc.</li>
  <li>'Buyer' has the right to select customers, sub-distributors or partners within the defined region. However, before choosing a partner, share the partner's information with 'Supplier' in advance.</li>
  <li>'Buyer' shall investigate the complaints received by customers or business partners within the region and shall discuss the relevant measures with 'Supplier'.</li>
  <li>'Buyer' shall not disclose and use any confidential products or information related to 'Supplier' to a third party without prior written consent of 'Supplier', except for the obligation under this contract.</li>
</ol>

<h3>7. Supplier's Obligation</h3>

<ol>
  <li>'Supplier' shall support distributor's marketing activities. Especially, 'Supplier' shall make its best effort to support the documentations for marketing. However, it is limited to approved data.</li>
  <li>'Supplier' shall make its best effort to support the documentations for distribution and sales such as product information, export customs clearance documents, import sales (including certification).</li>
  <li>'Supplier' shall not infringe the authority of 'Buyer' in direct or indirect based on this contract, through its employees, agents or other agencies or either on its own.</li>
  <li>In the case of changing specifications such as important raw materials or product design in 'Supplier', 'Supplier' shall notify 'Buyer' by e-mail or writing before 20 days prior to the expected application of the change.</li>
</ol>

<h3>8. Inspection</h3>

<ol>
  <li>The inspection of the Goods shall be done according to the export regulation of the Republic of Korea and/or by the manufacturer(s), and such inspection shall be considered as final.</li>
  <li>Should any specific inspector be designated by the Buyer, all additional charges incurred thereby shall be borne by the Buyer and shall be added to the invoice amount.</li>
</ol>

<h3>9. Supply and Payment</h3>

<ol>
  <li>When ordering, 'Buyer' shall clarify information such as packing method, destination information, and order quantity, etc.</li>
  <li>'Supplier' shall notify to 'Buyer' the available quantity and delivery date within 3 business days from the date the order is received.</li>
  <li>If the parties have agreed on payment by a documentary credit, then, unless otherwise agreed, a documentary credit in favor of the Seller shall be issued within <mark>[LC_DAYS]</mark> days from the date of this Sales Contract.</li>
  <li>If the parties have agreed on payment by a documentary collection, then, unless otherwise agreed, the collection will be subject to the latest Uniform Rules for Collection (URC) of the International Chamber of Commerce.</li>
  <li>Payment is negotiable like 50% pre-deposit on the date of order (proforma invoice issuance date), and balance the remaining 50% before shipment to the bank account designated by 'Supplier' as T/T.</li>
</ol>

<h3>10. Warranty</h3>

<ol>
  <li>The Goods shall conform to the specification set forth in this Sales Contract, and shall be of good material & workmanship and free from any defect for at least <mark>[WARRANTY_MONTHS]</mark> months from the date of shipment.</li>
  <li>The extent of the Seller's liability under this warranty shall be limited to the repair or replacement as herein provided of any defective goods or parts thereof.</li>
  <li>Except for the express limited warranties set forth in this article, the Seller makes no other warranty to the Buyer, express or implied.</li>
</ol>

<h3>11. Claims</h3>

<ol>
  <li>Any claim by the Buyer of whatever nature arising under this Sales Contract shall be made by facsimile, cable, or e-mail within <mark>[CLAIM_DAYS]</mark> days after arrival of the goods at the destination specified in the bills of lading.</li>
  <li>The Buyer must submit with particulars the inspection report sworn by a reputable surveyor acceptable to the Seller when the quality or quantity of the goods delivered is in dispute.</li>
</ol>

<h3>12. Remedy</h3>

<ol>
  <li>The Buyer shall, without limitation, be in default of this Sales Contract, if the Buyer shall become insolvent, bankrupt or fail to make any payment to the Seller including the establishment of the documentary credit within the due date.</li>
  <li>In case of the Buyer's default, Seller may terminate this Sales Contract and recover from the Buyer as liquidated damages, a sum of <mark>[LIQUIDATED_DAMAGES_PERCENT]</mark> percent of the price of the unshipped balance.</li>
</ol>

<h3>13. Force Majeure</h3>

<p>A party shall not be liable for a failure to perform any of his obligations herein if he proves that the failure was due to an impediment beyond his control such as prohibition of exportation, suspension of issuance of export license or other government restriction, act of God, war, blockade, revolution, insurrection, mobilization, strike, lockout or any labor dispute, civil commotion, riot, plague or other epidemic, fire, typhoon, flood, etc.</p>

<h3>14. Patents, Trade Marks, Designs, etc.</h3>

<p>The Buyer acknowledges and agrees that any and all the Seller's intellectual property rights are the sole and exclusive property of the Seller or its licensors. The Buyer shall not acquire any ownership interest in any of the Seller's intellectual property rights under this Sales Contract.</p>

<h3>15. Confidentiality</h3>

<p>Not only the duration of the contract, but also for one year after the expiration of the contract maintenance period, according to this contract, 'Buyer' must not divulge any information of product-related technology, commercial information, pricing structure, data, sales, marketing, distribution, projects, plans, management, etc. The violation of this confidentiality obligation constitutes the serious breach of this contract.</p>

<h3>16. Governing Law</h3>

<p>All matters arising out of or relating to this Sales Contract are governed by and construed in accordance with the laws of Republic of Korea.</p>

<h3>17. Arbitration</h3>

<p>Any dispute arising out of or in connection with this Sales Contract shall be finally settled by arbitration in Seoul in accordance with the International Arbitration Rules of the Korean Commercial Arbitration Board and laws of Korea.</p>

<h3>18. Trade Terms</h3>

<p>All delivery terms provided in the Contract shall be interpreted in accordance with the latest Incoterms of International Chamber of Commerce.</p>

<hr>

<h3>Signatures</h3>

<p><strong>Date:</strong> <mark>[CONTRACT_DATE]</mark></p>

<table>
  <tr>
    <th>The Seller (Supplier)</th>
    <th>The Buyer</th>
  </tr>
  <tr>
    <td>
      <p><strong>Company:</strong> <mark>[SUPPLIER_COMPANY]</mark></p>
      <p><strong>Address:</strong> <mark>[SUPPLIER_ADDRESS]</mark></p>
      <p><strong>CEO:</strong> <mark>[SUPPLIER_CEO]</mark></p>
      <p><strong>Signature:</strong> _________________</p>
    </td>
    <td>
      <p><strong>Company:</strong> <mark>[BUYER_COMPANY]</mark></p>
      <p><strong>Address:</strong> <mark>[BUYER_ADDRESS]</mark></p>
      <p><strong>CEO:</strong> <mark>[BUYER_CEO]</mark></p>
      <p><strong>Signature:</strong> _________________</p>
    </td>
  </tr>
</table>

<hr>

<h2>Appendix</h2>

<ol>
  <li><strong>Product:</strong> <mark>[APPENDIX_PRODUCT]</mark></li>
  <li><strong>Supply Price:</strong> <mark>[APPENDIX_PRICE]</mark></li>
  <li><strong>Total Order & Quantity:</strong> <mark>[APPENDIX_QUANTITY]</mark></li>
  <li><strong>Lead Time:</strong> Within 30 days after PO</li>
</ol>
`

// List of all editable fields in the template
export const contractFields = [
  { key: 'SUPPLIER_NAME', label: 'Supplier Name', section: 'Parties' },
  { key: 'BUYER_NAME', label: 'Buyer Name', section: 'Parties' },
  { key: 'PRODUCT_1', label: 'Product 1', section: 'Products' },
  { key: 'QTY_1', label: 'Quantity 1', section: 'Products' },
  { key: 'PRICE_1', label: 'Unit Price 1', section: 'Products' },
  { key: 'TOTAL_1', label: 'Total 1', section: 'Products' },
  { key: 'PRODUCT_2', label: 'Product 2', section: 'Products' },
  { key: 'QTY_2', label: 'Quantity 2', section: 'Products' },
  { key: 'PRICE_2', label: 'Unit Price 2', section: 'Products' },
  { key: 'TOTAL_2', label: 'Total 2', section: 'Products' },
  { key: 'PRODUCT_3', label: 'Product 3', section: 'Products' },
  { key: 'QTY_3', label: 'Quantity 3', section: 'Products' },
  { key: 'PRICE_3', label: 'Unit Price 3', section: 'Products' },
  { key: 'TOTAL_3', label: 'Total 3', section: 'Products' },
  { key: 'GRAND_TOTAL', label: 'Grand Total', section: 'Products' },
  { key: 'SHIPMENT_DATE', label: 'Shipment Date', section: 'Shipment' },
  { key: 'CANCELLATION_DATE', label: 'Cancellation Date', section: 'Shipment' },
  { key: 'PORT_OF_SHIPMENT', label: 'Port of Shipment', section: 'Shipment' },
  { key: 'PORT_OF_DESTINATION', label: 'Port of Destination', section: 'Shipment' },
  { key: 'PARTIAL_SHIPMENT', label: 'Partial Shipment', section: 'Shipment' },
  { key: 'TRANSHIPMENT', label: 'Transhipment', section: 'Shipment' },
  { key: 'DELIVERY_TERMS', label: 'Delivery Terms', section: 'Shipment' },
  { key: 'PAYMENT_METHOD', label: 'Payment Method', section: 'Payment' },
  { key: 'PAYMENT_TERMS', label: 'Payment Terms', section: 'Payment' },
  { key: 'PACKING', label: 'Packing', section: 'Insurance & Packing' },
  { key: 'MARKING', label: 'Marking', section: 'Insurance & Packing' },
  { key: 'SALES_COUNTRY', label: 'Sales Country', section: 'Sales Territory' },
  { key: 'SALES_REGION', label: 'Sales Region', section: 'Sales Territory' },
  { key: 'SALES_CHANNELS', label: 'Sales Channels', section: 'Sales Territory' },
  { key: 'LC_DAYS', label: 'L/C Days', section: 'Payment' },
  { key: 'WARRANTY_MONTHS', label: 'Warranty Months', section: 'Warranty' },
  { key: 'CLAIM_DAYS', label: 'Claim Days', section: 'Claims' },
  { key: 'LIQUIDATED_DAMAGES_PERCENT', label: 'Liquidated Damages %', section: 'Remedy' },
  { key: 'CONTRACT_DATE', label: 'Contract Date', section: 'Signatures' },
  { key: 'SUPPLIER_COMPANY', label: 'Supplier Company', section: 'Signatures' },
  { key: 'SUPPLIER_ADDRESS', label: 'Supplier Address', section: 'Signatures' },
  { key: 'SUPPLIER_CEO', label: 'Supplier CEO', section: 'Signatures' },
  { key: 'BUYER_COMPANY', label: 'Buyer Company', section: 'Signatures' },
  { key: 'BUYER_ADDRESS', label: 'Buyer Address', section: 'Signatures' },
  { key: 'BUYER_CEO', label: 'Buyer CEO', section: 'Signatures' },
  { key: 'APPENDIX_PRODUCT', label: 'Appendix Product', section: 'Appendix' },
  { key: 'APPENDIX_PRICE', label: 'Appendix Price', section: 'Appendix' },
  { key: 'APPENDIX_QUANTITY', label: 'Appendix Quantity', section: 'Appendix' },
]
