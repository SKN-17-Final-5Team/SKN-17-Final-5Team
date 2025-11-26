// Sale Contract Template
export const saleContractHTML = `
<h1 style="text-align: center; font-size: 24pt; font-weight: bold; text-decoration: underline; margin-bottom: 40px;">Sale Contract</h1>

<h2 style="text-align: center; font-size: 14pt; font-weight: bold; text-decoration: underline; margin-bottom: 30px;">Specific Conditions</h2>

<p style="text-align: justify;">
The purpose of this contract is to stipulate the rights, obligations, and all matters of both parties necessary for <span class="shared-field" data-field="SELLER_NAME">[SELLER_NAME]</span> (hereinafter referred to as "Seller") and <span class="shared-field" data-field="BUYER_NAME">[BUYER_NAME]</span> (hereinafter referred to as "Buyer") to sign the Overseas distribution contract for the distribution of the following goods. 'Supplier' and 'Buyer' are hereinafter referred to as "Parties to the Contract" or "Parties".
</p>

<table>
  <thead>
    <tr>
      <th style="width: 10%; background-color: #000; color: #fff;">Item No.</th>
      <th style="width: 35%; background-color: #000; color: #fff;">Commodity &amp; Description</th>
      <th style="width: 15%; background-color: #000; color: #fff;">Quantity</th>
      <th style="width: 15%; background-color: #000; color: #fff;">Unit Price</th>
      <th style="width: 15%; background-color: #000; color: #fff;">Total</th>
      <th style="width: 10%; background-color: #000; color: #fff;">Notice</th>
    </tr>
  </thead>
  <tbody>
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
    <tr style="background-color: #f0f0f0; font-weight: bold;">
      <td>합계</td>
      <td></td>
      <td></td>
      <td></td>
      <td><mark>[GRAND_TOTAL]</mark></td>
      <td></td>
    </tr>
  </tbody>
</table>

<ul>
  <li><strong>Time of Shipment</strong> : By <mark>[SHIPMENT_DATE]</mark><br/>
      * Cancellation Date for Late Shipment : <mark>[CANCELLATION_DATE]</mark></li>
  <li><strong>Port of Shipment</strong> : <span class="shared-field" data-field="PORT_OF_LOADING">[PORT_OF_LOADING]</span></li>
  <li><strong>Port of Destination</strong> : <span class="shared-field" data-field="FINAL_DESTINATION">[FINAL_DESTINATION]</span></li>
  <li><strong>Partial Shipment</strong> : <mark>[PARTIAL_SHIPMENT]</mark></li>
  <li><strong>Transhipment</strong> : <mark>[TRANSHIPMENT]</mark></li>
  <li><strong>Delivery Terms</strong> : <span class="shared-field" data-field="DELIVERY_TERMS">[DELIVERY_TERMS]</span> Incoterms 2020</li>
</ul>

<h3>Payment</h3>

<table>
  <tbody>
    <tr>
      <td rowspan="3" style="background-color: #d9d9d9; font-weight: bold; text-align: center; width: 20%;">Letter of Credit (L/C)</td>
      <td style="background-color: #d9d9d9; font-weight: bold; text-align: center; width: 20%;">Sight Credit</td>
      <td style="width: 5%; text-align: center;">○</td>
      <td>irrevocable documentary credit payable at sight</td>
    </tr>
    <tr>
      <td style="background-color: #d9d9d9; font-weight: bold; text-align: center;">Deferred Payment</td>
      <td style="text-align: center;">○</td>
      <td>irrevocable documentary credit with deferred payment at <mark>[DEFER_DAYS]</mark> days from B/L date</td>
    </tr>
    <tr>
      <td style="background-color: #d9d9d9; font-weight: bold; text-align: center;">Acceptance Credit</td>
      <td style="text-align: center;">○</td>
      <td>irrevocable documentary credit with acceptance of drafts at <mark>[ACCEPT_DAYS]</mark> days from B/L date</td>
    </tr>
    <tr>
      <td rowspan="2" style="background-color: #d9d9d9; font-weight: bold; text-align: center;">Documentary Collection</td>
      <td style="background-color: #d9d9d9; font-weight: bold; text-align: center;">D/P</td>
      <td style="text-align: center;">○</td>
      <td>documents against payment</td>
    </tr>
    <tr>
      <td style="background-color: #d9d9d9; font-weight: bold; text-align: center;">D/A</td>
      <td style="text-align: center;">○</td>
      <td>documents against acceptance payable at <mark>[DA_DAYS]</mark> days from B/L date</td>
    </tr>
    <tr>
      <td colspan="2" style="background-color: #d9d9d9; font-weight: bold; text-align: center;">Payment in Advance</td>
      <td style="text-align: center;">○</td>
      <td>T/T within <mark>[TT_DAYS]</mark> days from B/L date</td>
    </tr>
    <tr>
      <td colspan="2" style="background-color: #d9d9d9; font-weight: bold; text-align: center;">Other</td>
      <td style="text-align: center;">○</td>
      <td><mark>[OTHER_PAYMENT]</mark></td>
    </tr>
  </tbody>
</table>

<ul>
  <li><strong>Insurance</strong> : Under CIF (or CIP), the Seller shall arrange cargo insurance in accordance with CIF (or CIP) of the latest Incoterms.</li>
  <li><strong>Packing</strong> : <mark>[PACKING]</mark></li>
  <li><strong>Marking</strong> : <mark>[MARKING]</mark></li>
  <li><strong>Documents Required</strong> :
    <ul>
      <li>Full set of Clean on Board Bills of Lading</li>
      <li>Signed Commercial Invoices in 3 Originals</li>
      <li>Packing Lists in 3 Originals</li>
      <li>Certificate of Origin in 1 Original plus 1 copy</li>
      <li>Inspection Certificate in 2 Originals</li>
    </ul>
  </li>
  <li><strong>Other</strong> : <mark>[OTHER_CONDITIONS]</mark></li>
</ul>

<p style="margin-top: 60px;">These Specific Conditions are subject to the General Terms and Conditions set forth below.</p>

<table style="margin-top: 40px;">
  <tr>
    <th style="width: 50%;">The Seller</th>
    <th style="width: 50%;">The Buyer</th>
  </tr>
  <tr>
    <td>
      <p><strong>By :</strong> <span class="shared-field" data-field="SELLER_NAME">[SELLER_NAME]</span></p>
      <p><strong>Address :</strong> <span class="shared-field" data-field="SELLER_ADDRESS">[SELLER_ADDRESS]</span></p>
      <p><strong>Name / Title :</strong> <mark>[SELLER_TITLE]</mark></p>
      <p><strong>Signature :</strong> _________________</p>
    </td>
    <td>
      <p><strong>By :</strong> <span class="shared-field" data-field="BUYER_NAME">[BUYER_NAME]</span></p>
      <p><strong>Address :</strong> <span class="shared-field" data-field="BUYER_ADDRESS">[BUYER_ADDRESS]</span></p>
      <p><strong>Name / Title :</strong> <mark>[BUYER_TITLE]</mark></p>
      <p><strong>Signature :</strong> _________________</p>
    </td>
  </tr>
</table>

<hr style="margin-top: 60px;" />

<h2 style="text-align: center; font-size: 14pt; font-weight: bold; text-decoration: underline; margin-top: 40px;">General Terms and Conditions</h2>

<p style="text-align: justify;">Other detailed terms and conditions of this contract are specified in the following 'General Terms and Conditions'. 'Supplier' and 'Purchaser' agree to the main contracts and "General Terms and Conditions", and to prove the establishment of this contract, two copies of the contract shall be prepared, mutually signed, and each part shall keep one copy of contract.</p>

<h3>1. [General]</h3>
<p>These General Terms and Conditions are intended to be applied together with the Specific Conditions. In case of contradiction between these General Terms and Conditions and the Specific Conditions agreed between the parties, the Specific Conditions shall prevail.</p>

<h3>2. [Sales Territory, Channel]</h3>
<ol>
  <li>'Supplier' authorizes the sales right for distribution channel in <mark>[SALES_COUNTRY]</mark> to 'Buyer'.</li>
  <li>The sales permission area is designated as <mark>[SALES_REGION]</mark> (hereinafter referred to as "region"), and 'Buyer' has the right to distribute and sell within the contract period.</li>
  <li>'Buyer' can conduct general sales within the scope of authorized region when its sales.</li>
  <li>The buyer's sales channels are permitted only for this contract are as follows: <mark>[SALES_CHANNELS]</mark></li>
</ol>

<h3>3. [Shipment]</h3>
<ol>
  <li>The date of issuance of the bill of lading will be deemed to be the date of shipment unless the bill of lading contains an on board notation indicating the date of shipment.</li>
  <li>Partial shipment and/or transshipment shall be permitted unless otherwise stated in this Sales Contract.</li>
  <li>The Seller shall not be responsible for any delay in shipment due to the Buyer's failure to provide timely a documentary credit.</li>
  <li>If the parties agreed upon a cancellation date for late shipment, the Buyer may avoid the Sales Contract by notification to the Seller.</li>
</ol>

<h3>4. [Packing and Marking]</h3>
<ol>
  <li>Packing shall be performed at the Seller's option unless otherwise stated. All additional costs shall be borne by the Buyer.</li>
  <li>Shipping Mark shall be made as shown in the Specific Conditions, if any.</li>
</ol>

<h3>5. [Insurance]</h3>
<ol>
  <li>In case of CIF, 110% of the invoice amount shall be insured with Institute Cargo Clauses (C).</li>
  <li>In case of CIP, 110% of the invoice amount shall be insured with Institute Cargo Clauses (A).</li>
</ol>

<h3>6. [Buyer's Obligation]</h3>
<ol>
  <li>'Buyer' shall be responsible for sales account, advertisements, and sales promotions.</li>
  <li>'Buyer' must obtain permission from 'Supplier' for promotional materials.</li>
  <li>'Buyer' has the right to select customers, sub-distributors within the defined region.</li>
  <li>'Buyer' shall investigate complaints and discuss measures with 'Supplier'.</li>
  <li>'Buyer' shall not disclose confidential information without prior written consent.</li>
</ol>

<h3>7. [Supplier's Obligation]</h3>
<ol>
  <li>'Supplier' shall support distributor's marketing activities.</li>
  <li>'Supplier' shall support documentation for distribution and sales.</li>
  <li>'Supplier' shall not infringe the authority of 'Buyer'.</li>
  <li>'Supplier' shall notify 'Buyer' of specification changes 20 days in advance.</li>
</ol>

<h3>8. [Inspection]</h3>
<ol>
  <li>Inspection shall be done according to export regulation of Republic of Korea.</li>
  <li>Additional charges for specific inspector shall be borne by the Buyer.</li>
</ol>

<h3>9. [Supply and Payment]</h3>
<ol>
  <li>When ordering, 'Buyer' shall clarify packing method, destination, and quantity.</li>
  <li>'Supplier' shall notify available quantity and delivery date within 3 business days.</li>
  <li>Documentary credit shall be issued within <mark>[LC_DAYS]</mark> days from contract date.</li>
  <li>Documentary collection will be subject to URC of ICC.</li>
  <li>Payment: 50% pre-deposit on order, 50% balance before shipment by T/T.</li>
</ol>

<h3>10. [Warranty]</h3>
<p>The Goods shall be free from defect for at least <mark>[WARRANTY_MONTHS]</mark> months from the date of shipment. Warranty is limited to repair or replacement of defective goods.</p>

<h3>11. [Claims]</h3>
<p>Any claim shall be made within <mark>[CLAIM_DAYS]</mark> days after arrival of goods at destination. The Buyer must submit inspection report by reputable surveyor.</p>

<h3>12. [Remedy]</h3>
<p>In case of Buyer's default, Seller may terminate contract and recover liquidated damages of <mark>[DAMAGE_PERCENT]</mark>% of the unshipped balance.</p>

<h3>13. [Force Majeure]</h3>
<p>A party shall not be liable for failure due to impediment beyond control such as war, natural disasters, government restrictions, etc.</p>

<h3>14. [Patents, Trade Marks, Designs]</h3>
<p>The Buyer acknowledges that all intellectual property rights are the sole property of the Seller.</p>

<h3>15. [Confidentiality]</h3>
<p>'Buyer' must not divulge any confidential information during contract and for one year after expiration.</p>

<h3>16. [Governing Law]</h3>
<p>All matters are governed by the laws of Republic of Korea.</p>

<h3>17. [Arbitration]</h3>
<p>Any dispute shall be settled by arbitration in Seoul in accordance with Korean Commercial Arbitration Board rules.</p>

<h3>18. [Trade Terms]</h3>
<p>All delivery terms shall be interpreted in accordance with the latest Incoterms of ICC.</p>

<hr />

<h3>Signatures</h3>
<p><strong>Date:</strong> <mark>[CONTRACT_DATE]</mark></p>

<table>
  <tr>
    <td style="width: 50%;">
      <p><strong>Supplier:</strong> <span class="shared-field" data-field="SELLER_NAME">[SELLER_NAME]</span></p>
      <p><strong>Address:</strong> <span class="shared-field" data-field="SELLER_ADDRESS">[SELLER_ADDRESS]</span></p>
      <p><strong>CEO:</strong> <mark>[SELLER_CEO]</mark> (sign)</p>
    </td>
    <td style="width: 50%;">
      <p><strong>Buyer:</strong> <span class="shared-field" data-field="BUYER_NAME">[BUYER_NAME]</span></p>
      <p><strong>Address:</strong> <span class="shared-field" data-field="BUYER_ADDRESS">[BUYER_ADDRESS]</span></p>
      <p><strong>CEO:</strong> <mark>[BUYER_CEO]</mark> (sign)</p>
    </td>
  </tr>
</table>

<hr />

<h2>Appendix</h2>
<ol>
  <li><strong>Product:</strong> <mark>[APPENDIX_PRODUCT]</mark></li>
  <li><strong>Supply Price:</strong> <mark>[APPENDIX_PRICE]</mark></li>
  <li><strong>Total Order &amp; Quantity:</strong> <mark>[APPENDIX_QUANTITY]</mark></li>
  <li><strong>Lead Time:</strong> Within 30 days after PO</li>
</ol>
`
