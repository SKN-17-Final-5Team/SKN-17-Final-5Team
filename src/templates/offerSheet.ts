// Offer Sheet Template
export const offerSheetHTML = `
<h1 style="text-align: center; font-size: 20pt; font-weight: bold; margin-bottom: 10px;">OFFER SHEET</h1>
<h2 style="text-align: center; margin-bottom: 40px;"><span class="shared-field" data-field="SELLER_NAME">[SELLER_NAME]</span></h2>

<div style="margin-bottom: 40px;">
  <p><strong>Date</strong> : <span class="shared-field" data-field="DATE">[DATE]</span></p>
  <p><strong>Ref No.</strong> : <mark>[REF_NO]</mark></p>
  <p><strong>MESSRS.</strong> : <span class="shared-field" data-field="BUYER_NAME">[BUYER_NAME]</span></p>
</div>

<p>We are pleased to offer you as follows;</p>

<table>
  <thead>
    <tr>
      <th style="width: 12%;">Item No.</th>
      <th style="width: 18%;">HS-CODE</th>
      <th style="width: 30%;">Product</th>
      <th style="width: 10%;">Q'ty</th>
      <th style="width: 15%;">Unit Price</th>
      <th style="width: 15%;">Amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><mark>[ITEM_NO]</mark></td>
      <td><span class="shared-field" data-field="HS_CODE">[HS_CODE]</span></td>
      <td><mark>[PRODUCT_DESC]</mark></td>
      <td><mark>[QUANTITY]</mark></td>
      <td><mark>[UNIT_PRICE]</mark></td>
      <td><mark>[AMOUNT]</mark></td>
    </tr>
    <tr>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
  <tfoot>
    <tr>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td style="text-align: center; font-weight: bold;">TOTAL :</td>
      <td><mark>[TOTAL_AMOUNT]</mark></td>
    </tr>
  </tfoot>
</table>

<h3>Conditions</h3>

<ul>
  <li><strong>Country of Origin</strong> : <mark>[COUNTRY_OF_ORIGIN]</mark></li>
  <li><strong>Shipment</strong> : <mark>[SHIPMENT]</mark></li>
  <li><strong>Inspection</strong> : <mark>[INSPECTION]</mark></li>
  <li><strong>Payment</strong> : <span class="shared-field" data-field="PAYMENT">[PAYMENT]</span></li>
  <li><strong>Validity</strong> : <mark>[VALIDITY]</mark></li>
  <li><strong>Remarks</strong> : <mark>[REMARKS]</mark></li>
</ul>

<div style="margin-top: 40px;">
  <p>Sincerely yours,</p>
  <p style="margin-top: 20px;"><strong><span class="shared-field" data-field="SELLER_NAME">[SELLER_NAME]</span></strong></p>
</div>

<hr style="margin-top: 40px;" />

<p style="font-weight: bold;">APPENDIX</p>
`
