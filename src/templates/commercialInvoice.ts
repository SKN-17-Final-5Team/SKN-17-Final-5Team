// Commercial Invoice Template
export const commercialInvoiceHTML = `
<h1 style="text-align: center; font-size: 24pt; font-weight: bold; text-decoration: underline; margin-bottom: 30px;">COMMERCIAL INVOICE</h1>

<table>
  <tbody>
    <tr>
      <td style="width: 50%; vertical-align: top;" rowspan="2">
        <p style="font-weight: bold; font-size: 9pt; background-color: #f0f0f0; padding: 5px; margin: 0;">Shipper / Exporter</p>
        <div style="padding: 5px;">
          <p><span class="shared-field" data-field="SELLER_NAME">[SELLER_NAME]</span></p>
          <p><span class="shared-field" data-field="SELLER_ADDRESS">[SELLER_ADDRESS]</span></p>
        </div>
      </td>
      <td style="width: 25%;">
        <p style="font-weight: bold; font-size: 9pt; background-color: #f0f0f0; padding: 5px; margin: 0;">Invoice No.</p>
        <div style="padding: 5px;"><mark>[INVOICE_NO]</mark></div>
      </td>
      <td style="width: 25%;">
        <p style="font-weight: bold; font-size: 9pt; background-color: #f0f0f0; padding: 5px; margin: 0;">Date</p>
        <div style="padding: 5px;"><span class="shared-field" data-field="DATE">[DATE]</span></div>
      </td>
    </tr>
    <tr>
      <td>
        <p style="font-weight: bold; font-size: 9pt; background-color: #f0f0f0; padding: 5px; margin: 0;">L/C No.</p>
        <div style="padding: 5px;"><mark>[LC_NO]</mark></div>
      </td>
      <td>
        <p style="font-weight: bold; font-size: 9pt; background-color: #f0f0f0; padding: 5px; margin: 0;">L/C Date</p>
        <div style="padding: 5px;"><mark>[LC_DATE]</mark></div>
      </td>
    </tr>
    <tr>
      <td style="vertical-align: top;" rowspan="2">
        <p style="font-weight: bold; font-size: 9pt; background-color: #f0f0f0; padding: 5px; margin: 0;">Consignee</p>
        <div style="padding: 5px;">
          <p><span class="shared-field" data-field="BUYER_NAME">[BUYER_NAME]</span></p>
          <p><span class="shared-field" data-field="BUYER_ADDRESS">[BUYER_ADDRESS]</span></p>
        </div>
      </td>
      <td colspan="2">
        <p style="font-weight: bold; font-size: 9pt; background-color: #f0f0f0; padding: 5px; margin: 0;">Country of Origin</p>
        <div style="padding: 5px;"><mark>[COUNTRY_OF_ORIGIN]</mark></div>
      </td>
    </tr>
    <tr>
      <td colspan="2">
        <p style="font-weight: bold; font-size: 9pt; background-color: #f0f0f0; padding: 5px; margin: 0;">Terms of Delivery and Payment</p>
        <div style="padding: 5px;"><span class="shared-field" data-field="DELIVERY_TERMS">[DELIVERY_TERMS]</span></div>
      </td>
    </tr>
    <tr>
      <td style="vertical-align: top;">
        <p style="font-weight: bold; font-size: 9pt; background-color: #f0f0f0; padding: 5px; margin: 0;">Notify Party</p>
        <div style="padding: 5px;">
          <p><mark>[NOTIFY_PARTY]</mark></p>
        </div>
      </td>
      <td>
        <p style="font-weight: bold; font-size: 9pt; background-color: #f0f0f0; padding: 5px; margin: 0;">Port of Loading</p>
        <div style="padding: 5px;"><span class="shared-field" data-field="PORT_OF_LOADING">[PORT_OF_LOADING]</span></div>
      </td>
      <td>
        <p style="font-weight: bold; font-size: 9pt; background-color: #f0f0f0; padding: 5px; margin: 0;">Final Destination</p>
        <div style="padding: 5px;"><span class="shared-field" data-field="FINAL_DESTINATION">[FINAL_DESTINATION]</span></div>
      </td>
    </tr>
    <tr>
      <td style="vertical-align: top;">
        <p style="font-weight: bold; font-size: 9pt; background-color: #f0f0f0; padding: 5px; margin: 0;">Carrier</p>
        <div style="padding: 5px;"><span class="shared-field" data-field="CARRIER">[CARRIER]</span></div>
      </td>
      <td colspan="2">
        <p style="font-weight: bold; font-size: 9pt; background-color: #f0f0f0; padding: 5px; margin: 0;">Sailing on or about</p>
        <div style="padding: 5px;"><span class="shared-field" data-field="SAILING_DATE">[SAILING_DATE]</span></div>
      </td>
    </tr>
  </tbody>
</table>

<table style="margin-top: 20px;">
  <thead>
    <tr>
      <th style="width: 25%; background-color: #f0f0f0;">Marks &amp; Numbers</th>
      <th style="width: 35%; background-color: #f0f0f0;">Description of Goods</th>
      <th style="width: 10%; background-color: #f0f0f0;">Quantity</th>
      <th style="width: 15%; background-color: #f0f0f0;">Unit Price</th>
      <th style="width: 15%; background-color: #f0f0f0;">Amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><mark>[MARKS_NUMBERS]</mark></td>
      <td><mark>[GOODS_DESC]</mark></td>
      <td><mark>[QUANTITY]</mark></td>
      <td><mark>[UNIT_PRICE]</mark></td>
      <td><mark>[AMOUNT]</mark></td>
    </tr>
    <tr><td></td><td></td><td></td><td></td><td></td></tr>
    <tr><td></td><td></td><td></td><td></td><td></td></tr>
    <tr><td></td><td></td><td></td><td></td><td></td></tr>
    <tr><td></td><td></td><td></td><td></td><td></td></tr>
  </tbody>
  <tfoot>
    <tr>
      <td colspan="4" style="text-align: right; font-weight: bold;">TOTAL:</td>
      <td style="font-weight: bold;"><mark>[TOTAL_AMOUNT]</mark></td>
    </tr>
  </tfoot>
</table>

<div style="margin-top: 40px;">
  <p><strong>Bank Details:</strong></p>
  <ul>
    <li>Bank Name: <mark>[BANK_NAME]</mark></li>
    <li>Account No: <mark>[ACCOUNT_NO]</mark></li>
    <li>Swift Code: <mark>[SWIFT_CODE]</mark></li>
  </ul>
</div>

<div style="margin-top: 60px; text-align: right;">
  <p>Authorized Signature</p>
  <p style="margin-top: 40px;">_______________________</p>
  <p><span class="shared-field" data-field="SELLER_NAME">[SELLER_NAME]</span></p>
</div>
`
