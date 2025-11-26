// Packing List Template
export const packingListHTML = `
<h1 style="text-align: center; font-size: 24pt; font-weight: bold; margin-bottom: 5px;">PACKING LIST</h1>
<hr style="width: 300px; margin: 0 auto 40px; border-top: 2px solid #000;" />

<table style="margin-bottom: 0;">
  <tbody>
    <tr>
      <td style="width: 50%; vertical-align: top;" rowspan="2">
        <p style="font-weight: bold; font-size: 10pt; margin-bottom: 5px;">Shipper / Exporter</p>
        <p><span class="shared-field" data-field="SELLER_NAME">[SELLER_NAME]</span></p>
        <p><span class="shared-field" data-field="SELLER_ADDRESS">[SELLER_ADDRESS]</span></p>
      </td>
      <td style="width: 50%;">
        <p style="font-weight: bold; font-size: 10pt; margin-bottom: 5px;">No. &amp; Date of Invoice</p>
        <p><mark>[INVOICE_NO]</mark> / <span class="shared-field" data-field="DATE">[DATE]</span></p>
      </td>
    </tr>
    <tr>
      <td>
        <p style="font-weight: bold; font-size: 10pt; margin-bottom: 5px;">No. &amp; Date of L/C</p>
        <p><mark>[LC_NO]</mark> / <mark>[LC_DATE]</mark></p>
      </td>
    </tr>
    <tr>
      <td style="vertical-align: top;">
        <p style="font-weight: bold; font-size: 10pt; margin-bottom: 5px;">For Account &amp; Risk of Messrs.</p>
        <p><span class="shared-field" data-field="BUYER_NAME">[BUYER_NAME]</span></p>
        <p><span class="shared-field" data-field="BUYER_ADDRESS">[BUYER_ADDRESS]</span></p>
      </td>
      <td>
        <p style="font-weight: bold; font-size: 10pt; margin-bottom: 5px;">Remarks</p>
        <p><mark>[REMARKS]</mark></p>
      </td>
    </tr>
    <tr>
      <td style="vertical-align: top;">
        <p style="font-weight: bold; font-size: 10pt; margin-bottom: 5px;">Notify Party</p>
        <p><mark>[NOTIFY_PARTY_NAME]</mark></p>
        <p><mark>[NOTIFY_PARTY_ADDRESS]</mark></p>
      </td>
      <td rowspan="3"></td>
    </tr>
    <tr>
      <td>
        <table style="width: 100%; border: none;">
          <tr>
            <td style="width: 50%; border: none; border-right: 1px solid #000;">
              <p style="font-weight: bold; font-size: 10pt; margin-bottom: 5px;">Port of Loading</p>
              <p><span class="shared-field" data-field="PORT_OF_LOADING">[PORT_OF_LOADING]</span></p>
            </td>
            <td style="width: 50%; border: none;">
              <p style="font-weight: bold; font-size: 10pt; margin-bottom: 5px;">Final Destination</p>
              <p><span class="shared-field" data-field="FINAL_DESTINATION">[FINAL_DESTINATION]</span></p>
            </td>
          </tr>
        </table>
      </td>
    </tr>
    <tr>
      <td>
        <table style="width: 100%; border: none;">
          <tr>
            <td style="width: 50%; border: none; border-right: 1px solid #000;">
              <p style="font-weight: bold; font-size: 10pt; margin-bottom: 5px;">Carrier</p>
              <p><span class="shared-field" data-field="CARRIER">[CARRIER]</span></p>
            </td>
            <td style="width: 50%; border: none;">
              <p style="font-weight: bold; font-size: 10pt; margin-bottom: 5px;">Sailing on or about</p>
              <p><span class="shared-field" data-field="SAILING_DATE">[SAILING_DATE]</span></p>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </tbody>
</table>

<table>
  <thead>
    <tr>
      <th style="width: 30%;">Marks and Number of PKGS</th>
      <th style="width: 35%;">Description of Goods</th>
      <th style="width: 8%;">Q'ty</th>
      <th style="width: 9%;">Net Weight</th>
      <th style="width: 9%;">Gross Weight</th>
      <th style="width: 9%;">Measurement</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><mark>[MARKS_NUMBER]</mark></td>
      <td><mark>[GOODS_DESC]</mark></td>
      <td><mark>[QUANTITY]</mark></td>
      <td><mark>[NET_WEIGHT]</mark></td>
      <td><mark>[GROSS_WEIGHT]</mark></td>
      <td><mark>[MEASUREMENT]</mark></td>
    </tr>
    <tr><td></td><td></td><td></td><td></td><td></td><td></td></tr>
    <tr><td></td><td></td><td></td><td></td><td></td><td></td></tr>
    <tr><td></td><td></td><td></td><td></td><td></td><td></td></tr>
    <tr><td></td><td></td><td></td><td></td><td></td><td></td></tr>
  </tbody>
  <tfoot>
    <tr>
      <td style="text-align: right; font-weight: bold;">Total :</td>
      <td style="font-weight: bold;"><mark>[TOTAL_CARTONS]</mark> cartons</td>
      <td style="text-align: center; font-weight: bold;">Total</td>
      <td style="text-align: center; font-weight: bold;"><mark>[TOTAL_NET]</mark> KG</td>
      <td style="text-align: center; font-weight: bold;"><mark>[TOTAL_GROSS]</mark> KG</td>
      <td style="text-align: center; font-weight: bold;"><mark>[TOTAL_CBM]</mark> CBM</td>
    </tr>
  </tfoot>
</table>

<div style="margin-top: 40px; text-align: right; font-weight: bold;">
  Signed by _________________
</div>

<div style="margin-top: 40px; font-weight: bold;">
  <p>H.S Code: <span class="shared-field" data-field="HS_CODE">[HS_CODE]</span></p>
  <p>Trade Terms: <mark>[TRADE_TERMS]</mark></p>
</div>
`
