import os
from agents import function_tool

# Template directory path
# 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(BASE_DIR, "data_embedding", "data", "document_template", "preprocessed_template")
OUTPUT_DIR = os.path.join(BASE_DIR, "data_embedding", "data", "document_version")

# Mapping of document types to their HTML template filenames
TEMPLATE_MAP = {
    "Offer_Sheet": "Offer_Sheet.html",
    "PI": "PI.html",
    "Overseas_Distribution_Contract": "Overseas_Distribution_Contract.html",
    "Commercial_Invoice": "Commercial_Invoice.html",
    "PL": "PL.html",
    "BL": "Bill_of_Lading.html",
    "Letter_of_Credit": "Letter_of_Credit.html"
}

import json

@function_tool
def generate_trade_document(document_type: str, data_json: str, overwrite: bool = False) -> str:
    """
    Generates a trade document by filling a pre-defined HTML template with the provided data.

    Args:
        document_type: The type of document to generate. Must be one of:
                       ['Offer_Sheet', 'PI', 'Overseas_Distribution_Contract', 
                        'Commercial_Invoice', 'PL', 'BL', 'Letter_of_Credit']
        data_json: A JSON string representing a dictionary where keys are placeholders in the template 
                   and values are the strings to replace them with.
                   
                   For 'Offer_Sheet', valid keys are:
                   - [ Date ], [ Ref No ], [ Buyer Name ], [ Seller Name ], [ Item No ], [ HS Code ], 
                   - [ Product Description ], [ Quantity ], [ Unit Price ], [ Amount ], [ Total Amount ], 
                   - [ Country of Origin ], [ Shipment ], [ Inspection ], [ Payment ], [ Validity ], [ Remarks ]
                   
                   * Note: For multiple items, combine them into a single string separated by '\\n' (newline).
                     The tool will automatically convert '\\n' to '<br>' for proper formatting.
                     IMPORTANT: Even if values (like HS Code) are the same for all items, repeat them for each item (e.g., "1234\n1234") to maintain row alignment.

                   For 'PI' (Proforma Invoice), valid keys are:
                   - [ Date ], [ Proforma Invoice No ]
                   - [ Seller Name ], [ Seller Department ], [ Seller Address ], [ Seller City ], [ Seller Country ], [ Seller Tel ]
                   - [ Buyer Name ], [ Buyer Department ], [ Buyer Address ], [ Buyer City ], [ Buyer Country ], [ Buyer Tel ]
                   - [ Number of Pieces ], [ Total Gross Weight ], [ Total Net Weight ], [ Carrier ]
                   - [ Description of Goods ], [ Commodity Code ], [ Country of Origin ], [ Quantity ], [ Unit Value ], [ Subtotal Value ], [ Total Value ]
                   - [ Term of Transportation ], [ Reason for Export ]
                   
                   * Note: For multiple items, use '\\n' (newline) to separate values. DO NOT use '/'.
                   * IMPORTANT: Repeat values for EACH item (e.g., "Korea\\nKorea") to maintain table alignment.

                   For 'Overseas_Distribution_Contract', valid keys are:
                   - [ Seller Name ], [ Buyer Name ], [ Seller Address ], [ Buyer Address ]
                   - [ Contract Date ]
                   - [ Item No ], [ Product Description ], [ Quantity ], [ Unit Price ], [ Amount ], [ Remarks ]
                   - [ Time of Shipment ], [ Cancellation Date ], [ Port of Shipment ], [ Port of Destination ]
                   - [ Delivery Terms ], [ Territory ]
                   - [ Warranty Period ], [ Claim Period ], [ Claim Detail Period ]
                   - [ Appendix Product ], [ Appendix Price ], [ Appendix Quantity ]
                   - [ Marking ]
                   
                   * Note: For multiple items, combine them into a single string separated by '\\n' (newline).

                   For 'Commercial_Invoice', valid keys are:
                   - [ Shipper Name ], [ Address ], [ City, Country ]
                   - [ Invoice No. ], [ Date ], [ L/C No. ], [ Issuing Bank Name ]
                   - [ Consignee Name ], [ Address ], [ City, Country ]
                   - [ Notify Party Name ], [ Address ]
                   - [ Port of Loading ], [ Final Destination ], [ Carrier Name ], [ Date ]
                   - [ Marks ], [ Description ], [ EA/BOX ], [ Box ], [ Total EA ], [ Price ], [ Amount ], [ Total Amount ]

                   For 'BL' (Bill of Lading), valid keys are:
                   - [ Shipper Name ], [ Shipper Address ]
                   - [ B/L No ]
                   - [ Consignee Name ], [ Consignee Address ]
                   - [ Notify Party Name ], [ Notify Party Address ]
                   - [ Pre-Carriage ], [ Place of Receipt ]
                   - [ Ocean Vessel ], [ Voyage No ], [ Flag ]
                   - [ Port of Loading ], [ Port of Discharge ], [ Place of Delivery ], [ Final Destination ]
                   - [ Container No ], [ Seal No ], [ Marks and No ], [ No and Kinds of Packages ], [ Description of Goods ], [ Gross Weight ], [ Measurement ]
                   - [ Total Containers ]

                   For 'PL' (Packing List), valid keys are:
                   - [ Shipper Name ], [ Shipper Address ]
                   - [ Invoice No ], [ Date ], [ L/C No ], [ L/C Date ]
                   - [ Buyer Name ], [ Buyer Address ], [ Remarks ]
                   - [ Notify Party Name ], [ Notify Party Address ]
                   - [ Port of Loading ], [ Final Destination ], [ Carrier ], [ Sailing Date ]
                   - [ Marks and Number ], [ Description of Goods ], [ Quantity ], [ Net Weight ], [ Gross Weight ], [ Measurement ]

    Returns:
        The filled HTML content as a string.
    """
    print(f"DEBUG: Received document_type: {document_type}")
    print(f"DEBUG: Received data_json: {data_json}")
    print(f"DEBUG: Received overwrite: {overwrite}")

    if document_type not in TEMPLATE_MAP:
        valid_types = ", ".join(TEMPLATE_MAP.keys())
        return f"Error: Invalid document_type '{document_type}'. Valid types are: {valid_types}"

    try:
        new_data = json.loads(data_json)
        if not isinstance(new_data, dict):
             return "Error: data_json must represent a dictionary."
    except json.JSONDecodeError:
        return "Error: data_json must be a valid JSON string."

    # [State Persistence Logic]
    # Define path for the data file (e.g., latest_Offer_Sheet_data.json)
    data_file_name = f"latest_{document_type}_data.json"
    data_file_path = os.path.join(OUTPUT_DIR, data_file_name)

    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    final_data = {}

    # 1. Load existing data if available and not overwriting
    if not overwrite and os.path.exists(data_file_path):
        try:
            with open(data_file_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                if isinstance(existing_data, dict):
                    final_data = existing_data
                    print(f"DEBUG: Loaded existing data: {final_data.keys()}")
        except Exception as e:
            print(f"WARNING: Failed to load existing data: {e}")

    # 2. Merge new data (new data takes precedence)
    final_data.update(new_data)
    print(f"DEBUG: Final merged data keys: {final_data.keys()}")

    # 3. Save updated data back to JSON
    try:
        with open(data_file_path, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error saving data state: {str(e)}"

    template_filename = TEMPLATE_MAP[document_type]
    template_path = os.path.join(TEMPLATE_DIR, template_filename)

    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        return f"Error: Template file not found at {template_path}"
    except Exception as e:
        return f"Error reading template: {str(e)}"

    # Replace placeholders with data
    # Replace placeholders with data
    for key, value in final_data.items():
        str_value = str(value) if value is not None else ""
        
        # Convert newlines to <br> for HTML rendering
        str_value = str_value.replace('\n', '<br>')
        
        # [NEW] Auto-fill Date if empty
        # If the key is related to the main document date and is empty, use today's date
        if not str_value.strip():
            clean_key = key.strip("[] ")
            if clean_key in ["Date", "Contract Date"]:
                from datetime import datetime
                str_value = datetime.now().strftime("%Y-%m-%d")

        # [NEW] Highlight missing information
        # If the value is empty (meaning user didn't provide it), make it visually distinct
        if not str_value.strip():
            # Use a yellow background with red text for high visibility
            clean_key_for_display = key.strip("[] ")
            str_value = f'<span style="background-color: #ffffcc; color: #cc0000; font-weight: bold; border: 1px dashed #cc0000; padding: 2px 4px;">[ {clean_key_for_display} ]</span>'
        
        # The template uses [ Key ] format (with spaces).
        # The agent might send "Key" or "[ Key ]".
        
        # 1. Construct the standard placeholder format used in the template
        # If the key already has brackets, strip them first to normalize
        clean_key = key.strip("[] ")
        placeholder = f"[ {clean_key} ]"
        
        # 2. Replace
        if placeholder in content:
            content = content.replace(placeholder, str_value)
        else:
            # Fallback: try tight brackets [Key] just in case
            tight_placeholder = f"[{clean_key}]"
            if tight_placeholder in content:
                content = content.replace(tight_placeholder, str_value)
            # Fallback: try raw key if it's unique enough (risky but maybe needed)
            elif key in content and len(key) > 3: # Avoid replacing short common words
                 content = content.replace(key, str_value)

    # Save the file with versioning
    output_dir = OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Find the next version number
    version = 1
    while True:
        filename = f"{document_type}_v{version}.html"
        file_path = os.path.join(output_dir, filename)
        if not os.path.exists(file_path):
            break
        version += 1

    # Write the file
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        return f"Error saving file: {str(e)}"

    # Return a success message with the link
    # Using sandbox: protocol for the link if supported, or just the path
    return f"{document_type}가 성공적으로 작성되었습니다. [여기](sandbox:{file_path})에서 확인하실 수 있습니다."
