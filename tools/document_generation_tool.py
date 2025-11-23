import os
from agents import function_tool

# Template directory path
TEMPLATE_DIR = "/Users/woojin/Desktop/무역_최신/SKN-17-Final-5Team/document_template/preprocessed_template"

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
def generate_trade_document(document_type: str, data_json: str) -> str:
    """
    Generates a trade document by filling a pre-defined HTML template with the provided data.

    Args:
        document_type: The type of document to generate. Must be one of:
                       ['Offer_Sheet', 'PI', 'Overseas_Distribution_Contract', 
                        'Commercial_Invoice', 'PL', 'BL', 'Letter_of_Credit']
        data_json: A JSON string representing a dictionary where keys are placeholders in the template 
                   and values are the strings to replace them with.
                   
                   For 'Offer_Sheet', valid keys are:
                   - [ Date ]
                   - [ Ref No ]
                   - [ Buyer Name ]
                   - [ Seller Name ]
                   - [ Item No ]
                   - [ HS Code ]
                   - [ Product Description ]
                   - [ Quantity ]
                   - [ Unit Price ]
                   - [ Amount ]
                   - [ Total Amount ]
                   - [ Country of Origin ]
                   - [ Shipment ]
                   - [ Inspection ]
                   - [ Payment ]
                   - [ Validity ]
                   - [ Remarks ]

    Returns:
        The filled HTML content as a string.
    """
    print(f"DEBUG: Received document_type: {document_type}")
    print(f"DEBUG: Received data_json: {data_json}")

    if document_type not in TEMPLATE_MAP:
        valid_types = ", ".join(TEMPLATE_MAP.keys())
        return f"Error: Invalid document_type '{document_type}'. Valid types are: {valid_types}"

    try:
        data = json.loads(data_json)
        if not isinstance(data, dict):
             return "Error: data_json must represent a dictionary."
    except json.JSONDecodeError:
        return "Error: data_json must be a valid JSON string."

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
    for key, value in data.items():
        str_value = str(value) if value is not None else ""
        
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
    output_dir = "/Users/woojin/Desktop/무역_최신/SKN-17-Final-5Team/document_version"
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
