from utils.encryption import encrypt_data, decrypt_data
from utils.mod_logging import log_error, log_info

import sys
import os
import json

# command to run: sgb@SGBs-Laptop pte_api # python3 -m tests.test_encryption

# Import both functions to test encryption and decryption
# Sample data to test the API
test_data = {
    "action": "template_checker",
    "qtype": "di",
    "qtext": "a This is a sample text",
    "image_type": "bar_chart",
    "response_text": "The bar graph image is about vehicle colour involved in total-loss collision...",
    "tags": "vehicle, collision, frequency, accident, blue, green, red, white, black, grey, high, low, comparison, involvement, statistics, traffic, safety, analysis"
}

# Convert data to JSON format
json_data = json.dumps(test_data)

# Encrypt the JSON data
# encrypted_data = encrypt_data(json_data)
encrypted_data = '''
US-VUIWMQW7dN3TPitOVMZfFxOlQNnSSmIdjBdijQr5qc2DtYkgu5xWhmm_PXX94IWpJqsOpLq3_Tt4gIEZO7I45zoHq9tElWZ5xJTiO4k3u-mlABlA9yrIygob2lsO0FpkdPD6ThpshQyJ24tvIOeZwgWVMGsL81IvwwV5g44DLwK7y8zDyXs4pIgfDUhtrCn8nTzTaFaTXWeWkj_fppn3Ovau1CSAECdAfhCFNZs9JVMiG1zy9BWt5Yq8uWakWLxnfHEuLQLEz1C0rkBsY9cnPBIh1S6TuagWmGShxpMYw2RfaPwRxxVxjFPRwj0TF0WtOaec4ANKfzp9JiYD2dqhAY_CRzEX1-0zMEiNGSGJJFzIqPch0LHc21FgvdOrvoKZqJaxgPW5kP5C1g0IbLmtNXm5qdkWCW2BsaGRmbc4_MQnPKhg3mzpty8y2hQoMWSBijd_QARGspdxb4PSJOiGZe7jCtBHcvX_GrE-2v7Z9xzTlOM8S6ik7GZn-jaB7EnSBkJOIKJXWQssqG1nz9O8wxWzvPeF2Yy5KJD36u8o=
'''
print("Encrypted Data:", encrypted_data)

# Decrypt to verify
try:
    decrypted_data = decrypt_data(encrypted_data)
    # Convert decrypted JSON string back to dictionary
    decrypted_json = json.loads(decrypted_data)
    print("Decrypted Data:", decrypted_json)
except Exception as e:
    print("Decryption failed:", e)
