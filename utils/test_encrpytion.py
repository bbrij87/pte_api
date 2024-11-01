# Import both functions to test encryption and decryption
from encryption import encrypt_data, decrypt_data
import json

# Sample data to test the API
test_data = {
    "action": "template_checker",
    "qtype": "di",
    "qtext": "This is a sample text",
    "image_type": "bar_chart",
    "response_text": "The bar graph image is about vehicle colour involved in total-loss collision...",
    "tags": "vehicle, collision, frequency, accident, blue, green, red, white, black, grey, high, low, comparison, involvement, statistics, traffic, safety, analysis"
}

# Convert data to JSON format
json_data = json.dumps(test_data)

# Encrypt the JSON data
encrypted_data = encrypt_data(json_data)
print("Encrypted Data:", encrypted_data)

# Decrypt to verify
try:
    decrypted_data = decrypt_data(encrypted_data)
    # Convert decrypted JSON string back to dictionary
    decrypted_json = json.loads(decrypted_data)
    print("Decrypted Data:", decrypted_json)
except Exception as e:
    print("Decryption failed:", e)
