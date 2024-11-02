from flask import Flask, request, jsonify
from utils.mod_logging import log_error, log_info
from utils.encryption import decrypt_data, encrypt_data
import importlib
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Define a mapping for dynamic routing based on action and qtype
MODULE_PATHS = {
    "speak_words": "speak_words.main",
    "template_checker": {
        "di": "template_checker.di_template_checker",
        "rl": "template_checker.rl_template_checker"
    }
}


def get_module_path(action, qtype=None):
    """Return module path based on action and qtype."""
    if action not in MODULE_PATHS:
        raise ValueError(f"Invalid action '{action}'")

    # For template_checker, we check for qtype to choose the correct module
    if action == "template_checker":
        if qtype not in MODULE_PATHS[action]:
            raise ValueError(
                f"Invalid qtype '{qtype}' for action 'template_checker'")
        return MODULE_PATHS[action][qtype]

    # Return direct module path if action is not template_checker
    return MODULE_PATHS[action]


@app.route('/api', methods=['POST'])
def route_request():
    try:
        # Get encrypted data from request and decrypt it
        encrypted_data = request.json.get("data")
        if not encrypted_data:
            raise ValueError("Missing encrypted data")

        # Log received encrypted data (optional, for debugging only)
        log_info("general", f"Received encrypted data: {encrypted_data}")

        # Decrypt and parse the data
        decrypted_data = decrypt_data(encrypted_data)
        data = json.loads(decrypted_data)

        # Extract necessary parameters
        action = data.get("action")
        qtype = data.get("qtype")
        qtext = data.get("qtext")
        image_type = data.get("image_type")
        response_text = data.get("response_text")
        tags = data.get("tags")

        # Determine module path based on action and qtype
        try:
            module_path = get_module_path(action, qtype)
            api_name = action if action != "template_checker" else f"{action}_{qtype}"
        except ValueError as e:
            log_error(str(e))
            return jsonify({"error": str(e)}), 400

        # Log action details
        log_info(
            api_name, f"Processing request with action '{action}' and qtype '{qtype}'")

        # Import the module dynamically and call the process_request function
        module = importlib.import_module(module_path)

        # Try to call process_request and catch any errors that occur
        try:
            # Pass all the extracted parameters to process_request
            response = module.process_request(
                qtext, tags, image_type, response_text)
            # Debugging line to print the response before encryption
            print("Response before encryption:", response)
        except Exception as e:
            log_error(f"Error in process_request: {e}")
            return jsonify({"error": "Error in processing request"}), 500

        # Log informational message specific to the API action
        log_info(
            api_name, f"Request processed successfully for action '{action}' with qtype '{qtype}'")

        # Encrypt response before returning
        encrypted_response = encrypt_data(json.dumps(response))

        # Log encrypted response (optional)
        log_info(
            api_name, f"Returning encrypted response: {encrypted_response}")

        return jsonify({"data": encrypted_response})

    except Exception as e:
        # Log all errors to the general error log
        log_error(f"Error processing request: {str(e)}")
        return jsonify({"error": "An internal server error occurred"}), 500


if __name__ == '__main__':
    app.run(debug=True)
