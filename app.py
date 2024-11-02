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

        log_info("general", f"Received encrypted data: {encrypted_data}")

        # Decrypt and parse the data
        decrypted_data = decrypt_data(encrypted_data)
        data = json.loads(decrypted_data)

        # Mandatory parameters
        required_params = ["request_id", "hostname", "username", "qid"]
        missing_params = [
            param for param in required_params if param not in data or not data[param]]

        if missing_params:
            missing_params_str = ", ".join(missing_params)
            error_message = f"Missing mandatory parameters: {missing_params_str}"
            log_error(error_message)
            return jsonify({"error": "error_code_pmissing"}), 400

        request_id = data["request_id"]
        hostname = data["hostname"]
        username = data["username"]
        qid = data["qid"]

        # Extract action and qtype
        action = data.get("action")
        qtype = data.get("qtype")

        # Optional parameters for specific actions
        qtext = data.get("qtext")
        tags = data.get("tags")
        image_type = data.get("image_type")
        response_text = data.get("response_text")

        # Determine module path based on action and qtype
        try:
            module_path = get_module_path(action, qtype)
            api_name = action if action != "template_checker" else f"{action}_{qtype}"
        except ValueError as e:
            log_error(str(e))
            return jsonify({"error": str(e)}), 400

        # Log action details
        log_info(
            api_name, f"Passing request with action '{action}' and qtype '{qtype} to module '{module_path}'")

        # Import the module dynamically and call the process_request function
        module = importlib.import_module(module_path)

        # Prepare arguments for process_request
        process_request_kwargs = {
            "request_id": request_id,
            "hostname": hostname,
            "username": username,
            "qid": qid
        }

        # Add optional fields only if they are relevant (for example, only for di_template_checker)
        if action == "template_checker" and qtype == "di":
            process_request_kwargs.update({
                "qtext": qtext,
                "tags": tags,
                "image_type": image_type,
                "response_text": response_text
            })

        # Call process_request with the prepared arguments
        try:
            response = module.process_request(**process_request_kwargs)
            print("Response before encryption:", response)
        except Exception as e:
            log_error(f"Error in process_request: {e}")
            return jsonify({"error": "Error in processing request"}), 500

        log_info(
            api_name, f"Request processed successfully for action '{action}' with qtype '{qtype}'")

        # Encrypt response before returning
        encrypted_response = encrypt_data(json.dumps(response))

        log_info(
            api_name, f"Returning encrypted response: {encrypted_response}")
        return jsonify({"data": encrypted_response})

    except Exception as e:
        log_error(f"Error processing request: {str(e)}")
        return jsonify({"error": "An internal server error occurred"}), 500


if __name__ == '__main__':
    app.run(debug=True)
