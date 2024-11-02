from flask import Flask, request, jsonify
from utils.mod_logging import log_error, log_info
from utils.encryption import decrypt_data, encrypt_data
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from jsonschema import validate, ValidationError
import importlib
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask application
app = Flask(__name__)

# Set up Flask-Limiter for rate limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["30000 per hour"]  # Set global limits here
)

# Define payload size limit in bytes (e.g., 1MB)
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # 1MB

# Define a mapping for dynamic routing based on action and qtype
MODULE_PATHS = {
    "speak_words": "speak_words.main",
    "template_checker": {
        "di": "template_checker.di_template_checker",
        "rl": "template_checker.rl_template_checker"
    }
}

# Define JSON schema for input validation
request_schema = {
    "type": "object",
    "properties": {
        "request_id": {"type": "string"},
        "hostname": {"type": "string"},
        "username": {"type": "string"},
        "qid": {"type": "string"},
        "action": {"type": "string"},
        "qtype": {"type": ["string", "null"]},
        "qtext": {"type": ["string", "null"]},
        "tags": {"type": ["string", "null"]},
        "image_type": {"type": ["string", "null"]},
        "response_text": {"type": ["string", "null"]}
    },
    # Only these fields are required
    "required": ["request_id", "hostname", "username", "qid"],
    "additionalProperties": False  # No extra fields allowed
}


def validate_request_data(data):
    """Validate request data against the JSON schema."""
    try:
        validate(instance=data, schema=request_schema)
    except ValidationError as e:
        raise ValueError(f"Invalid input: {e.message}")


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
# Limit this endpoint to 10 requests per minute
@limiter.limit("10 per minute")
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

        # Validate input data against JSON schema
        validate_request_data(data)

        # Extract mandatory parameters
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
            api_name, f"Passing request with action '{action}' and qtype '{qtype}' to module '{module_path}'")

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
            log_info(api_name, f"Response before encryption: {response}")
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

    except ValueError as e:
        log_error(f"Validation Error: {e}")
        return jsonify({"error": "Invalid input"}), 400
    except Exception as e:
        log_error(f"Error processing request: {str(e)}")
        return jsonify({"error": "An internal server error occurred"}), 500


if __name__ == '__main__':
    app.run(debug=True)
