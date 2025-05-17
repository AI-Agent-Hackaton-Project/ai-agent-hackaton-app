import vertexai
from dotenv import load_dotenv
import os


def get_env_config():
    load_dotenv()
    gcp_project_id = os.getenv("GCP_PROJECT_ID")
    gcp_location = os.getenv("GCP_LOCATION")

    vertexai.init(project=gcp_project_id, location=gcp_location)

    config_data = {
        "gcp_project_id": gcp_project_id,
        "gcp_location": gcp_location,
        "model_name": os.getenv("VERTEX_AI_MODEL_NAME"),
        "max_output_tokens": 4096,
        "google_api_key": os.getenv("GOOGLE_API_KEY"),
        "google_cse_id": os.getenv("GOOGLE_CSE_ID"),
    }
    return config_data
