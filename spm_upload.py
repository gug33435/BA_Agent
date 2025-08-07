import requests
import time
import logging
import json
import html
from typing import Dict, Any

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SignavioImporter:
    """
    Authentifizierung per Signavio API und Import eines BPMN 2.0 XML Modells.
    Optimiert zur Nutzung als Modul.
    """

    def __init__(self, mail: str, password: str, workspace: str, host: str):
        if not all([mail, password, workspace, host]):
            raise ValueError("Missing credentials. Ensure USER_MAIL, USER_PASSWORD, WORKSPACE_ID, and HOST_URL are set.")
        
        self.host = host
        self.session = requests.Session()
        self._login(mail, password, workspace)

    def _login(self, mail: str, password: str, workspace: str):
        login_url = f"{self.host}/p/login"
        payload = {'tokenonly': 'true', 'name': mail, 'password': password, 'tenant': workspace}
        headers = {'Content-Type': 'application/x-www-form-urlencoded', 'Accept': "application/json"}

        try:
            response = self.session.post(login_url, data=payload, headers=headers)
            response.raise_for_status() # Raise an exception for HTTP error codes (4xx or 5xx)

            if len(response.text) > 50:
                logging.error("Login failed. The response body is unexpectedly long, suggesting an error page.")
                raise ValueError("Login failed. Please check your credentials and workspace ID.")

            auth_token = response.text
            self.session.headers.update({'x-signavio-id': auth_token})
            logging.info(f"Login successful for user {mail}.")

        except requests.exceptions.RequestException as e:
            logging.error(f"An API request error occurred during login: {e}")
            raise

    def get_root_directory_id(self) -> str | None:
        url = f"{self.host}/p/directory"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            content = response.json()
            
            if content and isinstance(content, list) and 'href' in content[0]:
                root_dir_id = content[0]['href'].replace('/directory/', '')
                logging.info(f"Root directory ID found: {root_dir_id}")
                return root_dir_id
            else:
                logging.error("Could not extract root directory ID from the response.")
                return None
        except (requests.exceptions.RequestException, ValueError, IndexError, KeyError) as e:
            logging.error(f"Failed to get root directory ID: {e}")
            return None

    def import_bpmn_xml_from_string(self, bpmn_xml_string: str, directory_id: str, diagram_name: str) -> Dict[str, Any] | None:
        import_url = f"{self.host}/p/bpmn2_0-import"
        form_data = {'directory': f'/directory/{directory_id}', 'filename': diagram_name}
        files = {'bpmn2_0file': (diagram_name, bpmn_xml_string, 'application/xml')}
        max_retries = 3
        retry_delay_seconds = 5
    
        for attempt in range(max_retries):
            response = None
            try:
                logging.info(f"Attempt {attempt + 1} of {max_retries} to import '{diagram_name}'...")
                response = self.session.post(import_url, data=form_data, files=files, timeout=60)
                response.raise_for_status()
                logging.info(f"Diagram '{diagram_name}' successfully imported to directory '{directory_id}'.")
                return json.loads(html.unescape(response.text))
            except requests.exceptions.RequestException as e:
                logging.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    logging.info(f"Retrying in {retry_delay_seconds} seconds...")
                    time.sleep(retry_delay_seconds)
                else:
                    logging.error(f"All {max_retries} attempts failed. Error importing BPMN string.")
                    if response is not None:
                        logging.error(f"Last server response: {response.text}")
                    return None
            except json.JSONDecodeError as e:
                logging.error(f"Error parsing the JSON response from server: {e}")
                return None
        return None
    
    def get_directory_content(self, directory_id: str) -> list[Dict[str, Any]] | None:
        if not directory_id:
            logging.error("Directory ID cannot be empty.")
            return None
        url = f"{self.host}/p/directory/{directory_id}"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to get content for directory {directory_id}: {e}")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response for directory content: {e}")
            return None
        
    def delete_model(self, model_id: str) -> bool:
        if not model_id.startswith('/model/'):
            model_id = f"/model/{model_id}"
        url = f"{self.host}/p{model_id}"
        try:
            response = self.session.delete(url)
            response.raise_for_status()
            logging.info(f"Successfully deleted model: {model_id}")
            return True
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to delete model {model_id}: {e}")
            if response is not None: logging.error(f"Server response: {response.text}")
            return False