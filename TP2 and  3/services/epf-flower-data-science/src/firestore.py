import google.auth
from google.cloud import firestore
from google.oauth2 import service_account


class FirestoreClient:
    """Wrapper around a database"""

    client: firestore.Client

    def __init__(self) -> None:
        """Init the client."""
        credentials, _ = google.auth.default()
        self.client = firestore.Client(credentials=credentials)

    def get(self, collection_name: str, document_id: str) -> dict:
        """Find one document by ID.
        Args:
            collection_name: The collection name
            document_id: The document id
        Return:
            Document value.
        """
        doc = self.client.collection(
            collection_name).document(document_id).get()
        if doc.exists:
            return doc.to_dict()
        raise FileExistsError(
            f"No document found at {collection_name} with the id {document_id}"
        )

    def set_parameters(self, collection_name: str, document_id: str, n_estimators: int, criterion: str):
        """Create or update the parameters document in Firestore."""
        parameters = {
            "n_estimators": n_estimators,
            "criterion": criterion
        }
        doc_ref = self.client.collection(collection_name).document(document_id)
        doc_ref.set(parameters)
        print(f"Document '{document_id}' in collection '{collection_name}' has been created/updated.")

    def update(self, collection_name: str, document_name: str, data: dict):
        # Update a document in Firestore
        doc_ref = self.client.collection(collection_name).document(document_name)
        
        # Update the document with the new data
        doc_ref.update(data)