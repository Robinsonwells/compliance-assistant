import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
collection_name = "legal_regulations"

if not qdrant_url or not qdrant_api_key:
    print("Error: QDRANT_URL and QDRANT_API_KEY environment variables must be set")
    exit(1)

print(f"Connecting to Qdrant at: {qdrant_url}")
print(f"Collection to delete: {collection_name}")

try:
    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
    )
    
    # Check if the collection exists before trying to delete
    try:
        collection_info = client.get_collection(collection_name)
        print(f"Found collection '{collection_name}' with {collection_info.points_count} points")
        print(f"Current vector size: {collection_info.config.params.vectors.size}")
        
        # Delete the collection
        client.delete_collection(collection_name=collection_name)
        print(f"✅ Collection '{collection_name}' deleted successfully!")
        
    except Exception as get_error:
        if "doesn't exist" in str(get_error).lower() or "not found" in str(get_error).lower():
            print(f"Collection '{collection_name}' does not exist - nothing to delete.")
        else:
            print(f"Error checking collection: {get_error}")
            # Try to delete anyway
            try:
                client.delete_collection(collection_name=collection_name)
                print(f"✅ Collection '{collection_name}' deleted successfully!")
            except Exception as delete_error:
                print(f"❌ Error deleting collection: {delete_error}")
                
except Exception as e:
    print(f"❌ Error connecting to Qdrant: {e}")

print("\nNext steps:")
print("1. The collection has been deleted")
print("2. Redeploy your Streamlit app")
print("3. The collection will be recreated with 384 dimensions")
print("4. You can then upload your documents again")