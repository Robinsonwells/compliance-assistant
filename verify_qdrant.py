import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Load environment variables
load_dotenv()

def init_qdrant_client():
    """Initialize Qdrant client"""
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if not qdrant_url or not qdrant_api_key:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY environment variables must be set")
    
    return QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
    )

def get_collection_info(client):
    """Get overall collection information"""
    try:
        collection_info = client.get_collection("legal_regulations")
        return {
            "total_points": collection_info.points_count,
            "vector_size": collection_info.config.params.vectors.size,
            "status": collection_info.status
        }
    except Exception as e:
        return {"error": str(e)}

def count_chunks_for_file(client, filename):
    """Count chunks for a specific file"""
    try:
        result = client.count(
            collection_name="legal_regulations",
            count_filter=Filter(
                must=[
                    FieldCondition(
                        key="source_file",
                        match=MatchValue(value=filename)
                    )
                ]
            )
        )
        return result.count
    except Exception as e:
        print(f"Error counting chunks for {filename}: {e}")
        return None

def list_all_files(client):
    """Get list of all unique source files in the collection"""
    try:
        # Scroll through all points to get unique source files
        all_files = set()
        next_page_offset = None
        
        while True:
            scroll_result = client.scroll(
                collection_name="legal_regulations",
                limit=1000,  # Process in batches
                offset=next_page_offset,
                with_payload=True,
                with_vectors=False
            )
            
            points, next_page_offset = scroll_result
            
            for point in points:
                source_file = point.payload.get('source_file', 'Unknown')
                if source_file != 'Unknown':
                    all_files.add(source_file)
            
            # Break if no more pages
            if next_page_offset is None:
                break
        
        return sorted(list(all_files))
    except Exception as e:
        print(f"Error listing files: {e}")
        return []

def get_file_details(client, filename):
    """Get detailed information about chunks for a specific file"""
    try:
        scroll_result = client.scroll(
            collection_name="legal_regulations",
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="source_file",
                        match=MatchValue(value=filename)
                    )
                ]
            ),
            limit=10,  # Just get first 10 for sample
            with_payload=True,
            with_vectors=False
        )
        
        points, _ = scroll_result
        
        if not points:
            return None
        
        # Extract sample metadata
        sample_chunk = points[0].payload
        return {
            "sample_citation": sample_chunk.get('citation', 'N/A'),
            "sample_jurisdiction": sample_chunk.get('jurisdiction', 'N/A'),
            "sample_section": sample_chunk.get('section_number', 'N/A'),
            "upload_date": sample_chunk.get('upload_date', 'N/A'),
            "content_hash": sample_chunk.get('content_hash', 'N/A')
        }
    except Exception as e:
        print(f"Error getting file details for {filename}: {e}")
        return None

def main():
    print("🔍 Qdrant Database Verification Tool")
    print("=" * 50)
    
    try:
        # Initialize client
        print("📡 Connecting to Qdrant...")
        client = init_qdrant_client()
        print("✅ Connected successfully!")
        
        # Get collection info
        print("\n📊 Collection Overview:")
        collection_info = get_collection_info(client)
        if "error" in collection_info:
            print(f"❌ Error: {collection_info['error']}")
            return
        
        print(f"   Total Points: {collection_info['total_points']}")
        print(f"   Vector Size: {collection_info['vector_size']}")
        print(f"   Status: {collection_info['status']}")
        
        # List all files
        print("\n📁 Files in Database:")
        all_files = list_all_files(client)
        
        if not all_files:
            print("   No files found in database")
            return
        
        for i, filename in enumerate(all_files, 1):
            count = count_chunks_for_file(client, filename)
            print(f"   {i}. {filename} ({count} chunks)")
        
        # Interactive verification
        print("\n" + "=" * 50)
        print("🔍 Interactive Verification")
        
        while True:
            print(f"\nAvailable files:")
            for i, filename in enumerate(all_files, 1):
                print(f"   {i}. {filename}")
            
            choice = input(f"\nEnter file number to verify (1-{len(all_files)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                break
            
            try:
                file_index = int(choice) - 1
                if 0 <= file_index < len(all_files):
                    filename = all_files[file_index]
                    
                    print(f"\n🔍 Verifying: {filename}")
                    print("-" * 40)
                    
                    # Count chunks
                    count = count_chunks_for_file(client, filename)
                    print(f"📊 Chunk Count: {count}")
                    
                    if count > 0:
                        print("❌ File NOT deleted - chunks still exist in Qdrant")
                        
                        # Get file details
                        details = get_file_details(client, filename)
                        if details:
                            print(f"📋 Sample Details:")
                            print(f"   Citation: {details['sample_citation']}")
                            print(f"   Jurisdiction: {details['sample_jurisdiction']}")
                            print(f"   Section: {details['sample_section']}")
                            print(f"   Upload Date: {details['upload_date']}")
                            print(f"   Content Hash: {details['content_hash'][:16]}...")
                    else:
                        print("✅ File successfully deleted - no chunks found in Qdrant")
                else:
                    print("❌ Invalid file number")
            except ValueError:
                print("❌ Please enter a valid number or 'q'")
            except Exception as e:
                print(f"❌ Error: {e}")
    
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check your .env file has QDRANT_URL and QDRANT_API_KEY")
        print("2. Verify your Qdrant credentials are correct")
        print("3. Ensure your internet connection is working")

if __name__ == "__main__":
    main()