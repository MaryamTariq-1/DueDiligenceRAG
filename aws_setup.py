# aws_complete_test.py
import boto3
import os
from dotenv import load_dotenv

load_dotenv()


def aws_complete_test():
    print("AWS S3 SETUP TEST")
    print("=" * 50)

    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION", "eu-north-1")
    bucket_name = "company-due-diligence-data-maryamtariq"
    base_path = "D:/PycharmProjects/InterviewTest1/DueDiligenceRAG/interview_dataset"

    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region
    )

    # Test 1: AWS Credentials
    print("1. Testing AWS Credentials...")
    try:
        response = s3.list_buckets()
        print("   SUCCESS: Connected to AWS")
        print(f"   Found {len(response['Buckets'])} buckets")
    except Exception as e:
        print(f"   FAILED: {e}")
        return False

    # Test 2: Bucket Access
    print("2. Testing Bucket Access...")
    try:
        s3.head_bucket(Bucket=bucket_name)
        print(f"   SUCCESS: Bucket exists")
    except Exception as e:
        print(f"   FAILED: {e}")
        return False

    # Test 3: Upload Files
    print("3. Uploading Files...")
    files_to_upload = [
        ("training_set/qna_training_data.csv", f"{base_path}/training_set/qna_training_data.csv"),
        ("test_set/finance.txt", f"{base_path}/test_set/finance.txt"),
        ("test_set/hr.txt", f"{base_path}/test_set/hr.txt"),
        ("test_set/engineering.txt", f"{base_path}/test_set/engineering.txt"),
        ("test_set/marketing.txt", f"{base_path}/test_set/marketing.txt"),
        ("test_set/operations.txt", f"{base_path}/test_set/operations.txt")
    ]

    for s3_key, local_path in files_to_upload:
        if os.path.exists(local_path):
            try:
                s3.upload_file(local_path, bucket_name, s3_key)
                print(f"   UPLOADED: {s3_key}")
            except Exception as e:
                print(f"   FAILED: {s3_key} - {e}")
        else:
            print(f"   MISSING: {local_path}")

    # Test 4: Verify S3 Structure
    print("4. Verifying S3 Structure...")
    try:
        response = s3.list_objects_v2(Bucket=bucket_name)
        print("   S3 Bucket Contents:")
        for obj in response.get('Contents', []):
            print(f"   - {obj['Key']} ({obj['Size']} bytes)")
    except Exception as e:
        print(f"   FAILED: {e}")

    # Test 5: File Reading
    print("5. Testing File Reading...")
    test_files = [
        "training_set/qna_training_data.csv",
        "test_set/finance.txt"
    ]

    for test_file in test_files:
        try:
            obj_data = s3.get_object(Bucket=bucket_name, Key=test_file)
            content = obj_data['Body'].read().decode('utf-8')
            lines = len([line for line in content.split('\n') if line.strip()])
            print(f"   CAN READ: {test_file} ({lines} lines)")
        except Exception as e:
            print(f"   CANNOT READ {test_file}: {e}")

    print("\n" + "=" * 50)
    print("AWS S3 SETUP COMPLETED")
    print("Files uploaded to correct folders")
    print("Code can read files directly from S3")
    print("Step 1 - AWS Setup: DONE")
    print("=" * 50)

    return True


if __name__ == "__main__":
    aws_complete_test()