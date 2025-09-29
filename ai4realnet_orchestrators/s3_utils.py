import os

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", False)
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", False)
AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL", None)
S3_UPLOAD_ROOT = os.getenv("S3_UPLOAD_PATH_TEMPLATE", "ai4realnet/submissions/")

S3_BUCKET = os.getenv("S3_BUCKET", "aicrowd-production")
S3_BUCKET_ACL = "public-read" if S3_BUCKET == "aicrowd-production" else ""


class s3_utils:

  @staticmethod
  def upload_to_s3(localpath, relative_upload_key, submission_id):
    s3 = s3_utils.get_boto_client()
    if not S3_BUCKET:
      raise Exception("S3_BUCKET not provided...")

    file_target_key = S3_UPLOAD_ROOT + relative_upload_key
    s3.put_object(
      ACL=S3_BUCKET_ACL,
      Bucket=S3_BUCKET,
      Key=file_target_key,
      Body=open(localpath, 'rb')
    )
    return file_target_key

  @staticmethod
  def get_boto_client():
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
      raise Exception("AWS Credentials not provided..")
    try:
      import boto3  # type: ignore
    except ImportError:
      raise Exception(
        "boto3 is not installed. Please manually install by : ",
        " pip install -U boto3"
      )

    return boto3.client(
      's3',
      # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/core/session.html
      endpoint_url=AWS_ENDPOINT_URL,
      aws_access_key_id=AWS_ACCESS_KEY_ID,
      aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

  @staticmethod
  def is_aws_configured():
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
      return False
    else:
      return True
