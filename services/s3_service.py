"""
S3 문서 저장/로드 서비스

문서 원본을 S3에 저장하고 필요 시 불러오는 서비스
"""

import os
import boto3
from botocore.exceptions import ClientError
from typing import Optional, BinaryIO
from datetime import datetime
import mimetypes


class S3Service:
    """S3 문서 관리 서비스"""

    def __init__(
        self,
        bucket_name: str,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: str = "ap-northeast-2"
    ):
        """
        Args:
            bucket_name: S3 버킷 이름
            aws_access_key_id: AWS Access Key (None이면 환경변수 사용)
            aws_secret_access_key: AWS Secret Key (None이면 환경변수 사용)
            region_name: AWS 리전 (기본값: 서울)
        """
        self.bucket_name = bucket_name

        # AWS 자격증명
        if aws_access_key_id and aws_secret_access_key:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name
            )
        else:
            # 환경변수에서 자동으로 로드
            self.s3_client = boto3.client('s3', region_name=region_name)

    def upload_document(
        self,
        file_path: str,
        s3_key: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> str:
        """
        문서를 S3에 업로드

        Args:
            file_path: 업로드할 로컬 파일 경로
            s3_key: S3에 저장될 키 (None이면 자동 생성)
            metadata: 추가 메타데이터

        Returns:
            S3 객체 키
        """
        # S3 키 생성
        if not s3_key:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = os.path.basename(file_path)
            s3_key = f"documents/{timestamp}_{file_name}"

        # Content-Type 자동 감지
        content_type, _ = mimetypes.guess_type(file_path)
        if not content_type:
            content_type = 'application/octet-stream'

        # 메타데이터 준비
        extra_args = {'ContentType': content_type}
        if metadata:
            extra_args['Metadata'] = metadata

        try:
            self.s3_client.upload_file(
                file_path,
                self.bucket_name,
                s3_key,
                ExtraArgs=extra_args
            )
            print(f"✓ S3 업로드 완료: s3://{self.bucket_name}/{s3_key}")
            return s3_key

        except ClientError as e:
            print(f"❌ S3 업로드 실패: {e}")
            raise

    def upload_fileobj(
        self,
        file_obj: BinaryIO,
        s3_key: str,
        content_type: str = 'application/octet-stream',
        metadata: Optional[dict] = None
    ) -> str:
        """
        파일 객체를 S3에 업로드

        Args:
            file_obj: 파일 객체 (바이너리 모드)
            s3_key: S3 객체 키
            content_type: MIME 타입
            metadata: 추가 메타데이터

        Returns:
            S3 객체 키
        """
        extra_args = {'ContentType': content_type}
        if metadata:
            extra_args['Metadata'] = metadata

        try:
            self.s3_client.upload_fileobj(
                file_obj,
                self.bucket_name,
                s3_key,
                ExtraArgs=extra_args
            )
            print(f"✓ S3 업로드 완료: s3://{self.bucket_name}/{s3_key}")
            return s3_key

        except ClientError as e:
            print(f"❌ S3 업로드 실패: {e}")
            raise

    def download_document(
        self,
        s3_key: str,
        local_path: Optional[str] = None
    ) -> str:
        """
        S3에서 문서 다운로드

        Args:
            s3_key: S3 객체 키
            local_path: 저장할 로컬 경로 (None이면 임시 폴더에 저장)

        Returns:
            다운로드된 로컬 파일 경로
        """
        # 로컬 경로 생성
        if not local_path:
            os.makedirs("temp", exist_ok=True)
            file_name = os.path.basename(s3_key)
            local_path = f"temp/{file_name}"

        try:
            self.s3_client.download_file(
                self.bucket_name,
                s3_key,
                local_path
            )
            print(f"✓ S3 다운로드 완료: {local_path}")
            return local_path

        except ClientError as e:
            print(f"❌ S3 다운로드 실패: {e}")
            raise

    def get_document_url(
        self,
        s3_key: str,
        expiration: int = 3600
    ) -> str:
        """
        문서의 presigned URL 생성 (임시 접근 URL)

        Args:
            s3_key: S3 객체 키
            expiration: URL 유효 시간 (초, 기본값: 1시간)

        Returns:
            Presigned URL
        """
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': s3_key
                },
                ExpiresIn=expiration
            )
            return url

        except ClientError as e:
            print(f"❌ Presigned URL 생성 실패: {e}")
            raise

    def delete_document(self, s3_key: str) -> bool:
        """
        S3에서 문서 삭제

        Args:
            s3_key: S3 객체 키

        Returns:
            삭제 성공 여부
        """
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            print(f"✓ S3 삭제 완료: s3://{self.bucket_name}/{s3_key}")
            return True

        except ClientError as e:
            print(f"❌ S3 삭제 실패: {e}")
            return False

    def list_documents(self, prefix: str = "documents/") -> list:
        """
        S3 버킷의 문서 목록 조회

        Args:
            prefix: 검색할 접두사

        Returns:
            문서 목록 (키, 크기, 수정 시간)
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )

            if 'Contents' not in response:
                return []

            documents = []
            for obj in response['Contents']:
                documents.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'].isoformat()
                })

            return documents

        except ClientError as e:
            print(f"❌ S3 목록 조회 실패: {e}")
            return []

    def check_document_exists(self, s3_key: str) -> bool:
        """
        S3에 문서가 존재하는지 확인

        Args:
            s3_key: S3 객체 키

        Returns:
            존재 여부
        """
        try:
            self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            return True

        except ClientError:
            return False
