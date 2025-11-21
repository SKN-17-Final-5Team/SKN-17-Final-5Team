from typing import List, Union
import numpy as np
from openai import OpenAI
import os


class EmbeddingService:

    def __init__(
        self,
        model_name: str = "text-embedding-3-large",
        device: str = None
    ):
        """
        Args:
            model_name: 사용할 임베딩 모델 (기본값: text-embedding-3-large)
        """
        self.model_name = model_name
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._embedding_dim = 3072  # text-embedding-3-large의 차원

        print(f"임베딩 모델 설정: {model_name}")
        print(f"임베딩 모델 설정 완료")

    def encode(
        self,
        texts: Union[str, List[str]],
        max_length: int = 8191,  # text-embedding-3-large의 최대 토큰 수
        normalize: bool = True
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        텍스트를 임베딩 벡터로 변환

        Args:
            texts: 단일 텍스트 또는 텍스트 리스트
            max_length: 최대 토큰 길이 (사용하지 않음, API가 자동 처리)
            normalize: L2 정규화 여부 (OpenAI 임베딩은 이미 정규화됨)

        Returns:
            임베딩 벡터 (단일 텍스트면 1D array, 리스트면 2D array)
        """
        # 단일 텍스트를 리스트로 변환
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]

        # OpenAI API 호출
        response = self.client.embeddings.create(
            input=texts,
            model=self.model_name
        )

        # 임베딩 추출
        embeddings = [np.array(item.embedding) for item in response.data]
        embeddings = np.array(embeddings)

        # 단일 텍스트면 1D 배열 반환
        if is_single:
            return embeddings[0]

        return embeddings

    def get_embedding_dim(self) -> int:
        """
        임베딩 벡터 차원 반환

        Returns:
            임베딩 차원 (text-embedding-3-large: 3072)
        """
        return self._embedding_dim

    def similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        두 임베딩 간 코사인 유사도 계산

        Args:
            embedding1: 첫 번째 임베딩
            embedding2: 두 번째 임베딩

        Returns:
            코사인 유사도 (0~1)
        """
        # OpenAI 임베딩은 이미 정규화되어 있으므로 내적만 계산
        return float(np.dot(embedding1, embedding2))

    def batch_encode(
        self,
        texts: List[str],
        batch_size: int = 2048,  # OpenAI API는 한 번에 최대 2048개 처리
        max_length: int = 8191,
        normalize: bool = True,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        대량의 텍스트를 배치로 임베딩

        Args:
            texts: 텍스트 리스트
            batch_size: 배치 크기 (최대 2048)
            max_length: 최대 토큰 길이 (사용하지 않음)
            normalize: L2 정규화 여부 (사용하지 않음)
            show_progress: 진행률 표시 여부

        Returns:
            임베딩 배열 [num_texts, embedding_dim]
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.encode(batch, max_length, normalize)
            all_embeddings.append(embeddings)

            if show_progress:
                print(f"  임베딩 진행: {min(i + batch_size, len(texts))}/{len(texts)}")

        return np.vstack(all_embeddings)


# 싱글톤 인스턴스 (전역에서 재사용)
_embedding_service_instance = None


def get_embedding_service(
    model_name: str = "text-embedding-3-large",
    device: str = None
) -> EmbeddingService:
    """
    임베딩 서비스 싱글톤 인스턴스 반환

    Args:
        model_name: 모델 이름 (기본값: text-embedding-3-large)
        device: 디바이스 (사용하지 않음)

    Returns:
        EmbeddingService 인스턴스
    """
    global _embedding_service_instance

    if _embedding_service_instance is None:
        _embedding_service_instance = EmbeddingService(
            model_name=model_name,
            device=device
        )

    return _embedding_service_instance
