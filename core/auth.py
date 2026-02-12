"""Upbit JWT 인증 모듈. 주문 요청 시 query hash 포함 토큰 생성."""

from __future__ import annotations

import hashlib
import uuid
from urllib.parse import urlencode, unquote

import jwt


class UpbitAuth:
    """Upbit API 인증 토큰 생성."""

    def __init__(self, access_key: str, secret_key: str) -> None:
        self._access_key = access_key
        self._secret_key = secret_key

    def _base_payload(self) -> dict:
        return {
            "access_key": self._access_key,
            "nonce": str(uuid.uuid4()),
        }

    def create_token(self, query: dict | None = None) -> str:
        """JWT 토큰 생성. query가 있으면 해시 포함."""
        payload = self._base_payload()

        if query:
            query_string = unquote(urlencode(query, doseq=True))
            h = hashlib.sha512()
            h.update(query_string.encode("utf-8"))
            payload["query_hash"] = h.hexdigest()
            payload["query_hash_alg"] = "SHA512"

        return jwt.encode(payload, self._secret_key, algorithm="HS256")

    def get_auth_header(self, query: dict | None = None) -> dict[str, str]:
        """Authorization 헤더 반환."""
        token = self.create_token(query)
        return {"Authorization": f"Bearer {token}"}
