"""
Marketplace client integrations.
Supports Shutterstock, OpenSea, Magic Eden, and Adobe Stock SFTP.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AssetUpload:
    """Asset upload result."""
    success: bool
    asset_id: Optional[str] = None
    url: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseMarketplaceClient(ABC):
    """Abstract base for marketplace clients."""

    def __init__(self, api_key: str, name: str = "marketplace"):
        self.api_key = api_key
        self.name = name
        self.logger = logging.getLogger(f"marketplace.{name}")

    @abstractmethod
    def authenticate(self) -> bool:
        """Verify API credentials."""
        pass

    @abstractmethod
    def upload(self, file_path: Path, metadata: Dict) -> AssetUpload:
        """Upload asset to marketplace."""
        pass

    @abstractmethod
    def list_assets(self, limit: int = 100) -> List[Dict]:
        """List uploaded assets."""
        pass


class ShutterstockClient(BaseMarketplaceClient):
    """
    Shutterstock contributor API client.

    Uses the Shutterstock Python SDK for uploads and management.

    Example:
        >>> client = ShutterstockClient(api_key="...")
        >>> result = client.upload(Path("image.png"), {"title": "..."})
    """

    def __init__(self, api_key: str, secret: Optional[str] = None):
        super().__init__(api_key, "shutterstock")
        self.secret = secret
        self._client = None

    def authenticate(self) -> bool:
        """Verify Shutterstock API credentials."""
        try:
            import httpx
            response = httpx.get(
                "https://api.shutterstock.com/v2/user",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Auth failed: {e}")
            return False

    def upload(self, file_path: Path, metadata: Dict) -> AssetUpload:
        """Upload image to Shutterstock."""
        try:
            import httpx

            with open(file_path, "rb") as f:
                files = {"file": (file_path.name, f, "image/png")}
                headers = {"Authorization": f"Bearer {self.api_key}"}

                response = httpx.post(
                    "https://api.shutterstock.com/v2/images",
                    headers=headers,
                    files=files,
                    data=metadata,
                    timeout=120.0,
                )

            if response.status_code in (200, 201):
                data = response.json()
                return AssetUpload(
                    success=True,
                    asset_id=data.get("id"),
                    metadata=data,
                )
            else:
                return AssetUpload(
                    success=False,
                    error=f"HTTP {response.status_code}: {response.text}"
                )

        except Exception as e:
            self.logger.error(f"Upload failed: {e}")
            return AssetUpload(success=False, error=str(e))

    def list_assets(self, limit: int = 100) -> List[Dict]:
        """List contributor's uploaded assets."""
        try:
            import httpx
            response = httpx.get(
                "https://api.shutterstock.com/v2/contributor/assets",
                headers={"Authorization": f"Bearer {self.api_key}"},
                params={"per_page": limit},
            )
            if response.status_code == 200:
                return response.json().get("data", [])
            return []
        except Exception as e:
            self.logger.error(f"List failed: {e}")
            return []


class OpenSeaClient(BaseMarketplaceClient):
    """
    OpenSea NFT marketplace client.

    Note: OpenSea doesn't have direct upload API.
    Requires smart contract deployment or browser automation.
    """

    def __init__(self, api_key: str, wallet_address: Optional[str] = None):
        super().__init__(api_key, "opensea")
        self.wallet_address = wallet_address
        self.base_url = "https://api.opensea.io/api/v2"

    def authenticate(self) -> bool:
        """Verify OpenSea API key."""
        try:
            import httpx
            response = httpx.get(
                f"{self.base_url}/listings/collection/boredapeyachtclub/all",
                headers={"X-API-KEY": self.api_key},
            )
            return response.status_code == 200
        except Exception:
            return False

    def upload(self, file_path: Path, metadata: Dict) -> AssetUpload:
        """
        OpenSea doesn't support direct upload.
        Must use smart contract or Playwright automation.
        """
        self.logger.warning(
            "OpenSea requires smart contract deployment or browser automation. "
            "Use deploy_nft() or browser_upload() instead."
        )
        return AssetUpload(
            success=False,
            error="Direct upload not supported. Use smart contract or browser automation."
        )

    def list_assets(self, limit: int = 100) -> List[Dict]:
        """List NFTs owned by wallet."""
        if not self.wallet_address:
            return []

        try:
            import httpx
            response = httpx.get(
                f"{self.base_url}/chain/ethereum/account/{self.wallet_address}/nfts",
                headers={"X-API-KEY": self.api_key},
                params={"limit": limit},
            )
            if response.status_code == 200:
                return response.json().get("nfts", [])
            return []
        except Exception as e:
            self.logger.error(f"List failed: {e}")
            return []

    def get_collection_stats(self, collection_slug: str) -> Dict:
        """Get stats for an NFT collection."""
        try:
            import httpx
            response = httpx.get(
                f"{self.base_url}/collections/{collection_slug}/stats",
                headers={"X-API-KEY": self.api_key},
            )
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            self.logger.error(f"Stats fetch failed: {e}")
            return {}


class MagicEdenClient(BaseMarketplaceClient):
    """
    Magic Eden NFT marketplace client (Solana).

    Requires Solana wallet and Metaplex for minting.
    """

    def __init__(self, api_key: str, wallet_keypair: Optional[str] = None):
        super().__init__(api_key, "magiceden")
        self.wallet_keypair = wallet_keypair
        self.base_url = "https://api-mainnet.magiceden.dev/v2"

    def authenticate(self) -> bool:
        """Verify Magic Eden API access."""
        try:
            import httpx
            response = httpx.get(
                f"{self.base_url}/popular_collections",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            return response.status_code == 200
        except Exception:
            return False

    def upload(self, file_path: Path, metadata: Dict) -> AssetUpload:
        """
        Magic Eden requires Metaplex minting.
        Use mint_nft() with Solana SDK.
        """
        self.logger.warning(
            "Magic Eden requires Metaplex minting. "
            "Use mint_nft() with Solana SDK."
        )
        return AssetUpload(
            success=False,
            error="Direct upload not supported. Use Metaplex minting."
        )

    def list_assets(self, limit: int = 100) -> List[Dict]:
        """List NFTs by wallet (requires wallet address)."""
        return []

    def get_listings(
        self,
        collection_symbol: str,
        limit: int = 20,
    ) -> List[Dict]:
        """Get active listings for a collection."""
        try:
            import httpx
            response = httpx.get(
                f"{self.base_url}/collections/{collection_symbol}/listings",
                params={"limit": limit},
            )
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            self.logger.error(f"Listings fetch failed: {e}")
            return []

    def get_collection_stats(self, collection_symbol: str) -> Dict:
        """Get collection statistics."""
        try:
            import httpx
            response = httpx.get(
                f"{self.base_url}/collections/{collection_symbol}/stats",
            )
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            self.logger.error(f"Stats fetch failed: {e}")
            return {}


class AdobeStockSFTPClient:
    """
    Adobe Stock contributor SFTP client.

    Note: Adobe Stock doesn't have public upload API.
    SFTP access requires "qualified account" status.

    Example:
        >>> client = AdobeStockSFTPClient(
        ...     host="sftp.contributor.adobe.com",
        ...     username="your_username",
        ...     password="your_password"
        ... )
        >>> client.upload_file(Path("image.png"))
    """

    def __init__(
        self,
        host: str = "sftp.contributor.adobe.com",
        username: Optional[str] = None,
        password: Optional[str] = None,
        port: int = 22,
    ):
        self.host = host
        self.username = username
        self.password = password
        self.port = port
        self._sftp = None
        self.logger = logging.getLogger("marketplace.adobestock")

    def connect(self) -> bool:
        """Establish SFTP connection."""
        try:
            import paramiko

            transport = paramiko.Transport((self.host, self.port))
            transport.connect(username=self.username, password=self.password)
            self._sftp = paramiko.SFTPClient.from_transport(transport)
            self.logger.info("Connected to Adobe Stock SFTP")
            return True
        except Exception as e:
            self.logger.error(f"SFTP connection failed: {e}")
            return False

    def upload_file(
        self,
        local_path: Path,
        remote_dir: str = "/upload",
    ) -> bool:
        """Upload file via SFTP."""
        if not self._sftp:
            if not self.connect():
                return False

        try:
            remote_path = f"{remote_dir}/{local_path.name}"
            self._sftp.put(str(local_path), remote_path)
            self.logger.info(f"Uploaded: {local_path.name}")
            return True
        except Exception as e:
            self.logger.error(f"Upload failed: {e}")
            return False

    def upload_batch(
        self,
        file_paths: List[Path],
        remote_dir: str = "/upload",
    ) -> Dict[str, bool]:
        """Upload multiple files."""
        results = {}
        for path in file_paths:
            results[str(path)] = self.upload_file(path, remote_dir)
        return results

    def close(self):
        """Close SFTP connection."""
        if self._sftp:
            self._sftp.close()
            self._sftp = None
