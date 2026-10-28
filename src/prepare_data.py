"""Prepare the pinned public sources for TRAYECTORIA-UCI.

Usage: python -m src.prepare_data --root .
Only Python's standard library is required. Existing files are verified and
never overwritten. Missing files are installed atomically after SHA-256 checks.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import zipfile


MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024
MAX_CSV_BYTES = 16 * 1024 * 1024
USER_AGENT = "TRAYECTORIA-UCI/1.0 (academic source reproduction)"


class DataPreparationError(RuntimeError):
    """A source cannot be reproduced with the declared integrity."""


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _target(root: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise DataPreparationError("El manifiesto debe contener rutas relativas.")
    result = (root / candidate).resolve()
    if not result.is_relative_to(root):
        raise DataPreparationError("La ruta del manifiesto sale del directorio del proyecto.")
    return result


def _expected_hash(value: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise DataPreparationError("Falta un SHA-256 válido en el manifiesto.")
    try:
        int(value, 16)
    except ValueError as exc:
        raise DataPreparationError("El SHA-256 del manifiesto no es hexadecimal.") from exc
    return value.lower()


def _verify_existing(path: Path, expected: str) -> bool:
    if not path.exists():
        return False
    if not path.is_file():
        raise DataPreparationError(f"La ruta existe y no es un archivo: {path.name}.")
    actual = sha256_file(path)
    if actual != expected:
        raise DataPreparationError(
            f"SHA-256 diferente en {path.name}; se conserva el archivo existente. "
            f"Esperado: {expected}; obtenido: {actual}."
        )
    return True


def _download(url: str, attempts: int, timeout: float) -> bytes:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise DataPreparationError("Se requiere una URL HTTPS pública sin credenciales.")
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=timeout) as response:
                if urlparse(response.geturl()).scheme != "https":
                    raise DataPreparationError("La descarga se redirigió fuera de HTTPS.")
                declared = response.headers.get("Content-Length")
                if declared and int(declared) > MAX_DOWNLOAD_BYTES:
                    raise DataPreparationError("La descarga supera el límite de 64 MiB.")
                content = response.read(MAX_DOWNLOAD_BYTES + 1)
                if len(content) > MAX_DOWNLOAD_BYTES:
                    raise DataPreparationError("La descarga supera el límite de 64 MiB.")
                return content
        except HTTPError as exc:
            last_error = exc
            if exc.code not in {408, 429, 500, 502, 503, 504}:
                break
        except (URLError, TimeoutError, OSError) as exc:
            last_error = exc
        if attempt + 1 < attempts:
            time.sleep(min(2**attempt, 4))
    raise DataPreparationError(
        f"No fue posible descargar desde {parsed.hostname}: {last_error}."
    )


def _install(path: Path, content: bytes, expected: str) -> None:
    actual = sha256_bytes(content)
    if actual != expected:
        raise DataPreparationError(
            f"SHA-256 diferente para {path.name}; la descarga no se instala. "
            f"Esperado: {expected}; obtenido: {actual}."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".prepare-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        # A second process may have created the target during the download.
        if _verify_existing(path, expected):
            return
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _csv_from_uci_archive(content: bytes, expected: str) -> bytes:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            members = [item for item in archive.infolist()
                       if not item.is_dir() and item.filename.lower().endswith(".csv")]
            if not members:
                raise DataPreparationError("El archivo UCI no contiene un CSV.")
            for item in members:
                if item.file_size > MAX_CSV_BYTES:
                    continue
                # Read bytes directly: no ZIP paths are extracted to disk.
                with archive.open(item) as stream:
                    candidate = stream.read(MAX_CSV_BYTES + 1)
                if len(candidate) <= MAX_CSV_BYTES and sha256_bytes(candidate) == expected:
                    return candidate
    except (zipfile.BadZipFile, RuntimeError, EOFError) as exc:
        raise DataPreparationError("La descarga de UCI no es un ZIP legible.") from exc
    raise DataPreparationError(
        "Ningún CSV del archivo UCI coincide con csv_sha256 de data/dataset.json."
    )


def prepare(root: Path, *, attempts: int = 3, timeout: float = 20,
            verify_only: bool = False, emit: Callable[[str], None] = print) -> list[dict]:
    """Verify or acquire the dataset and three documents, without credentials."""
    root = root.resolve()
    if not 1 <= attempts <= 5 or not 0 < timeout <= 60:
        raise ValueError("attempts debe estar entre 1 y 5; timeout, entre 0 y 60 segundos.")
    dataset = json.loads((root / "data/dataset.json").read_text(encoding="utf-8"))
    sources = json.loads((root / "data/sources.json").read_text(encoding="utf-8"))
    results: list[dict] = []
    entries = [("uci_697", dataset["csv_path"], dataset["csv_sha256"], None)]
    entries += [(document["id"], document["relative_path"], document["sha256"], document)
                for document in sources["documents"]]
    for source_id, relative_path, checksum, document in entries:
        try:
            path = _target(root, relative_path)
            expected = _expected_hash(checksum)
            if _verify_existing(path, expected):
                status = "verificado"
            elif verify_only:
                raise DataPreparationError(f"Falta {relative_path}; ejecute sin --verify-only.")
            elif document is None:
                archive = _download(dataset["download_url"], attempts, timeout)
                _install(path, _csv_from_uci_archive(archive, expected), expected)
                status = "descargado y verificado"
            else:
                urls = list(dict.fromkeys(filter(None, [document.get("download_url"), document.get("url")])))
                errors: list[str] = []
                for url in urls:
                    try:
                        content = _download(url, attempts, timeout)
                        if not content.startswith(b"%PDF-"):
                            raise DataPreparationError("El recurso descargado no es un PDF.")
                        _install(path, content, expected)
                        break
                    except DataPreparationError as exc:
                        errors.append(str(exc))
                else:
                    raise DataPreparationError(" | ".join(errors) or "No hay URL de descarga.")
                status = "descargado y verificado"
            results.append({"id": source_id, "path": relative_path, "status": status})
            emit(f"OK {relative_path}: {status}.")
        except (DataPreparationError, OSError, ValueError) as exc:
            results.append({"id": source_id, "path": relative_path, "status": "error", "error": str(exc)})
            emit(f"ERROR {relative_path}: {exc}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."), help="Directorio raíz del proyecto.")
    parser.add_argument("--verify-only", action="store_true", help="Verificar archivos existentes sin acceso a red.")
    parser.add_argument("--attempts", type=int, choices=range(1, 6), default=3,
                        help="Intentos por URL; predeterminado: 3.")
    parser.add_argument("--timeout", type=float, default=20, help="Tiempo máximo de espera por conexión, hasta 60 s.")
    args = parser.parse_args()
    try:
        results = prepare(args.root, attempts=args.attempts, timeout=args.timeout, verify_only=args.verify_only)
    except (OSError, ValueError, KeyError, DataPreparationError) as exc:
        parser.exit(2, f"ERROR de configuración: {exc}\n")
    failed = sum(item["status"] == "error" for item in results)
    print(f"Fuentes verificadas: {len(results) - failed}/{len(results)}.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
