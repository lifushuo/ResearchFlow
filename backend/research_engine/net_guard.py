"""代理可控 URL 的 SSRF 防护，见 docs/architecture/06-security.md §3。

模型可能受不可信网页影响，因此读取地址也视为不可信。
检查协议和端口，解析全部 IP 并拒绝私网、回环、链路本地地址，再固定已验证 IP 连接以防 DNS 重绑定。"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

_ALLOWED_SCHEMES = {"http", "https"}
_ALLOWED_PORTS = {80, 443, 8080, 8443}


class SSRFBlocked(Exception):
    """URL 不符合 SSRF 策略时抛出。"""


def _ip_is_forbidden(ip: str) -> bool:
    addr = ipaddress.ip_address(ip)
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
        # CGNAT 100.64.0.0/10
        or (addr.version == 4 and addr in ipaddress.ip_network("100.64.0.0/10"))
    )


def validate_url(url: str) -> list[str]:
    """按 SSRF 策略校验 URL，通过则返回可连接的 IP 列表，否则抛 SSRFBlocked。

    每次重定向都需重新校验，不能只检查最初地址。"""
    if not url:
        raise SSRFBlocked("empty url")

    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise SSRFBlocked(f"scheme '{parsed.scheme}' not allowed")
    if parsed.username or parsed.password:
        raise SSRFBlocked("userinfo in URL not allowed")

    host = parsed.hostname
    if not host:
        raise SSRFBlocked("no host in URL")

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if port not in _ALLOWED_PORTS:
        raise SSRFBlocked(f"port {port} not allowed")

    # 先检查字面 IP，禁止范围无需 DNS 解析即可拒绝。
    try:
        ipaddress.ip_address(host)
        if _ip_is_forbidden(host):
            raise SSRFBlocked(f"host IP {host} is in a forbidden range")
        return [host]
    except ValueError:
        pass  # 不是字面 IP，继续解析域名。

    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise SSRFBlocked(f"DNS resolution failed: {e}") from e

    resolved = {info[4][0] for info in infos}
    if not resolved:
        raise SSRFBlocked("no addresses resolved")
    for ip in resolved:
        if _ip_is_forbidden(ip):
            raise SSRFBlocked(f"resolved IP {ip} is in a forbidden range")
    return sorted(resolved)
