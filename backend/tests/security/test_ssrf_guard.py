"""Regression: SSRF guard blocks internal/metadata targets (docs/06 §3, docs/08 §3)."""

import socket

import pytest

from research_engine.net_guard import SSRFBlocked, validate_url


@pytest.fixture(autouse=True)
def fixed_dns(monkeypatch):
    # 控制 DNS 边界，不替换被测防护；禁网时仍需验证解析结果的公私网判定，而非依赖外部 DNS。
    addresses = {
        "example.com": ["93.184.216.34"],
        "metadata.google.internal": ["169.254.169.254"],
        "localhost": ["127.0.0.1"],
    }

    def resolve(host, port, **kwargs):
        assert kwargs["proto"] == socket.IPPROTO_TCP
        return [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, port))
            for ip in addresses[host]
        ]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    return addresses


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",  # AWS IMDS
        "http://metadata.google.internal/",  # GCP metadata (resolves internal)
        "http://127.0.0.1:6379/",  # local Redis
        "http://localhost:8000/",  # local API
        "http://10.0.0.5/",  # RFC1918
        "http://192.168.1.1/",  # RFC1918
        "http://[::1]/",  # IPv6 loopback
        "http://100.64.0.1/",  # CGNAT
        "ftp://example.com/",  # bad scheme
        "http://user:pass@example.com/",  # userinfo
        "http://example.com:22/",  # bad port
    ],
)
def test_forbidden_urls_are_blocked(url):
    with pytest.raises(SSRFBlocked):
        validate_url(url)


def test_public_https_is_allowed():
    ips = validate_url("https://example.com/")
    assert ips == ["93.184.216.34"]


def test_a_public_address_cannot_hide_a_private_dns_answer(fixed_dns):
    fixed_dns["example.com"].append("127.0.0.1")
    with pytest.raises(SSRFBlocked):
        validate_url("https://example.com/")


def test_dns_failure_stays_blocked(monkeypatch):
    def fail(*args, **kwargs):
        raise socket.gaierror("offline DNS failure")

    monkeypatch.setattr(socket, "getaddrinfo", fail)
    with pytest.raises(SSRFBlocked, match="DNS resolution failed"):
        validate_url("https://example.com/")
