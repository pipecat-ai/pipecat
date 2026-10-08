#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.transports.daily.utils import (
    DailyRESTHelper,
    DailySIPClientParams,
)

API_URL = "https://api.daily.co/v1"

CLIENT_RESPONSE = {
    "username": "support.1",
    "domain": "mydomain.sip-us.daily.co",
    "sip_uri": "sip:support.1@mydomain.sip-us.daily.co",
    "expires_at": None,
}


class FakeResponse:
    def __init__(self, status=200, payload=None, text=""):
        self.status = status
        self._payload = payload
        self._text = text

    async def json(self):
        return self._payload

    async def text(self):
        return self._text


class FakeSession:
    """Records requests and plays back a canned response, aiohttp-style."""

    def __init__(self, response: FakeResponse):
        self.response = response
        self.requests: list[dict] = []

    def _request(self, method, url, **kwargs):
        self.requests.append({"method": method, "url": url, **kwargs})
        response = self.response

        class _Ctx:
            async def __aenter__(self):
                return response

            async def __aexit__(self, *exc):
                return False

        return _Ctx()

    def post(self, url, **kwargs):
        return self._request("POST", url, **kwargs)

    def get(self, url, **kwargs):
        return self._request("GET", url, **kwargs)

    def delete(self, url, **kwargs):
        return self._request("DELETE", url, **kwargs)


def make_helper(response: FakeResponse) -> tuple[DailyRESTHelper, FakeSession]:
    session = FakeSession(response)
    helper = DailyRESTHelper(
        daily_api_key="test-key", daily_api_url=API_URL, aiohttp_session=session
    )
    return helper, session


class TestCreateSIPClient(unittest.IsolatedAsyncioTestCase):
    async def test_create_returns_password(self):
        payload = {**CLIENT_RESPONSE, "password": "s3cretpass12"}
        helper, session = make_helper(FakeResponse(payload=payload))

        client = await helper.create_sip_client(DailySIPClientParams(username="support.1"))

        request = session.requests[0]
        self.assertEqual(request["method"], "POST")
        self.assertEqual(request["url"], f"{API_URL}/sip-clients")
        self.assertEqual(request["headers"], {"Authorization": "Bearer test-key"})
        # Unset optionals must not reach the wire.
        self.assertEqual(request["json"], {"username": "support.1"})
        self.assertEqual(client.password, "s3cretpass12")
        self.assertEqual(client.sip_uri, "sip:support.1@mydomain.sip-us.daily.co")

    async def test_create_sends_password_and_ttl(self):
        payload = {**CLIENT_RESPONSE, "password": "chosenpass12"}
        helper, session = make_helper(FakeResponse(payload=payload))

        await helper.create_sip_client(
            DailySIPClientParams(
                username="support.1", password="chosenpass12", expires_in_seconds=3600
            )
        )

        self.assertEqual(
            session.requests[0]["json"],
            {"username": "support.1", "password": "chosenpass12", "expires_in_seconds": 3600},
        )

    async def test_create_error_surfaces_body_text(self):
        helper, _ = make_helper(FakeResponse(status=403, text="SIP client limit reached"))

        with self.assertRaises(Exception) as ctx:
            await helper.create_sip_client(DailySIPClientParams(username="support.1"))

        self.assertIn("403", str(ctx.exception))
        self.assertIn("SIP client limit reached", str(ctx.exception))


class TestGetAndListSIPClients(unittest.IsolatedAsyncioTestCase):
    async def test_get_has_no_password(self):
        helper, session = make_helper(FakeResponse(payload=CLIENT_RESPONSE))

        client = await helper.get_sip_client("support.1")

        request = session.requests[0]
        self.assertEqual(request["method"], "GET")
        self.assertEqual(request["url"], f"{API_URL}/sip-clients/support.1")
        self.assertIsNone(client.password)

    async def test_get_missing_raises(self):
        helper, _ = make_helper(FakeResponse(status=404, text="not found"))

        with self.assertRaises(Exception) as ctx:
            await helper.get_sip_client("ghost")

        self.assertIn("404", str(ctx.exception))

    async def test_list_returns_page(self):
        payload = {"sip_clients": [CLIENT_RESPONSE], "total": 1, "limit": 50, "offset": 0}
        helper, session = make_helper(FakeResponse(payload=payload))

        page = await helper.list_sip_clients()

        self.assertEqual(session.requests[0]["params"], {})
        self.assertEqual(page.total, 1)
        self.assertEqual(page.sip_clients[0].username, "support.1")
        self.assertIsNone(page.sip_clients[0].password)

    async def test_list_passes_pagination(self):
        payload = {"sip_clients": [], "total": 0, "limit": 10, "offset": 20}
        helper, session = make_helper(FakeResponse(payload=payload))

        await helper.list_sip_clients(limit=10, offset=20)

        self.assertEqual(session.requests[0]["params"], {"limit": 10, "offset": 20})


class TestDeleteSIPClient(unittest.IsolatedAsyncioTestCase):
    async def test_delete_ok(self):
        helper, session = make_helper(FakeResponse(payload={"status": "ok"}))

        self.assertTrue(await helper.delete_sip_client("support.1"))
        request = session.requests[0]
        self.assertEqual(request["method"], "DELETE")
        self.assertEqual(request["url"], f"{API_URL}/sip-clients/support.1")

    async def test_delete_tolerates_404(self):
        helper, _ = make_helper(FakeResponse(status=404, text="not found"))

        self.assertTrue(await helper.delete_sip_client("ghost"))

    async def test_delete_other_error_raises(self):
        helper, _ = make_helper(FakeResponse(status=500, text="boom"))

        with self.assertRaises(Exception) as ctx:
            await helper.delete_sip_client("support.1")

        self.assertIn("500", str(ctx.exception))


class TestRotateSIPClientPassword(unittest.IsolatedAsyncioTestCase):
    async def test_rotate_returns_new_password(self):
        payload = {**CLIENT_RESPONSE, "password": "newpass12345"}
        helper, session = make_helper(FakeResponse(payload=payload))

        client = await helper.rotate_sip_client_password("support.1")

        request = session.requests[0]
        self.assertEqual(request["url"], f"{API_URL}/sip-clients/support.1/rotate")
        self.assertEqual(request["json"], {})
        self.assertEqual(client.password, "newpass12345")

    async def test_rotate_sends_chosen_password(self):
        payload = {**CLIENT_RESPONSE, "password": "chosenpass12"}
        helper, session = make_helper(FakeResponse(payload=payload))

        await helper.rotate_sip_client_password("support.1", password="chosenpass12")

        self.assertEqual(session.requests[0]["json"], {"password": "chosenpass12"})

    async def test_rotate_missing_raises(self):
        helper, _ = make_helper(FakeResponse(status=404, text="not found"))

        with self.assertRaises(Exception) as ctx:
            await helper.rotate_sip_client_password("ghost")

        self.assertIn("404", str(ctx.exception))
