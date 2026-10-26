"""Non-paid tests for account selection, failover and cookie isolation.

Run with:  .venv\\Scripts\\python.exe -m unittest test_account_rotation -v
"""
import tempfile
import time
import unittest
from pathlib import Path

import engine as E

A = "a" * 32
B = "b" * 32
C = "c" * 32
D = "d" * 32  # default (non-container) Firefox session
UNKNOWN = "e" * 32

_ORIGINAL_GET_SESSION_ID = E.get_session_id


def setUpModule():
    # Tests run without a live Firefox profile: the registry is the only source
    # of accounts, so the real default session must not sneak into the list.
    E.get_session_id = lambda cookies=None: None


def tearDownModule():
    E.get_session_id = _ORIGINAL_GET_SESSION_ID


def credit_info(gift: int) -> dict:
    return {
        "gift_credit": gift,
        "vip_credit": 0,
        "purchase_credit": 0,
        "generations_remaining": gift // E.COST_PER_IMAGE,
        "status": "active" if gift >= E.COST_PER_IMAGE else ("ready" if gift else "exhausted"),
        "shark_rejected": False,
    }


def make_engine(accounts: list[str], balances: dict[str, int]) -> E.DreaminaEngine:
    eng = E.DreaminaEngine(D, Path(tempfile.gettempdir()) / "dreamina_engine_tests")
    eng._accounts = [
        {"sessionid": sid, "name": sid[:4], "cookies": [{"name": "sessionid", "value": sid}]}
        for sid in accounts
    ]
    eng._accounts_loaded_at = time.time()

    async def fake_credits(session_id=None, force_refresh=False):
        info = credit_info(balances[session_id])
        eng._remember_credits(session_id, info)
        return info

    eng.get_credits_info = fake_credits
    return eng


class SelectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_lru_prefers_least_recently_used(self):
        eng = make_engine([A, B], {A: 120, B: 120})
        first, _ = await eng._select_account()
        second, _ = await eng._select_account()
        third, _ = await eng._select_account()
        self.assertEqual([first, second, third], [A, B, A])

    async def test_zero_balance_account_is_skipped(self):
        eng = make_engine([A, B], {A: 0, B: 120})
        sid, info = await eng._select_account()
        self.assertEqual(sid, B)
        self.assertEqual(info["gift_credit"], 120)

    async def test_cooled_down_account_is_skipped(self):
        eng = make_engine([A, B], {A: 120, B: 120})
        eng._cooldowns[A] = time.time() + 600
        sid, _ = await eng._select_account()
        self.assertEqual(sid, B)

    async def test_activation_only_when_nobody_is_funded(self):
        eng = make_engine([A, B], {A: 0, B: 0})
        activated = []

        async def fake_activate(sid):
            activated.append(sid)
            return credit_info(120)

        eng._activate_account = fake_activate
        sid, info = await eng._select_account()
        self.assertEqual(sid, A)
        self.assertEqual(info["gift_credit"], 120)
        self.assertEqual(activated, [A])

    async def test_funded_account_never_triggers_activation(self):
        eng = make_engine([A], {A: 120})

        async def boom(sid):
            raise AssertionError("activation must not run while credits exist")

        eng._activate_account = boom
        sid, _ = await eng._select_account()
        self.assertEqual(sid, A)

    async def test_all_empty_raises_after_two_activation_attempts(self):
        eng = make_engine([A, B, C], {A: 0, B: 0, C: 0})
        activated = []

        async def fake_activate(sid):
            activated.append(sid)
            return credit_info(0)

        eng._activate_account = fake_activate
        with self.assertRaises(E.InsufficientCreditsError):
            await eng._select_account()
        self.assertEqual(activated, [A, B])


class ChargeTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_charges_exactly_once(self):
        balances = {A: 120}
        eng = make_engine([A], balances)
        eng._remember_credits(A, credit_info(120))
        calls = []

        async def fake_generate(prompt, size=None, session_id=None):
            calls.append(session_id)
            balances[A] = 96  # the studio charges the run
            return ["one.png"]

        eng._generate_image_internal = fake_generate
        paths = await eng._generate_once_locked("prompt")
        self.assertEqual(paths, ["one.png"])
        self.assertEqual(calls, [A])
        self.assertEqual(eng._credits_by_sid[A]["gift_credit"], 96)
        self.assertEqual(eng._credits_by_sid[A]["generations_remaining"], 4)

    async def test_promo_grant_is_not_double_counted(self):
        balances = {A: 120}
        eng = make_engine([A], balances)
        eng._remember_credits(A, credit_info(120))

        async def fake_generate(prompt, size=None, session_id=None):
            balances[A] = 191  # 120 - 24 charged + 95 promo credits
            return ["one.png"]

        eng._generate_image_internal = fake_generate
        await eng._generate_once_locked("prompt")
        self.assertEqual(eng._credits_by_sid[A]["gift_credit"], 191)

    async def test_failure_fails_over_and_charges_only_the_second_account(self):
        balances = {A: 120, B: 120}
        eng = make_engine([A, B], balances)
        eng._remember_credits(A, credit_info(120))
        eng._remember_credits(B, credit_info(120))
        picks = iter([(A, credit_info(120)), (B, credit_info(120))])

        async def fake_select():
            return next(picks)

        async def fake_generate(prompt, size=None, session_id=None):
            if session_id == A:
                raise RuntimeError("studio blew up")
            balances[B] = 96
            return ["second.png"]

        eng._select_account = fake_select
        eng._generate_image_internal = fake_generate
        before = time.time()
        paths = await eng._generate_once_locked("prompt")
        self.assertEqual(paths, ["second.png"])
        self.assertEqual(eng._credits_by_sid[A]["gift_credit"], 120)
        self.assertEqual(eng._credits_by_sid[B]["gift_credit"], 96)
        self.assertGreaterEqual(eng._cooldowns[A], before + 7190)

    async def test_empty_account_gets_short_cooldown(self):
        eng = make_engine([A], {A: 120})

        async def fake_pick():
            return A, credit_info(120)

        async def fake_select():
            raise E.InsufficientCreditsError("nothing else", balance=0)

        async def fake_generate(prompt, size=None, session_id=None):
            raise E.InsufficientCreditsError("empty", balance=0)

        eng._pick_funded_account = fake_pick
        eng._select_account = fake_select
        eng._generate_image_internal = fake_generate
        eng._remember_credits(A, credit_info(120))
        before = time.time()
        with self.assertRaises(E.InsufficientCreditsError):
            await eng._generate_once_locked("prompt")
        self.assertLessEqual(eng._cooldowns[A] - before, 601)
        self.assertGreaterEqual(eng._cooldowns[A] - before, 590)

    async def test_charged_failure_never_retries(self):
        balances = {A: 120, B: 120}
        eng = make_engine([A, B], balances)
        eng._remember_credits(A, credit_info(120))
        eng._remember_credits(B, credit_info(120))
        calls = []

        async def fake_pick():
            return A, credit_info(120)

        async def fake_generate(prompt, size=None, session_id=None):
            calls.append(session_id)
            balances[A] = 96
            raise E.GenerationChargedError("rendered but download failed")

        eng._pick_funded_account = fake_pick
        eng._generate_image_internal = fake_generate
        before = time.time()
        with self.assertRaises(E.GenerationChargedError):
            await eng._generate_once_locked("prompt")
        self.assertEqual(calls, [A])
        self.assertEqual(eng._credits_by_sid[A]["gift_credit"], 96)
        self.assertEqual(eng._credits_by_sid[B]["gift_credit"], 120)
        self.assertGreaterEqual(eng._cooldowns[A], before + 590)

    async def test_stale_cache_is_rechecked_before_charging(self):
        eng = make_engine([A], {A: 0})

        async def fake_select():
            return A, credit_info(120)  # stale cached value

        eng._select_account = fake_select
        with self.assertRaises(E.InsufficientCreditsError):
            await eng._pick_funded_account()
        self.assertGreater(eng._cooldowns[A], time.time())


class IsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_session_gets_only_its_own_jar(self):
        eng = make_engine([A, B], {A: 120, B: 120})
        jar_a = await self._cookies(eng, A)
        jar_b = await self._cookies(eng, B)
        self.assertEqual(jar_a[0]["value"], A)
        self.assertEqual(jar_b[0]["value"], B)

    async def test_unknown_session_never_falls_back_to_the_default_jar(self):
        eng = make_engine([A], {A: 120})
        original = E.get_session_id
        E.get_session_id = lambda cookies=None: D
        try:
            self.assertIsNone(eng._get_cookies_for_session(UNKNOWN))
        finally:
            E.get_session_id = original

    async def test_only_the_default_session_may_reload_live_firefox_cookies(self):
        eng = make_engine([A], {A: 120})
        original = E.get_session_id
        E.get_session_id = lambda cookies=None: D
        try:
            self.assertTrue(eng._can_refresh_from_live_firefox(D))
            self.assertFalse(eng._can_refresh_from_live_firefox(A))
            self.assertFalse(eng._can_refresh_from_live_firefox(""))
        finally:
            E.get_session_id = original

    async def test_cached_balance_is_per_session(self):
        eng = make_engine([A, B], {A: 120, B: 0})
        del eng.get_credits_info  # fall back to the real implementation
        eng._remember_credits(A, credit_info(120))
        eng._remember_credits(B, credit_info(0))

        def boom(sid):
            raise AssertionError("cache must answer without a network call")

        eng._get_cookies_for_session = boom
        info_a = await eng.get_credits_info(session_id=A)
        info_b = await eng.get_credits_info(session_id=B)
        self.assertEqual(info_a["gift_credit"], 120)
        self.assertEqual(info_b["gift_credit"], 0)

    async def test_deduction_touches_one_account_only(self):
        eng = make_engine([A, B], {A: 120, B: 120})
        eng._remember_credits(A, credit_info(120))
        eng._remember_credits(B, credit_info(120))
        eng._deduct_credits(A)
        self.assertEqual(eng._credits_by_sid[A]["gift_credit"], 96)
        self.assertEqual(eng._credits_by_sid[B]["gift_credit"], 120)

    async def _cookies(self, eng, sid):
        original = E.get_session_id
        E.get_session_id = lambda cookies=None: D
        try:
            return eng._get_cookies_for_session(sid)
        finally:
            E.get_session_id = original


if __name__ == "__main__":
    unittest.main(verbosity=2)
