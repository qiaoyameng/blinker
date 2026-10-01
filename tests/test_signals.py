from __future__ import annotations

import asyncio
import collections.abc as c
import gc
import sys
import typing as t

import pytest

import blinker


def collect_acyclic_refs() -> None:
    # cpython releases these immediately without a collection
    if sys.implementation.name == "pypy":
        gc.collect()


class Sentinel(list):  # type: ignore[type-arg]
    """A signal receipt accumulator."""

    def make_receiver(self, key: t.Any) -> c.Callable[..., t.Any]:
        """Return a generic signal receiver function logging as *key*

        When connected to a signal, appends (key, sender, kw) to the Sentinel.

        """

        def receiver(*sentby: t.Any, **kw: t.Any) -> None:
            self.append((key, sentby[0], kw))

        return receiver


def _test_signal_signals(sender: t.Any) -> None:
    sentinel = Sentinel()
    sig = blinker.Signal()

    connected = sentinel.make_receiver("receiver_connected")
    disconnected = sentinel.make_receiver("receiver_disconnected")
    receiver1 = sentinel.make_receiver("receiver1")
    receiver2 = sentinel.make_receiver("receiver2")

    assert not sig.receiver_connected.receivers
    assert not sig.receiver_disconnected.receivers
    sig.receiver_connected.connect(connected)
    sig.receiver_disconnected.connect(disconnected)

    assert sig.receiver_connected.receivers
    assert not sentinel

    for receiver, weak in [(receiver1, True), (receiver2, False)]:
        sig.connect(receiver, sender=sender, weak=weak)

        expected = (
            "receiver_connected",
            sig,
            {"receiver": receiver, "sender": sender, "weak": weak},
        )

        assert sentinel[-1] == expected

    # disconnect from explicit sender
    sig.disconnect(receiver1, sender=sender)

    expected = ("receiver_disconnected", sig, {"receiver": receiver1, "sender": sender})
    assert sentinel[-1] == expected

    # disconnect from ANY and all senders (implicit disconnect signature)
    sig.disconnect(receiver2)
    assert sentinel[-1] == (
        "receiver_disconnected",
        sig,
        {"receiver": receiver2, "sender": blinker.ANY},
    )


def test_signal_signals_any_sender() -> None:
    _test_signal_signals(blinker.ANY)


def test_signal_signals_strong_sender() -> None:
    _test_signal_signals("squiznart")


def test_signal_weak_receiver_vanishes() -> None:
    # non-edge-case path for weak receivers is exercised in the ANY sender
    # test above.
    sentinel = Sentinel()
    sig = blinker.Signal()

    connected = sentinel.make_receiver("receiver_connected")
    disconnected = sentinel.make_receiver("receiver_disconnected")
    receiver1 = sentinel.make_receiver("receiver1")
    receiver2 = sentinel.make_receiver("receiver2")

    sig.receiver_connected.connect(connected)
    sig.receiver_disconnected.connect(disconnected)

    # explicit disconnect on a weak does emit the signal
    sig.connect(receiver1, weak=True)
    sig.disconnect(receiver1)

    assert len(sentinel) == 2
    assert sentinel[-1][2]["receiver"] is receiver1

    del sentinel[:]
    sig.connect(receiver2, weak=True)
    assert len(sentinel) == 1

    del sentinel[:]  # holds a ref to receiver2
    del receiver2
    collect_acyclic_refs()

    # no disconnect signal is fired
    assert len(sentinel) == 0

    # and everything really is disconnected
    sig.send("abc")
    assert len(sentinel) == 0


def test_signal_signals_weak_sender() -> None:
    sentinel = Sentinel()
    sig = blinker.Signal()

    connected = sentinel.make_receiver("receiver_connected")
    disconnected = sentinel.make_receiver("receiver_disconnected")
    receiver1 = sentinel.make_receiver("receiver1")
    receiver2 = sentinel.make_receiver("receiver2")

    class Sender:
        """A weakref-able object."""

    sig.receiver_connected.connect(connected)
    sig.receiver_disconnected.connect(disconnected)

    sender1 = Sender()
    sig.connect(receiver1, sender=sender1, weak=False)
    # regular disconnect of weak-able sender works fine
    sig.disconnect(receiver1, sender=sender1)

    assert len(sentinel) == 2

    del sentinel[:]
    sender2 = Sender()
    sig.connect(receiver2, sender=sender2, weak=False)

    # force sender2 to go out of scope
    del sender2
    collect_acyclic_refs()

    # no disconnect signal is fired
    assert len(sentinel) == 1

    # and everything really is disconnected
    sig.send("abc")
    assert len(sentinel) == 1


def test_empty_bucket_growth() -> None:
    def senders() -> tuple[int, int]:
        return (
            len(sig._by_sender),
            sum(len(i) for i in sig._by_sender.values()),
        )

    def receivers() -> tuple[int, int]:
        return (
            len(sig._by_receiver),
            sum(len(i) for i in sig._by_receiver.values()),
        )

    sentinel = Sentinel()
    sig = blinker.Signal()
    receiver1 = sentinel.make_receiver("receiver1")
    receiver2 = sentinel.make_receiver("receiver2")

    class Sender:
        """A weakref-able object."""

    sender = Sender()
    sig.connect(receiver1, sender=sender)
    sig.connect(receiver2, sender=sender)

    assert senders() == (1, 2)
    assert receivers() == (2, 2)

    sig.disconnect(receiver1, sender=sender)

    assert senders() == (1, 1)
    assert receivers() == (2, 1)

    sig.disconnect(receiver2, sender=sender)

    assert senders() == (1, 0)
    assert receivers() == (2, 0)

    sig._cleanup_bookkeeping()
    assert senders() == (0, 0)
    assert receivers() == (0, 0)


def test_namespace() -> None:
    ns = blinker.Namespace()
    assert not ns
    s1 = ns.signal("abc")
    assert s1 is ns.signal("abc")
    assert s1 is not ns.signal("def")
    assert "abc" in ns

    del s1
    collect_acyclic_refs()

    assert "def" in ns
    assert "abc" in ns


def test_weak_receiver() -> None:
    sentinel = []

    def received(sender: t.Any, **kw: t.Any) -> None:
        sentinel.append(kw)

    sig = blinker.Signal()

    sig.connect(received, weak=True)

    del received
    collect_acyclic_refs()

    assert not sentinel
    sig.send()
    assert not sentinel
    assert not sig.receivers
    values_are_empty_sets_(sig._by_receiver)
    values_are_empty_sets_(sig._by_sender)


def test_strong_receiver() -> None:
    sentinel = []

    def received(sender: t.Any) -> None:
        sentinel.append(sender)

    fn_id = id(received)

    sig = blinker.Signal()
    sig.connect(received, weak=False)

    del received
    collect_acyclic_refs()

    assert not sentinel
    sig.send()
    assert sentinel
    assert [id(fn) for fn in sig.receivers.values()] == [fn_id]


async def test_async_receiver() -> None:
    sentinel = []

    async def received_async(sender: t.Any) -> None:
        sentinel.append(sender)

    def received(sender: t.Any) -> None:
        sentinel.append(sender)

    def wrapper(func: c.Callable[..., t.Any]) -> c.Callable[..., None]:
        async def inner(*args: t.Any, **kwargs: t.Any) -> None:
            func(*args, **kwargs)

        return inner  # type: ignore[return-value]

    sig = blinker.Signal()
    sig.connect(received)
    sig.connect(received_async)

    await sig.send_async(_sync_wrapper=wrapper)  # type: ignore[arg-type]
    assert len(sentinel) == 2

    with pytest.raises(RuntimeError):
        sig.send()


def test_instancemethod_receiver() -> None:
    sentinel: list[t.Any] = []

    class Receiver:
        def __init__(self, bucket: list[t.Any]) -> None:
            self.bucket = bucket

        def received(self, sender: t.Any) -> None:
            self.bucket.append(sender)

    receiver = Receiver(sentinel)

    sig = blinker.Signal()
    sig.connect(receiver.received)

    assert not sentinel
    sig.send()
    assert sentinel
    del receiver
    collect_acyclic_refs()

    sig.send()
    assert len(sentinel) == 1


def test_filtered_receiver() -> None:
    sentinel = []

    def received(sender: t.Any) -> None:
        sentinel.append(sender)

    sig = blinker.Signal()

    sig.connect(received, 123)

    assert not sentinel
    sig.send()
    assert not sentinel
    sig.send(123)
    assert sentinel == [123]
    sig.send()
    assert sentinel == [123]

    sig.disconnect(received, 123)
    sig.send(123)
    assert sentinel == [123]

    sig.connect(received, 123)
    sig.send(123)
    assert sentinel == [123, 123]

    sig.disconnect(received)
    sig.send(123)
    assert sentinel == [123, 123]


def test_filtered_receiver_weakref() -> None:
    sentinel = []

    def received(sender: t.Any) -> None:
        sentinel.append(sender)

    class Object:
        pass

    obj = Object()

    sig = blinker.Signal()
    sig.connect(received, obj)

    assert not sentinel
    sig.send(obj)
    assert sentinel == [obj]
    del sentinel[:]
    del obj
    collect_acyclic_refs()

    # general index isn't cleaned up
    assert sig.receivers
    # but receiver/sender pairs are
    values_are_empty_sets_(sig._by_receiver)
    values_are_empty_sets_(sig._by_sender)


def test_decorated_receiver() -> None:
    sentinel = []

    class Object:
        pass

    obj = Object()

    sig = blinker.Signal()

    @sig.connect_via(obj)
    def receiver(sender: t.Any, **kw: t.Any) -> None:
        sentinel.append(kw)

    assert not sentinel
    sig.send()
    assert not sentinel
    sig.send(1)
    assert not sentinel
    sig.send(obj)
    assert sig.receivers

    del receiver
    collect_acyclic_refs()
    assert sig.receivers


def test_no_double_send() -> None:
    sentinel = []

    def received(sender: t.Any) -> None:
        sentinel.append(sender)

    sig = blinker.Signal()

    sig.connect(received, 123)
    sig.connect(received)

    assert not sentinel
    sig.send()
    assert sentinel == [None]
    sig.send(123)
    assert sentinel == [None, 123]
    sig.send()
    assert sentinel == [None, 123, None]


def test_has_receivers() -> None:
    def received(_: t.Any) -> None:
        return None

    sig = blinker.Signal()
    assert not sig.has_receivers_for(None)
    assert not sig.has_receivers_for(blinker.ANY)

    sig.connect(received, "xyz")
    assert not sig.has_receivers_for(None)
    assert not sig.has_receivers_for(blinker.ANY)
    assert sig.has_receivers_for("xyz")

    class Object:
        pass

    o = Object()

    sig.connect(received, o)
    assert sig.has_receivers_for(o)

    del received
    collect_acyclic_refs()

    assert not sig.has_receivers_for("xyz")
    assert list(sig.receivers_for("xyz")) == []
    assert list(sig.receivers_for(o)) == []

    sig.connect(lambda sender: None, weak=False)
    assert sig.has_receivers_for("xyz")
    assert sig.has_receivers_for(o)
    assert sig.has_receivers_for(None)
    assert sig.has_receivers_for(blinker.ANY)
    assert sig.has_receivers_for("xyz")


def test_instance_doc() -> None:
    sig = blinker.Signal(doc="x")
    assert sig.__doc__ == "x"

    sig = blinker.Signal("x")
    assert sig.__doc__ == "x"


def test_named_blinker() -> None:
    sig = blinker.NamedSignal("squiznart")
    assert "squiznart" in repr(sig)


def test_mute_signal() -> None:
    sentinel = []

    def received(sender: t.Any) -> None:
        sentinel.append(sender)

    sig = blinker.Signal()
    sig.connect(received)

    sig.send(123)
    assert 123 in sentinel

    with sig.muted():
        sig.send(456)
    assert 456 not in sentinel


def values_are_empty_sets_(dictionary: dict[t.Any, t.Any]) -> None:
    for val in dictionary.values():
        assert val == set()


def test_int_sender() -> None:
    count = 0

    def received(sender: t.Any) -> None:
        nonlocal count
        count += 1

    sig = blinker.Signal()
    sig.connect(received, sender=123456789)
    sig.send(123456789)  # Python compiler uses same instance for same literal value.
    assert count == 1
    sig.send(int("123456789"))  # Force different instance with same value.
    assert count == 2


def test_send_robust() -> None:
    sig = blinker.Signal()
    calls = []

    def first(sender: t.Any) -> str:
        calls.append("first")
        return "one"

    def failing(sender: t.Any) -> None:
        calls.append("failing")
        raise ValueError("boom")

    def third(sender: t.Any) -> str:
        calls.append("third")
        return "three"

    sig.connect(first, weak=False)
    sig.connect(failing, weak=False)
    sig.connect(third, weak=False)

    results = dict(sig.send_robust("sender"))

    assert sorted(calls) == ["failing", "first", "third"]
    assert len(results) == 3
    assert results[first] == "one"
    assert results[third] == "three"

    err = results[failing]
    assert isinstance(err, ValueError)
    assert str(err) == "boom"
    assert err.__traceback__ is not None


def test_send_robust_base_exception_propagates() -> None:
    sig = blinker.Signal()
    calls = []

    def interrupting(sender: t.Any) -> None:
        calls.append("interrupting")
        raise KeyboardInterrupt

    def after(sender: t.Any) -> None:
        calls.append("after")

    sig.connect(interrupting, weak=False)
    sig.connect(after, weak=False)

    with pytest.raises(KeyboardInterrupt):
        sig.send_robust()

    assert calls == ["interrupting"]
    assert "after" not in calls


def test_send_robust_async_receiver_without_wrapper() -> None:
    sig = blinker.Signal()

    async def received_async(sender: t.Any) -> None:
        pass

    sig.connect(received_async, weak=False)

    with pytest.raises(RuntimeError):
        sig.send_robust()


def test_send_robust_with_async_wrapper() -> None:
    sig = blinker.Signal()

    async def received_async(sender: t.Any) -> str:
        return "async-result"

    def wrapper(func: c.Callable[..., t.Any]) -> c.Callable[..., t.Any]:
        def inner(*args: t.Any, **kwargs: t.Any) -> t.Any:
            return asyncio.run(func(*args, **kwargs))

        return inner

    sig.connect(received_async, weak=False)
    results = sig.send_robust(_async_wrapper=wrapper)
    assert results == [(received_async, "async-result")]


async def test_send_async_robust() -> None:
    sig = blinker.Signal()
    calls = []

    async def first(sender: t.Any) -> str:
        calls.append("first")
        return "one"

    async def failing(sender: t.Any) -> None:
        calls.append("failing")
        raise ValueError("boom")

    async def third(sender: t.Any) -> str:
        calls.append("third")
        return "three"

    sig.connect(first, weak=False)
    sig.connect(failing, weak=False)
    sig.connect(third, weak=False)

    results = dict(await sig.send_async_robust("sender"))

    assert sorted(calls) == ["failing", "first", "third"]
    assert len(results) == 3
    assert results[first] == "one"
    assert results[third] == "three"

    err = results[failing]
    assert isinstance(err, ValueError)
    assert str(err) == "boom"
    assert err.__traceback__ is not None


async def test_send_async_robust_cancelled() -> None:
    class OrderedSet(dict):  # type: ignore[type-arg]
        """A dict-backed ordered set, dispatching in connection order."""

        def add(self, item: t.Any) -> None:
            self[item] = None

        def discard(self, item: t.Any) -> None:
            self.pop(item, None)

    class OrderedSignal(blinker.Signal):
        set_class = OrderedSet

    sig = OrderedSignal()
    calls = []

    async def first(sender: t.Any) -> None:
        calls.append("first")

    async def cancelling(sender: t.Any) -> None:
        calls.append("cancelling")
        raise asyncio.CancelledError

    async def third(sender: t.Any) -> None:
        calls.append("third")

    sig.connect(first, weak=False)
    sig.connect(cancelling, weak=False)
    sig.connect(third, weak=False)

    with pytest.raises(asyncio.CancelledError):
        await sig.send_async_robust()

    assert calls == ["first", "cancelling"]


async def test_send_async_robust_sync_receiver_without_wrapper() -> None:
    sig = blinker.Signal()

    def received(sender: t.Any) -> None:
        pass

    sig.connect(received, weak=False)

    with pytest.raises(RuntimeError):
        await sig.send_async_robust()


async def test_send_async_robust_with_sync_wrapper() -> None:
    sig = blinker.Signal()

    def received(sender: t.Any) -> str:
        return "sync-result"

    def wrapper(func: c.Callable[..., t.Any]) -> c.Callable[..., t.Any]:
        async def inner(*args: t.Any, **kwargs: t.Any) -> t.Any:
            return func(*args, **kwargs)

        return inner

    sig.connect(received, weak=False)
    results = await sig.send_async_robust(_sync_wrapper=wrapper)
    assert results == [(received, "sync-result")]
