from __future__ import annotations

import asyncio
import collections.abc as c
import gc
import sys
import traceback
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


class OrderedSet(set):  # type: ignore[type-arg]
    def __init__(self, values: c.Iterable[t.Any] = ()) -> None:
        self._values: list[t.Any] = []
        super().__init__()

        for value in values:
            self.add(value)

    def add(self, value: t.Any) -> None:
        if value not in self:
            self._values.append(value)

        super().add(value)

    def discard(self, value: t.Any) -> None:
        super().discard(value)

        if value in self._values:
            self._values.remove(value)

    def copy(self) -> OrderedSet:
        return OrderedSet(self)

    def __iter__(self) -> c.Iterator[t.Any]:
        return iter(self._values)

    def __or__(self, other: c.Iterable[t.Any]) -> OrderedSet:
        return OrderedSet((*self, *other))


class OrderedSignal(blinker.Signal):
    set_class = OrderedSet


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


def test_send_still_short_circuits_on_error() -> None:
    calls = []

    def first(sender: t.Any) -> None:
        calls.append(1)

    def second(sender: t.Any) -> t.Any:
        calls.append(2)
        raise RuntimeError("failed")

    def third(sender: t.Any) -> None:
        calls.append(3)

    sig = OrderedSignal()
    sig.connect(first, weak=False)
    sig.connect(second, weak=False)
    sig.connect(third, weak=False)

    with pytest.raises(RuntimeError, match="failed"):
        sig.send()

    assert calls == [1, 2]


async def test_send_async_still_short_circuits_on_error() -> None:
    calls = []

    async def first(sender: t.Any) -> None:
        calls.append(1)

    async def second(sender: t.Any) -> t.Any:
        calls.append(2)
        raise RuntimeError("failed")

    async def third(sender: t.Any) -> None:
        calls.append(3)

    sig = OrderedSignal()
    sig.connect(first, weak=False)
    sig.connect(second, weak=False)
    sig.connect(third, weak=False)

    with pytest.raises(RuntimeError, match="failed"):
        await sig.send_async()

    assert calls == [1, 2]


def test_send_robust_collects_results_and_tracebacks() -> None:
    calls = []

    def first(sender: t.Any, **kwargs: t.Any) -> str:
        calls.append(1)
        return "first"

    def second(sender: t.Any, **kwargs: t.Any) -> t.Any:
        calls.append(2)
        raise RuntimeError("second failed")

    def third(sender: t.Any, **kwargs: t.Any) -> str:
        calls.append(3)
        return "third"

    sig = OrderedSignal()
    sig.connect(first, weak=False)
    sig.connect(second, weak=False)
    sig.connect(third, weak=False)

    results = sig.send_robust("sender", value="value")

    assert calls == [1, 2, 3]
    assert [receiver for receiver, _ in results] == [first, second, third]
    assert results[0] == (first, "first")
    assert results[2] == (third, "third")

    failed_receiver, failure = results[1]
    assert failed_receiver is second
    assert isinstance(failure, RuntimeError)
    assert str(failure) == "second failed"
    assert traceback.extract_tb(failure.__traceback__)[-1].name == "second"


def test_send_robust_handles_mixed_receivers_and_wrappers() -> None:
    calls = []

    def sync_success(sender: t.Any) -> str:
        calls.append(("sync", "success"))
        return "sync"

    async def async_failure(sender: t.Any) -> t.Any:
        calls.append(("async", "failure"))
        raise RuntimeError("async failed")

    def sync_after(sender: t.Any) -> str:
        calls.append(("sync", "after"))
        return "after"

    def async_wrapper(func: c.Callable[..., t.Any]) -> c.Callable[..., t.Any]:
        def wrapped(*args: t.Any, **kwargs: t.Any) -> t.Any:
            return asyncio.run(func(*args, **kwargs))

        return wrapped

    sig = OrderedSignal()
    sig.connect(sync_success, weak=False)
    sig.connect(async_failure, weak=False)
    sig.connect(sync_after, weak=False)

    results = sig.send_robust(_async_wrapper=async_wrapper)

    assert calls == [("sync", "success"), ("async", "failure"), ("sync", "after")]
    assert [receiver for receiver, _ in results] == [
        sync_success,
        async_failure,
        sync_after,
    ]
    assert isinstance(results[1][1], RuntimeError)
    assert traceback.extract_tb(results[1][1].__traceback__)[-1].name == "async_failure"


async def test_send_robust_async_collects_mixed_receiver_results() -> None:
    calls = []

    def sync_success(sender: t.Any) -> str:
        calls.append(("sync", "success"))
        return "sync"

    async def async_failure(sender: t.Any) -> t.Any:
        calls.append(("async", "failure"))
        raise RuntimeError("async failed")

    async def async_success(sender: t.Any) -> str:
        calls.append(("async", "success"))
        return "async"

    def sync_wrapper(func: c.Callable[..., t.Any]) -> c.Callable[..., t.Any]:
        async def wrapped(*args: t.Any, **kwargs: t.Any) -> t.Any:
            return func(*args, **kwargs)

        return wrapped

    sig = OrderedSignal()
    sig.connect(sync_success, weak=False)
    sig.connect(async_failure, weak=False)
    sig.connect(async_success, weak=False)

    results = await sig.send_robust_async(_sync_wrapper=sync_wrapper)

    assert calls == [("sync", "success"), ("async", "failure"), ("async", "success")]
    assert [receiver for receiver, _ in results] == [
        sync_success,
        async_failure,
        async_success,
    ]
    assert results[0][1] == "sync"
    assert isinstance(results[1][1], RuntimeError)
    assert results[2][1] == "async"


def test_send_robust_reports_incompatible_receiver_and_continues() -> None:
    calls = []

    async def async_receiver(sender: t.Any) -> None:
        calls.append("async")

    def sync_receiver(sender: t.Any) -> None:
        calls.append("sync")

    sig = OrderedSignal()
    sig.connect(async_receiver, weak=False)
    sig.connect(sync_receiver, weak=False)

    results = sig.send_robust()

    assert calls == ["sync"]
    assert [receiver for receiver, _ in results] == [async_receiver, sync_receiver]
    assert type(results[0][1]) is RuntimeError


async def test_send_robust_async_reports_incompatible_receiver_and_continues() -> None:
    calls = []

    def sync_receiver(sender: t.Any) -> None:
        calls.append("sync")

    async def async_receiver(sender: t.Any) -> None:
        calls.append("async")

    sig = OrderedSignal()
    sig.connect(sync_receiver, weak=False)
    sig.connect(async_receiver, weak=False)

    results = await sig.send_robust_async()

    assert calls == ["async"]
    assert [receiver for receiver, _ in results] == [sync_receiver, async_receiver]
    assert type(results[0][1]) is RuntimeError


def test_send_robust_collects_wrapper_errors() -> None:
    calls = []

    async def async_receiver(sender: t.Any) -> None:
        calls.append("async")

    def sync_receiver(sender: t.Any) -> None:
        calls.append("sync")

    def bad_wrapper(func: c.Callable[..., t.Any]) -> c.Callable[..., t.Any]:
        raise RuntimeError("wrapper failed")

    sig = OrderedSignal()
    sig.connect(async_receiver, weak=False)
    sig.connect(sync_receiver, weak=False)

    results = sig.send_robust(_async_wrapper=bad_wrapper)

    assert calls == ["sync"]
    assert isinstance(results[0][1], RuntimeError)
    assert str(results[0][1]) == "wrapper failed"


async def test_send_robust_async_collects_wrapper_errors() -> None:
    calls = []

    def sync_receiver(sender: t.Any) -> None:
        calls.append("sync")

    async def async_receiver(sender: t.Any) -> None:
        calls.append("async")

    def bad_wrapper(func: c.Callable[..., t.Any]) -> c.Callable[..., t.Any]:
        raise RuntimeError("wrapper failed")

    sig = OrderedSignal()
    sig.connect(sync_receiver, weak=False)
    sig.connect(async_receiver, weak=False)

    results = await sig.send_robust_async(_sync_wrapper=bad_wrapper)

    assert calls == ["async"]
    assert isinstance(results[0][1], RuntimeError)
    assert str(results[0][1]) == "wrapper failed"


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit])
def test_send_robust_propagates_base_exceptions(exception: type[BaseException]) -> None:
    calls = []

    def first(sender: t.Any) -> None:
        calls.append(1)

    def second(sender: t.Any) -> t.Any:
        calls.append(2)
        raise exception()

    def third(sender: t.Any) -> None:
        calls.append(3)

    sig = OrderedSignal()
    sig.connect(first, weak=False)
    sig.connect(second, weak=False)
    sig.connect(third, weak=False)

    with pytest.raises(exception):
        sig.send_robust()

    assert calls == [1, 2]


async def test_send_robust_async_propagates_cancellation() -> None:
    calls = []
    second_started = asyncio.Event()

    async def first(sender: t.Any) -> None:
        calls.append(1)

    async def second(sender: t.Any) -> None:
        calls.append(2)
        second_started.set()
        await asyncio.sleep(3600)

    async def third(sender: t.Any) -> None:
        calls.append(3)

    sig = OrderedSignal()
    sig.connect(first, weak=False)
    sig.connect(second, weak=False)
    sig.connect(third, weak=False)

    task = asyncio.ensure_future(sig.send_robust_async())
    await asyncio.wait_for(second_started.wait(), timeout=1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert calls == [1, 2]


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
