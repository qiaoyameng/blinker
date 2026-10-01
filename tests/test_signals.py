from __future__ import annotations

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


def test_receiver_priority_and_registration_order() -> None:
    calls: list[str] = []
    sig = blinker.Signal()

    def first_default(sender: t.Any) -> None:
        calls.append("first-default")

    def second_default(sender: t.Any) -> None:
        calls.append("second-default")

    def low_priority(sender: t.Any) -> None:
        calls.append("low-priority")

    sig.connect(low_priority, priority=1)
    sig.connect(first_default)
    sig.connect(second_default)

    sig.send()

    assert calls == [
        "first-default",
        "second-default",
        "low-priority",
    ]


def test_connect_via_priority() -> None:
    calls: list[str] = []
    sig = blinker.Signal()
    sender = object()

    @sig.connect_via(sender, priority=1)
    def low_priority(sender: t.Any) -> None:
        calls.append("low-priority")

    @sig.connect_via(sender)
    def default_priority(sender: t.Any) -> None:
        calls.append("default-priority")

    sig.send(sender)
    assert calls == ["default-priority", "low-priority"]


def test_connected_to_priority() -> None:
    calls: list[str] = []
    sig = blinker.Signal()

    def default_priority(sender: t.Any) -> None:
        calls.append("default-priority")

    def high_number_priority(sender: t.Any) -> None:
        calls.append("high-number-priority")

    sig.connect(default_priority)

    with sig.connected_to(high_number_priority, priority=1):
        sig.send()

    assert calls == ["default-priority", "high-number-priority"]


def test_specific_sender_priority_overrides_any_priority() -> None:
    calls: list[str] = []
    sig = blinker.Signal()
    sender = object()

    def any_only(sender: t.Any) -> None:
        calls.append("any-only")

    def shared(sender: t.Any) -> None:
        calls.append("shared")

    sig.connect(any_only, priority=-5)
    sig.connect(shared, priority=100)
    sig.connect(shared, sender=sender, priority=-10)

    sig.send(sender)
    assert calls == ["shared", "any-only"]

    calls.clear()
    sig.send(blinker.ANY)
    assert calls == ["any-only", "shared"]

    sig.connect(shared, sender=sender, priority=20)
    sig.connect(shared, priority=0)

    calls.clear()
    sig.send(sender)
    assert calls == ["any-only", "shared"]


def test_reconnect_updates_priority_without_changing_equal_priority_order() -> None:
    calls: list[str] = []
    sig = blinker.Signal()

    def first(sender: t.Any) -> None:
        calls.append("first")

    def second(sender: t.Any) -> None:
        calls.append("second")

    sig.connect(first)
    sig.connect(second)

    sig.send()
    assert calls == ["first", "second"]

    sig.connect(first, priority=1)
    calls.clear()
    sig.send()
    assert calls == ["second", "first"]

    sig.connect(first, priority=0)
    calls.clear()
    sig.send()
    assert calls == ["first", "second"]

    sig.disconnect(first)
    sig.connect(first)
    calls.clear()
    sig.send()
    assert calls == ["second", "first"]


def test_send_uses_snapshot_when_receivers_change() -> None:
    calls: list[str] = []
    sig = blinker.Signal()

    def added(sender: t.Any) -> None:
        calls.append("added")

    def first(sender: t.Any) -> None:
        calls.append("first")
        sig.connect(added, weak=False)

    def second(sender: t.Any) -> None:
        calls.append("second")
        sig.disconnect(removed)

    def removed(sender: t.Any) -> None:
        calls.append("removed")

    sig.connect(first)
    sig.connect(second)
    sig.connect(removed)

    sig.send()
    assert calls == ["first", "second", "removed"]

    calls.clear()
    sig.send()
    assert calls == ["first", "second", "added"]


def test_priority_change_during_send_starts_next_round() -> None:
    calls: list[str] = []
    sig = blinker.Signal()

    def prioritized(sender: t.Any) -> None:
        calls.append("prioritized")

    def reprioritize(sender: t.Any) -> None:
        calls.append("reprioritize")
        sig.connect(prioritized, priority=-1)

    sig.connect(prioritized, priority=1)
    sig.connect(reprioritize, priority=0)

    sig.send()
    assert calls == ["reprioritize", "prioritized"]

    calls.clear()
    sig.send()
    assert calls == ["prioritized", "reprioritize"]


def test_dead_weak_receiver_is_cleaned_up_during_snapshot() -> None:
    calls: list[str] = []
    sig = blinker.Signal()
    holder: list[c.Callable[..., None]] = []

    def release_weak_receiver(sender: t.Any) -> None:
        calls.append("first")
        holder.clear()
        collect_acyclic_refs()

    def make_weak_receiver() -> c.Callable[..., None]:
        def weak_receiver(sender: t.Any) -> None:
            calls.append("weak")

        return weak_receiver

    holder.append(make_weak_receiver())
    sig.connect(release_weak_receiver)
    sig.connect(holder[0])

    sig.send()
    assert calls == ["first"]

    calls.clear()
    sig.send()
    assert calls == ["first"]


async def test_send_and_send_async_use_same_priority_order() -> None:
    calls: list[str] = []
    sig = blinker.Signal()

    def first(sender: t.Any) -> None:
        calls.append("first")

    def second(sender: t.Any) -> None:
        calls.append("second")

    def third(sender: t.Any) -> None:
        calls.append("third")

    sig.connect(first, priority=-1)
    sig.connect(second)
    sig.connect(third, priority=1)

    sig.send()
    assert calls == ["first", "second", "third"]

    calls.clear()

    def wrapper(func: c.Callable[..., t.Any]) -> c.Callable[..., t.Any]:
        async def inner(sender: t.Any) -> None:
            func(sender)

        return inner

    await sig.send_async(_sync_wrapper=wrapper)
    assert calls == ["first", "second", "third"]
