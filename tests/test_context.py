from __future__ import annotations

import gc
import typing as t
import weakref

import pytest

from blinker import Signal
from blinker.base import ANY_ID


def test_temp_connection() -> None:
    sig = Signal()

    canary = []

    def receiver(sender: t.Any) -> None:
        canary.append(sender)

    sig.send(1)
    with sig.connected_to(receiver):
        sig.send(2)
    sig.send(3)

    assert canary == [2]
    assert not sig.receivers


def test_temp_connection_for_sender() -> None:
    sig = Signal()

    canary = []

    def receiver(sender: t.Any) -> None:
        canary.append(sender)

    with sig.connected_to(receiver, sender=2):
        sig.send(1)
        sig.send(2)

    assert canary == [2]
    assert not sig.receivers


class Failure(Exception):
    pass


class BaseFailure(BaseException):
    pass


@pytest.mark.parametrize("exc_type", [Failure, BaseFailure])
def test_temp_connection_failure(exc_type: type[BaseException]) -> None:
    sig = Signal()

    canary = []

    def receiver(sender: t.Any) -> None:
        canary.append(sender)

    with pytest.raises(exc_type):
        sig.send(1)
        with sig.connected_to(receiver):
            sig.send(2)
            raise exc_type
        sig.send(3)

    assert canary == [2]
    assert not sig.receivers


def test_temp_connection_preserves_existing_sender_and_any() -> None:
    sig = Signal()
    events: list[tuple[str, t.Any]] = []
    sender_a = object()
    sender_b = object()

    def receiver(sender: t.Any) -> None:
        events.append(("receive", sender))

    def connected(
        signal: Signal, *, receiver: t.Any, sender: t.Any, weak: bool
    ) -> None:
        events.append(("connect", sender, weak))

    def disconnected(signal: Signal, *, receiver: t.Any, sender: t.Any) -> None:
        events.append(("disconnect", sender))

    sig.receiver_connected.connect(connected, weak=False)
    sig.receiver_disconnected.connect(disconnected, weak=False)
    sig.connect(receiver, sender=sender_a, weak=True)

    with sig.connected_to(receiver, sender=sender_b):
        sig.send(sender_a)
        sig.send(sender_b)

    sig.send(sender_a)
    sig.send(sender_b)

    assert [event for event in events if event[0] == "receive"] == [
        ("receive", sender_a),
        ("receive", sender_b),
        ("receive", sender_a),
    ]
    assert [event for event in events if event[0] in {"connect", "disconnect"}] == [
        ("connect", sender_a, True),
        ("connect", sender_b, True),
        ("disconnect", sender_b),
    ]
    assert list(sig.receivers)
    assert isinstance(next(iter(sig.receivers.values())), weakref.ref)


def test_temp_any_connection_preserves_explicit_sender() -> None:
    sig = Signal()
    sender_a = object()
    sender_b = object()
    canary: list[t.Any] = []

    def receiver(sender: t.Any) -> None:
        canary.append(sender)

    sig.connect(receiver, sender=sender_a, weak=True)

    with sig.connected_to(receiver):
        sig.send(sender_a)
        sig.send(sender_b)

    sig.send(sender_a)
    sig.send(sender_b)

    assert canary == [sender_a, sender_b, sender_a]
    assert len(next(iter(sig._by_receiver.values()))) == 1
    assert isinstance(next(iter(sig.receivers.values())), weakref.ref)


def test_temp_connection_preserves_existing_any() -> None:
    sig = Signal()
    sender_a = object()
    sender_b = object()
    canary: list[t.Any] = []

    def receiver(sender: t.Any) -> None:
        canary.append(sender)

    sig.connect(receiver)

    with sig.connected_to(receiver, sender=sender_b):
        sig.send(sender_a)
        sig.send(sender_b)

    sig.send(sender_a)
    sig.send(sender_b)

    assert canary == [sender_a, sender_b, sender_a, sender_b]
    assert next(iter(sig._by_receiver.values())) == {ANY_ID}


def test_nested_temp_connection_inner_failure() -> None:
    sig = Signal()
    sender_a = object()
    sender_b = object()
    events: list[tuple[str, t.Any]] = []

    def receiver(sender: t.Any) -> None:
        events.append(("receive", sender))

    def connected(
        signal: Signal, *, receiver: t.Any, sender: t.Any, weak: bool
    ) -> None:
        events.append(("connect", sender, weak))

    def disconnected(signal: Signal, *, receiver: t.Any, sender: t.Any) -> None:
        events.append(("disconnect", sender))

    sig.receiver_connected.connect(connected, weak=False)
    sig.receiver_disconnected.connect(disconnected, weak=False)

    with sig.connected_to(receiver, sender=sender_a):
        with pytest.raises(Failure):
            with sig.connected_to(receiver, sender=sender_b):
                sig.send(sender_a)
                sig.send(sender_b)
                raise Failure

        sig.send(sender_a)
        sig.send(sender_b)

    sig.send(sender_a)

    assert [event for event in events if event[0] == "receive"] == [
        ("receive", sender_a),
        ("receive", sender_b),
        ("receive", sender_a),
    ]
    assert [event for event in events if event[0] in {"connect", "disconnect"}] == [
        ("connect", sender_a, False),
        ("connect", sender_b, False),
        ("disconnect", sender_b),
        ("disconnect", sender_a),
    ]
    assert not sig.receivers


def test_nested_temp_connection_same_sender() -> None:
    sig = Signal()
    sender_a = object()
    events: list[tuple[str, t.Any]] = []

    def receiver(sender: t.Any) -> None:
        events.append(("receive", sender))

    def connected(
        signal: Signal, *, receiver: t.Any, sender: t.Any, weak: bool
    ) -> None:
        events.append(("connect", sender, weak))

    def disconnected(signal: Signal, *, receiver: t.Any, sender: t.Any) -> None:
        events.append(("disconnect", sender))

    sig.receiver_connected.connect(connected, weak=False)
    sig.receiver_disconnected.connect(disconnected, weak=False)

    with sig.connected_to(receiver, sender=sender_a):
        with pytest.raises(Failure):
            with sig.connected_to(receiver, sender=sender_a):
                raise Failure

    assert [event for event in events if event[0] == "receive"] == []
    assert events == [
        ("connect", sender_a, False),
        ("disconnect", sender_a),
    ]
    assert not sig.receivers


def test_temp_connection_matches_bound_method_and_preserves_weak_ref() -> None:
    class Receiver:
        def __init__(self) -> None:
            self.canary: list[t.Any] = []

        def received(self, sender: t.Any) -> None:
            self.canary.append(sender)

    sig = Signal()
    sender_a = object()
    sender_b = object()
    receiver = Receiver()

    sig.connect(receiver.received, sender=sender_a)

    with sig.connected_to(receiver.received, sender=sender_b):
        sig.send(sender_a)
        sig.send(sender_b)

    sig.send(sender_a)
    sig.send(sender_b)

    assert receiver.canary == [sender_a, sender_b, sender_a]
    assert isinstance(next(iter(sig.receivers.values())), weakref.ref)

    del receiver
    gc.collect()

    assert not sig.receivers


def test_temp_connection_receiver_can_be_garbage_collected() -> None:
    sig = Signal()
    sender = object()
    canary: list[t.Any] = []

    def receiver(value: t.Any) -> None:
        canary.append(value)

    with sig.connected_to(receiver, sender=sender):
        sig.send(sender)

    assert canary == [sender]
    del receiver
    gc.collect()

    assert not sig.receivers


def test_temp_connection_after_explicit_sender_disconnect_can_be_collected() -> None:
    sig = Signal()
    sender_a = object()
    sender_b = object()
    canary: list[t.Any] = []

    def receiver(sender: t.Any) -> None:
        canary.append(sender)

    sig.connect(receiver, sender=sender_a, weak=False)
    sig.disconnect(receiver, sender=sender_a)

    with sig.connected_to(receiver, sender=sender_b):
        sig.send(sender_b)

    assert canary == [sender_b]
    del receiver
    gc.collect()

    assert not sig.receivers
