from __future__ import annotations

import gc
import typing as t
import weakref

import pytest

from blinker import ANY
from blinker import Signal
from blinker._utilities import make_id


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


def test_temp_connection_keeps_existing_sender_connection() -> None:
    sig = Signal()
    sender_a = object()
    sender_b = object()
    events = []

    def receiver(sender: t.Any) -> None:
        events.append(("receive", sender))

    def connected(signal: t.Any, **kwargs: t.Any) -> None:
        events.append(("connect", kwargs["sender"]))

    def disconnected(signal: t.Any, **kwargs: t.Any) -> None:
        events.append(("disconnect", kwargs["sender"]))

    sig.receiver_connected.connect(connected, weak=False)
    sig.receiver_disconnected.connect(disconnected, weak=False)
    sig.connect(receiver, sender=sender_a, weak=False)

    with sig.connected_to(receiver, sender=sender_b):
        sig.send(sender_a)
        sig.send(sender_b)

    sig.send(sender_a)
    sig.send(sender_b)

    assert events == [
        ("connect", sender_a),
        ("connect", sender_b),
        ("receive", sender_a),
        ("receive", sender_b),
        ("disconnect", sender_b),
        ("receive", sender_a),
    ]
    assert id(sender_a) in sig._by_sender
    assert id(sender_b) not in sig._weak_senders


def test_temp_any_connection_keeps_existing_sender_connection() -> None:
    sig = Signal()
    sender = object()
    events = []

    def receiver(received_sender: t.Any) -> None:
        events.append(("receive", received_sender))

    def disconnected(signal: t.Any, **kwargs: t.Any) -> None:
        events.append(("disconnect", kwargs["sender"]))

    sig.receiver_disconnected.connect(disconnected, weak=False)
    sig.connect(receiver, sender=sender, weak=False)
    other_sender = object()

    with sig.connected_to(receiver):
        sig.send(sender)
        sig.send(other_sender)

    sig.send(sender)

    assert events == [
        ("receive", sender),
        ("receive", other_sender),
        ("disconnect", ANY),
        ("receive", sender),
    ]
    assert id(sender) in sig._by_sender


def test_temp_sender_connection_keeps_existing_any_weak_connection() -> None:
    sig = Signal()
    sender = object()
    events = []

    def receiver(received_sender: t.Any) -> None:
        events.append(("receive", received_sender))

    def connected(signal: t.Any, **kwargs: t.Any) -> None:
        events.append(("connect", kwargs["sender"], kwargs["weak"]))

    def disconnected(signal: t.Any, **kwargs: t.Any) -> None:
        events.append(("disconnect", kwargs["sender"]))

    other_sender = object()

    sig.receiver_connected.connect(connected, weak=False)
    sig.receiver_disconnected.connect(disconnected, weak=False)
    sig.connect(receiver)

    with sig.connected_to(receiver, sender=sender):
        sig.send(other_sender)
        sig.send(sender)

    sig.send(sender)

    assert events == [
        ("connect", ANY, True),
        ("connect", sender, False),
        ("receive", other_sender),
        ("receive", sender),
        ("disconnect", sender),
        ("receive", sender),
    ]
    assert 0 in sig._by_sender
    assert id(sender) not in sig._weak_senders
    assert isinstance(next(iter(sig.receivers.values())), weakref.ref)


def test_nested_temp_connection_failure_keeps_outer_connection() -> None:
    sig = Signal()
    sender_a = object()
    sender_b = object()
    canary = []

    def receiver(sender: t.Any) -> None:
        canary.append(sender)

    with sig.connected_to(receiver, sender=sender_a):
        try:
            with sig.connected_to(receiver, sender=sender_b):
                sig.send(sender_a)
                sig.send(sender_b)
                raise Failure
        except Failure:
            pass

        sig.send(sender_a)
        assert canary == [sender_a, sender_b, sender_a]
        assert id(sender_a) in sig._by_sender
        assert id(sender_b) not in sig._weak_senders

    sig.send(sender_a)
    sig.send(sender_b)

    assert canary == [sender_a, sender_b, sender_a]


def test_nested_temp_connection_propagated_failure_only_removes_inner() -> None:
    sig = Signal()
    sender_a = object()
    sender_b = object()
    events = []

    def receiver(sender: t.Any) -> None:
        events.append(("receive", sender))

    def disconnected(signal: t.Any, **kwargs: t.Any) -> None:
        events.append(("disconnect", kwargs["sender"]))

    sig.receiver_disconnected.connect(disconnected, weak=False)

    with pytest.raises(Failure):
        with sig.connected_to(receiver, sender=sender_a):
            with sig.connected_to(receiver, sender=sender_b):
                raise Failure

    assert events == [("disconnect", sender_b), ("disconnect", sender_a)]
    assert not sig.receivers


def test_nested_same_sender_failure_keeps_outer_connection() -> None:
    sig = Signal()
    sender = object()
    events = []

    def receiver(received_sender: t.Any) -> None:
        events.append(("receive", received_sender))

    def disconnected(signal: t.Any, **kwargs: t.Any) -> None:
        events.append(("disconnect", kwargs["sender"]))

    sig.receiver_disconnected.connect(disconnected, weak=False)

    with sig.connected_to(receiver, sender=sender):
        try:
            with sig.connected_to(receiver, sender=sender):
                sig.send(sender)
                raise Failure
        except Failure:
            pass

        sig.send(sender)
        assert events == [("receive", sender), ("receive", sender)]

    assert events == [
        ("receive", sender),
        ("receive", sender),
        ("disconnect", sender),
    ]
    assert not sig.receivers


def test_temp_bound_method_restores_weak_identity_and_can_be_collected() -> None:
    class Receiver:
        def __init__(self) -> None:
            self.events = []

        def received(self, sender: t.Any) -> None:
            self.events.append(sender)

    sig = Signal()
    sender = object()
    receiver = Receiver()
    receiver_ref = weakref.ref(receiver)
    receiver_id = make_id(receiver.received)
    assert make_id(receiver.received) == receiver_id

    sig.connect(receiver.received, sender=sender)

    with sig.connected_to(receiver.received):
        sig.send(sender)

    stored_receiver = sig.receivers[receiver_id]
    assert isinstance(stored_receiver, weakref.ref)
    assert stored_receiver() == receiver.received
    assert id(sender) in sig._by_sender
    assert sig._by_receiver[receiver_id] == {id(sender)}

    sig.send(sender)
    assert receiver.events == [sender, sender]

    del receiver
    gc.collect()

    assert receiver_ref() is None
    assert not sig.receivers
    sig.send(sender)


def test_temporary_bound_method_can_be_collected_after_exit() -> None:
    class Receiver:
        def received(self, sender: t.Any) -> None:
            pass

    sig = Signal()
    receiver = Receiver()
    receiver_ref = weakref.ref(receiver)

    with sig.connected_to(receiver.received):
        assert sig.receivers
        sig.send(None)

    del receiver
    gc.collect()

    assert receiver_ref() is None
    assert not sig.receivers
