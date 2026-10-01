from __future__ import annotations

import asyncio
import threading
import typing as t

import pytest

from blinker import Signal


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


def test_muted_for_sender() -> None:
    sig = Signal()
    sender_a = object()
    sender_b = object()
    calls: list[object] = []

    def receiver(sender: t.Any) -> None:
        calls.append(sender)

    sig.connect(receiver, weak=False)

    with sig.muted_for(sender_a):
        sig.send(sender_a)
        sig.send(sender_b)

        with sig.muted_for(sender_b):
            sig.send(sender_a)
            sig.send(sender_b)

        sig.send(sender_a)
        sig.send(sender_b)

    sig.send(sender_a)

    assert calls == [sender_b, sender_b, sender_a]


def test_muted_for_all_senders_and_exception_cleanup() -> None:
    sig = Signal()
    sender = object()
    calls: list[object] = []

    def receiver(sender: t.Any) -> None:
        calls.append(sender)

    sig.connect(receiver, weak=False)

    with pytest.raises(RuntimeError):
        with sig.muted_for():
            sig.send(sender)
            raise RuntimeError

    sig.send(sender)
    assert calls == [sender]


def test_muted_for_exception_restores_outer_scope() -> None:
    sig = Signal()
    sender_a = object()
    sender_b = object()
    calls: list[object] = []

    def receiver(sender: t.Any) -> None:
        calls.append(sender)

    sig.connect(receiver, weak=False)

    with sig.muted_for(sender_a):
        with pytest.raises(RuntimeError):
            with sig.muted_for(sender_b):
                raise RuntimeError

        sig.send(sender_a)
        sig.send(sender_b)

    sig.send(sender_a)

    assert calls == [sender_b, sender_a]


async def test_muted_for_sender_async() -> None:
    sig = Signal()
    muted_sender = object()
    unmuted_sender = object()
    calls: list[object] = []

    async def receiver(sender: t.Any) -> None:
        calls.append(sender)

    sig.connect(receiver, weak=False)

    with sig.muted_for(muted_sender):
        await sig.send_async(muted_sender)
        await sig.send_async(unmuted_sender)

    assert calls == [unmuted_sender]


def test_muted_for_is_isolated_between_threads() -> None:
    sig = Signal()
    sender = object()
    calls: list[object] = []
    entered = threading.Event()
    proceed = threading.Event()
    exited = threading.Event()

    def receiver(sender: t.Any) -> None:
        calls.append(sender)

    def muted_sender() -> None:
        with sig.muted_for(sender):
            entered.set()
            proceed.wait(2)
            sig.send(sender)
        exited.set()

    sig.connect(receiver, weak=False)
    thread = threading.Thread(target=muted_sender)
    thread.start()

    assert entered.wait(2)
    sig.send(sender)
    proceed.set()
    assert exited.wait(2)
    thread.join(2)
    assert not thread.is_alive()
    sig.send(sender)

    assert calls == [sender, sender]


async def test_muted_for_async_tasks_interleave() -> None:
    sig = Signal()
    sender_a = object()
    sender_b = object()
    calls: list[object] = []
    unmuted_started = asyncio.Event()
    allow_unmuted_send = asyncio.Event()

    async def receiver(sender: t.Any) -> None:
        calls.append(sender)
        await asyncio.sleep(0)

    async def unmuted_sender() -> None:
        unmuted_started.set()
        await allow_unmuted_send.wait()
        await sig.send_async(sender_b)

    sig.connect(receiver, weak=False)
    unmuted_task = asyncio.create_task(unmuted_sender())
    await unmuted_started.wait()

    with sig.muted_for(sender_a):
        with sig.muted_for(sender_b):
            await sig.send_async(sender_a)
            await sig.send_async(sender_b)
            allow_unmuted_send.set()
            await asyncio.sleep(0)

        await sig.send_async(sender_b)
        await sig.send_async(sender_a)

    await unmuted_task
    await sig.send_async(sender_a)

    assert calls == [sender_b, sender_b, sender_a]
