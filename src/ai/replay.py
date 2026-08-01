from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ai.encoding import ACTION_SIZE, INPUT_CHANNELS
from ai.self_play import GameResult

SCHEMA_VERSION = 1
ENCODING_VERSION = 1
ACTION_VERSION = 1


class ReplayCompatibilityError(RuntimeError):
    """Replay 数据格式与当前程序不兼容。"""


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    states: NDArray[np.float32]
    policy_indices: tuple[NDArray[np.int64], ...]
    policy_probabilities: tuple[NDArray[np.float32], ...]
    values: NDArray[np.float32]


class ReplayBuffer:
    """按完整棋局原子提交的磁盘 Replay Buffer。"""

    def __init__(self, path: Path | str, *, capacity_games: int) -> None:
        if type(capacity_games) is not int or capacity_games <= 0:
            raise ValueError("capacity_games 必须是大于 0 的整数")
        self.path = Path(path)
        self.games_path = self.path / "games"
        self.manifest_path = self.path / "manifest.json"
        self.capacity_games = capacity_games
        self.games_path.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            self._manifest = self._read_manifest()
        else:
            self._manifest = {
                "schema_version": SCHEMA_VERSION,
                "encoding_version": ENCODING_VERSION,
                "action_version": ACTION_VERSION,
                "next_game_id": 1,
                "games": [],
            }
            self._write_manifest(self._manifest)

    @property
    def game_ids(self) -> tuple[int, ...]:
        return tuple(self._manifest["games"])

    @property
    def manifest_hash(self) -> str:
        return hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()

    @property
    def manifest_version(self) -> int:
        return SCHEMA_VERSION

    def append_game(self, game: GameResult) -> int:
        game_id = int(self._manifest["next_game_id"])
        destination = self._game_path(game_id)
        arrays = self._encode_game(game)
        temporary = self._temporary_path(self.games_path, destination.name)
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            self._fsync_directory(self.games_path)
        finally:
            temporary.unlink(missing_ok=True)

        old_games = list(self._manifest["games"])
        new_games = [*old_games, game_id]
        evicted = new_games[: -self.capacity_games]
        new_games = new_games[-self.capacity_games :]
        updated = {**self._manifest, "next_game_id": game_id + 1, "games": new_games}
        self._write_manifest(updated)
        self._manifest = updated
        for old_id in evicted:
            self._game_path(old_id).unlink(missing_ok=True)
        self._fsync_directory(self.games_path)
        return game_id

    def sample(self, batch_size: int, rng: np.random.Generator) -> ReplayBatch:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size 必须是大于 0 的整数")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng 必须是 numpy.random.Generator")

        locations: list[tuple[int, int]] = []
        for game_id in self.game_ids:
            data = self._read_game(game_id)
            count = int(data["values"].shape[0])
            locations.extend((game_id, index) for index in range(count))
        if batch_size > len(locations):
            raise ValueError(
                f"样本不足：请求 {batch_size}，Replay 中只有 {len(locations)}"
            )

        selected = rng.choice(len(locations), size=batch_size, replace=False)
        cache: dict[int, dict[str, NDArray[np.generic]]] = {}
        states: list[NDArray[np.float32]] = []
        indices: list[NDArray[np.int64]] = []
        probabilities: list[NDArray[np.float32]] = []
        values: list[float] = []
        for selection in np.asarray(selected).reshape(-1):
            game_id, sample_index = locations[int(selection)]
            if game_id not in cache:
                cache[game_id] = self._read_game(game_id)
            data = cache[game_id]
            start = int(data["policy_offsets"][sample_index])
            end = int(data["policy_offsets"][sample_index + 1])
            states.append(np.asarray(data["states"][sample_index], dtype=np.float32))
            indices.append(
                np.asarray(data["policy_indices"][start:end], dtype=np.int64)
            )
            probabilities.append(
                np.asarray(data["policy_probabilities"][start:end], dtype=np.float32)
            )
            values.append(float(data["values"][sample_index]))
        return ReplayBatch(
            states=np.stack(states).astype(np.float32, copy=False),
            policy_indices=tuple(indices),
            policy_probabilities=tuple(probabilities),
            values=np.asarray(values, dtype=np.float32),
        )

    def _read_game(self, game_id: int) -> dict[str, NDArray[np.generic]]:
        try:
            with np.load(self._game_path(game_id), allow_pickle=False) as stored:
                data = {key: stored[key].copy() for key in stored.files}
        except (OSError, ValueError, KeyError) as error:
            raise ReplayCompatibilityError(f"Replay 棋局 {game_id} 无法读取") from error
        expected = {
            "schema_version": SCHEMA_VERSION,
            "encoding_version": ENCODING_VERSION,
            "action_version": ACTION_VERSION,
        }
        for name, version in expected.items():
            try:
                actual = int(np.asarray(data[name]).reshape(-1)[0])
            except (KeyError, IndexError, TypeError, ValueError) as error:
                raise ReplayCompatibilityError(
                    f"Replay 棋局 {game_id} 缺少有效 {name}"
                ) from error
            if actual != version:
                raise ReplayCompatibilityError(
                    f"Replay 棋局 {game_id} 的 {name} 不兼容：{actual} != {version}"
                )
        required = {
            "states",
            "policy_indices",
            "policy_probabilities",
            "policy_offsets",
            "values",
            "sides",
            "plies",
        }
        missing = sorted(required.difference(data))
        if missing:
            raise ReplayCompatibilityError(
                f"Replay 棋局 {game_id} 缺少数组 {', '.join(missing)}"
            )
        states = data["states"]
        indices = data["policy_indices"]
        probabilities = data["policy_probabilities"]
        offsets = data["policy_offsets"]
        values = data["values"]
        sides = data["sides"]
        sample_count = int(values.shape[0]) if values.ndim == 1 else -1
        valid_offsets = (
            offsets.ndim == 1
            and offsets.shape[0] == sample_count + 1
            and offsets.shape[0] > 0
            and int(offsets[0]) == 0
            and np.all(np.diff(offsets) > 0)
            and int(offsets[-1]) == indices.size
        )
        if not valid_offsets:
            raise ReplayCompatibilityError(
                f"Replay 棋局 {game_id} 的 policy_offsets 无效"
            )
        if states.shape != (sample_count, INPUT_CHANNELS, 10, 9) or sides.shape != (
            sample_count,
        ):
            raise ReplayCompatibilityError(f"Replay 棋局 {game_id} 的样本形状无效")
        if (
            indices.ndim != 1
            or probabilities.shape != indices.shape
            or np.any(indices < 0)
            or np.any(indices >= ACTION_SIZE)
            or not np.all(np.isfinite(probabilities))
            or np.any(probabilities < 0)
            or not np.all(np.isfinite(values))
            or np.any(values < -1)
            or np.any(values > 1)
        ):
            raise ReplayCompatibilityError(f"Replay 棋局 {game_id} 的稀疏策略无效")
        for start, end in pairwise(offsets):
            if not np.isclose(float(probabilities[int(start) : int(end)].sum()), 1.0):
                raise ReplayCompatibilityError(
                    f"Replay 棋局 {game_id} 的策略概率之和无效"
                )
        return data

    def _encode_game(self, game: GameResult) -> dict[str, NDArray[np.generic]]:
        states: list[NDArray[np.float32]] = []
        all_indices: list[NDArray[np.int64]] = []
        all_probabilities: list[NDArray[np.float32]] = []
        offsets = [0]
        values: list[float] = []
        sides: list[int] = []
        for sample in game.samples:
            state = np.asarray(sample.state, dtype=np.float32)
            indices = np.asarray(sample.policy_indices, dtype=np.int64)
            probabilities = np.asarray(sample.policy_probabilities, dtype=np.float32)
            if state.shape != (INPUT_CHANNELS, 10, 9):
                raise ValueError(f"局面形状必须是 {(INPUT_CHANNELS, 10, 9)}")
            if (
                indices.ndim != 1
                or probabilities.shape != indices.shape
                or indices.size == 0
            ):
                raise ValueError("稀疏策略索引和概率必须是一维、非空且等长")
            if np.any(indices < 0) or np.any(indices >= ACTION_SIZE):
                raise ValueError("策略动作索引越界")
            if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
                raise ValueError("策略概率必须有限且非负")
            if not np.isclose(float(probabilities.sum()), 1.0, atol=1e-5):
                raise ValueError("策略概率之和必须为 1")
            if not np.isfinite(sample.value) or sample.value < -1 or sample.value > 1:
                raise ValueError("价值目标必须位于 [-1, 1]")
            states.append(state)
            all_indices.append(indices)
            all_probabilities.append(probabilities)
            offsets.append(offsets[-1] + indices.size)
            values.append(float(sample.value))
            sides.append(0 if sample.side.value == "red" else 1)
        if not states:
            raise ValueError("不能提交没有训练样本的棋局")
        return {
            "schema_version": np.asarray([SCHEMA_VERSION], dtype=np.int64),
            "encoding_version": np.asarray([ENCODING_VERSION], dtype=np.int64),
            "action_version": np.asarray([ACTION_VERSION], dtype=np.int64),
            "states": np.stack(states).astype(np.float32, copy=False),
            "policy_indices": np.concatenate(all_indices).astype(np.int64, copy=False),
            "policy_probabilities": np.concatenate(all_probabilities).astype(
                np.float32, copy=False
            ),
            "policy_offsets": np.asarray(offsets, dtype=np.int64),
            "values": np.asarray(values, dtype=np.float32),
            "sides": np.asarray(sides, dtype=np.int8),
            "plies": np.asarray([game.plies], dtype=np.int64),
        }

    def _read_manifest(self) -> dict[str, object]:
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ReplayCompatibilityError("Replay manifest 无法读取") from error
        expected = {
            "schema_version": SCHEMA_VERSION,
            "encoding_version": ENCODING_VERSION,
            "action_version": ACTION_VERSION,
        }
        for name, version in expected.items():
            if manifest.get(name) != version:
                raise ReplayCompatibilityError(
                    f"Replay {name} 不兼容：{manifest.get(name)!r} != {version}"
                )
        games = manifest.get("games")
        next_game_id = manifest.get("next_game_id")
        if not isinstance(games, list) or not all(type(item) is int for item in games):
            raise ReplayCompatibilityError("Replay games 清单无效")
        if type(next_game_id) is not int or next_game_id <= 0:
            raise ReplayCompatibilityError("Replay next_game_id 无效")
        if (
            any(game_id <= 0 for game_id in games)
            or games != sorted(set(games))
            or (games and next_game_id <= games[-1])
        ):
            raise ReplayCompatibilityError("Replay 棋局 ID 或 next_game_id 无效")
        for game_id in games:
            if not self._game_path(game_id).is_file():
                raise ReplayCompatibilityError(f"Replay 棋局文件缺失：{game_id}")
        return manifest

    def _write_manifest(self, manifest: dict[str, object]) -> None:
        payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode(
            "utf-8"
        )
        temporary = self._temporary_path(self.path, "manifest.json")
        try:
            with temporary.open("wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.manifest_path)
            self._fsync_directory(self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def _game_path(self, game_id: int) -> Path:
        return self.games_path / f"{game_id:012d}.npz"

    @staticmethod
    def _temporary_path(directory: Path, prefix: str) -> Path:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{prefix}.", suffix=".tmp", dir=directory
        )
        os.close(descriptor)
        return Path(name)

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
