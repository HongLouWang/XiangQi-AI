from __future__ import annotations

import hashlib
import json
import os
import tempfile
from bisect import bisect_right
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ai.encoding import ACTION_SIZE, INPUT_CHANNELS
from ai.self_play import GameResult

SCHEMA_VERSION = 2
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
        self.migration_path = self.path / "migration-v1-to-v2.json"
        self.capacity_games = capacity_games
        self._legacy_manifest_hash: str | None = None
        self.games_path.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            self._manifest = self._read_manifest()
        else:
            self._manifest = {
                "schema_version": SCHEMA_VERSION,
                "encoding_version": ENCODING_VERSION,
                "action_version": ACTION_VERSION,
                "next_game_id": 1,
                "total_games": 0,
                "games": [],
                "sample_counts": {},
            }
            self._write_manifest(self._manifest)

    @property
    def game_ids(self) -> tuple[int, ...]:
        return tuple(self._manifest["games"])

    @property
    def game_count(self) -> int:
        """返回当前持久化 Replay 中的完整棋局数量。"""
        return len(self.game_ids)

    @property
    def total_games(self) -> int:
        """返回历史上已经原子提交的棋局总数，容量淘汰不会减少它。"""
        return int(self._manifest["total_games"])

    @property
    def sample_count(self) -> int:
        """返回当前持久化 Replay 中可训练局面样本总数。"""
        counts = self._manifest["sample_counts"]
        return sum(int(counts[str(game_id)]) for game_id in self.game_ids)

    @property
    def manifest_hash(self) -> str:
        return hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()

    @property
    def manifest_version(self) -> int:
        return SCHEMA_VERSION

    @property
    def legacy_manifest_hash(self) -> str | None:
        """仅在本对象刚完成 v1 到 v2 迁移时返回旧 manifest 哈希。"""
        return self._legacy_manifest_hash

    def clear_migration(self) -> None:
        """在新 checkpoint 落盘后耐久撤销旧 checkpoint 的一次性豁免。"""
        if self.migration_path.exists():
            payload = self.migration_path.read_bytes()
            try:
                self.migration_path.unlink()
                self._fsync_directory(self.path)
            except OSError:
                if not self.migration_path.exists():
                    self._write_bytes_atomic(self.migration_path, payload)
                raise
        self._legacy_manifest_hash = None

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
        old_counts = dict(self._manifest["sample_counts"])
        sample_counts = {
            str(retained_id): (
                len(game.samples)
                if retained_id == game_id
                else int(old_counts[str(retained_id)])
            )
            for retained_id in new_games
        }
        updated = {
            **self._manifest,
            "next_game_id": game_id + 1,
            "total_games": game_id,
            "games": new_games,
            "sample_counts": sample_counts,
        }
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

        counts = [
            int(self._manifest["sample_counts"][str(game_id)])
            for game_id in self.game_ids
        ]
        cumulative = np.cumsum(counts, dtype=np.int64)
        total_samples = int(cumulative[-1]) if cumulative.size else 0
        if batch_size > total_samples:
            raise ValueError(
                f"样本不足：请求 {batch_size}，Replay 中只有 {total_samples}"
            )

        selected = rng.choice(total_samples, size=batch_size, replace=False)
        cache: dict[int, dict[str, NDArray[np.generic]]] = {}
        states: list[NDArray[np.float32]] = []
        indices: list[NDArray[np.int64]] = []
        probabilities: list[NDArray[np.float32]] = []
        values: list[float] = []
        for selection in np.asarray(selected).reshape(-1):
            flat_index = int(selection)
            game_offset = bisect_right(cumulative, flat_index)
            game_id = self.game_ids[game_offset]
            previous = 0 if game_offset == 0 else int(cumulative[game_offset - 1])
            sample_index = flat_index - previous
            if game_id not in cache:
                cache[game_id] = self._read_game(game_id)
                actual_count = int(cache[game_id]["values"].shape[0])
                if actual_count != counts[game_offset]:
                    raise ReplayCompatibilityError(
                        f"Replay 棋局 {game_id} 的 sample_count 与 manifest 不一致"
                    )
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

    def _read_game(
        self, game_id: int, *, schema_versions: tuple[int, ...] = (1, 2)
    ) -> dict[str, NDArray[np.generic]]:
        try:
            with np.load(self._game_path(game_id), allow_pickle=False) as stored:
                data = {key: stored[key].copy() for key in stored.files}
        except (OSError, ValueError, KeyError) as error:
            raise ReplayCompatibilityError(f"Replay 棋局 {game_id} 无法读取") from error
        expected = {
            "encoding_version": (ENCODING_VERSION,),
            "action_version": (ACTION_VERSION,),
            "schema_version": schema_versions,
        }
        for name, accepted_versions in expected.items():
            try:
                version_array = np.asarray(data[name])
                if version_array.shape != (1,) or not np.issubdtype(
                    version_array.dtype, np.integer
                ):
                    raise ValueError
                actual = int(version_array[0])
            except (KeyError, IndexError, TypeError, ValueError) as error:
                raise ReplayCompatibilityError(
                    f"Replay 棋局 {game_id} 缺少有效 {name}"
                ) from error
            if actual not in accepted_versions:
                raise ReplayCompatibilityError(
                    f"Replay 棋局 {game_id} 的 {name} 不兼容："
                    f"{actual} not in {accepted_versions}"
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
        plies = data["plies"]
        sample_count = int(values.shape[0]) if values.ndim == 1 else -1
        valid_offsets = (
            offsets.ndim == 1
            and np.issubdtype(offsets.dtype, np.integer)
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
            not np.all(np.isfinite(states))
            or not np.issubdtype(sides.dtype, np.integer)
            or np.any((sides != 0) & (sides != 1))
            or plies.shape != (1,)
            or not np.issubdtype(plies.dtype, np.integer)
            or int(plies[0]) < 0
            or int(plies[0]) != sample_count
        ):
            raise ReplayCompatibilityError(f"Replay 棋局 {game_id} 的局面元数据无效")
        if (
            not np.issubdtype(indices.dtype, np.integer)
            or indices.ndim != 1
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
            sample_indices = indices[int(start) : int(end)]
            if np.unique(sample_indices).size != sample_indices.size:
                raise ReplayCompatibilityError(f"Replay 棋局 {game_id} 的动作索引重复")
            if not np.isclose(float(probabilities[int(start) : int(end)].sum()), 1.0):
                raise ReplayCompatibilityError(
                    f"Replay 棋局 {game_id} 的策略概率之和无效"
                )
        return data

    def _encode_game(self, game: GameResult) -> dict[str, NDArray[np.generic]]:
        if type(game.plies) is not int or game.plies < 0:
            raise ValueError("棋局 plies 必须是非负整数")
        if game.plies != len(game.samples):
            raise ValueError("棋局 plies 必须等于训练样本数")
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
            if not np.all(np.isfinite(state)):
                raise ValueError("局面特征必须全部为有限数")
            if (
                indices.ndim != 1
                or probabilities.shape != indices.shape
                or indices.size == 0
            ):
                raise ValueError("稀疏策略索引和概率必须是一维、非空且等长")
            if np.any(indices < 0) or np.any(indices >= ACTION_SIZE):
                raise ValueError("策略动作索引越界")
            if np.unique(indices).size != indices.size:
                raise ValueError("单个样本的策略动作索引不能重复")
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
            raw = self.manifest_path.read_bytes()
            manifest = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ReplayCompatibilityError("Replay manifest 无法读取") from error
        if manifest.get("schema_version") == 1:
            manifest = self._migrate_v1_manifest(manifest, raw)
            raw = self.manifest_path.read_bytes()
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
        total_games = manifest.get("total_games")
        sample_counts = manifest.get("sample_counts")
        if not isinstance(games, list) or not all(type(item) is int for item in games):
            raise ReplayCompatibilityError("Replay games 清单无效")
        if type(next_game_id) is not int or next_game_id <= 0:
            raise ReplayCompatibilityError("Replay next_game_id 无效")
        if type(total_games) is not int or total_games < 0:
            raise ReplayCompatibilityError("Replay total_games 无效")
        if total_games != next_game_id - 1:
            raise ReplayCompatibilityError("Replay total_games 与 next_game_id 不一致")
        if (
            any(game_id <= 0 for game_id in games)
            or games != sorted(set(games))
            or (games and next_game_id <= games[-1])
            or (games and games[-1] != total_games)
            or games != list(range(total_games - len(games) + 1, total_games + 1))
        ):
            raise ReplayCompatibilityError("Replay 棋局 ID 或 next_game_id 无效")
        if (
            not isinstance(sample_counts, dict)
            or set(sample_counts) != {str(game_id) for game_id in games}
            or any(
                type(value) is not int or value <= 0 for value in sample_counts.values()
            )
        ):
            raise ReplayCompatibilityError("Replay sample_counts 清单无效")
        for game_id in games:
            if not self._game_path(game_id).is_file():
                raise ReplayCompatibilityError(f"Replay 棋局文件缺失：{game_id}")
            data = self._read_game(game_id)
            if int(data["values"].shape[0]) != int(sample_counts[str(game_id)]):
                raise ReplayCompatibilityError(
                    f"Replay 棋局 {game_id} 的 sample_count 与 manifest 不一致"
                )
        if self.migration_path.exists():
            sidecar = self._read_migration_sidecar()
            self._validate_migration_sidecar(
                sidecar,
                legacy_hash=None,
                new_hash=hashlib.sha256(raw).hexdigest(),
                total_games=total_games,
            )
            self._legacy_manifest_hash = str(sidecar["legacy_manifest_hash"])
        return manifest

    def _migrate_v1_manifest(
        self, manifest: dict[str, object], raw: bytes
    ) -> dict[str, object]:
        expected = {
            "schema_version": 1,
            "encoding_version": ENCODING_VERSION,
            "action_version": ACTION_VERSION,
        }
        for name, version in expected.items():
            if manifest.get(name) != version:
                raise ReplayCompatibilityError(
                    f"Replay v1 {name} 不兼容：{manifest.get(name)!r} != {version}"
                )
        if set(manifest) != {*expected, "next_game_id", "games"}:
            raise ReplayCompatibilityError("Replay v1 manifest 字段无效")
        games = manifest.get("games")
        next_game_id = manifest.get("next_game_id")
        if not isinstance(games, list) or not all(type(item) is int for item in games):
            raise ReplayCompatibilityError("Replay v1 games 清单无效")
        if type(next_game_id) is not int or next_game_id <= 0:
            raise ReplayCompatibilityError("Replay v1 next_game_id 无效")
        total_games = next_game_id - 1
        if (
            any(game_id <= 0 for game_id in games)
            or games != sorted(set(games))
            or (games and games[-1] != total_games)
            or games != list(range(total_games - len(games) + 1, total_games + 1))
        ):
            raise ReplayCompatibilityError("Replay v1 棋局 ID 或 next_game_id 无效")
        sample_counts: dict[str, int] = {}
        for game_id in games:
            if not self._game_path(game_id).is_file():
                raise ReplayCompatibilityError(f"Replay v1 棋局文件缺失：{game_id}")
            data = self._read_game(game_id, schema_versions=(1,))
            sample_counts[str(game_id)] = int(data["values"].shape[0])
        migrated = {
            **manifest,
            "schema_version": SCHEMA_VERSION,
            "total_games": total_games,
            "sample_counts": sample_counts,
        }
        legacy_hash = hashlib.sha256(raw).hexdigest()
        new_hash = hashlib.sha256(self._manifest_payload(migrated)).hexdigest()
        if self.migration_path.exists():
            sidecar = self._read_migration_sidecar()
            self._validate_migration_sidecar(
                sidecar,
                legacy_hash=legacy_hash,
                new_hash=new_hash,
                total_games=total_games,
            )
        else:
            self._write_migration_sidecar(legacy_hash, new_hash, total_games)
        # sidecar 已耐久发布后才原子发布 manifest；任一步失败都可由下次启动重放。
        self._write_manifest(migrated)
        self._legacy_manifest_hash = legacy_hash
        return migrated

    def _write_migration_sidecar(
        self, legacy_hash: str, new_hash: str, total_games: int
    ) -> None:
        sidecar: dict[str, object] = {
            "schema_version": 1,
            "legacy_manifest_hash": legacy_hash,
            "legacy_version": 1,
            "new_manifest_hash": new_hash,
            "new_version": SCHEMA_VERSION,
            "total_games": total_games,
        }
        self._write_bytes_atomic(
            self.migration_path,
            json.dumps(sidecar, ensure_ascii=False, sort_keys=True).encode("utf-8"),
        )

    def _read_migration_sidecar(self) -> dict[str, object]:
        try:
            value = json.loads(self.migration_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ReplayCompatibilityError(
                "Replay migration sidecar 无法读取"
            ) from error
        if not isinstance(value, dict):
            raise ReplayCompatibilityError("Replay migration sidecar 必须是对象")
        return value

    @staticmethod
    def _validate_migration_sidecar(
        sidecar: dict[str, object],
        *,
        legacy_hash: str | None,
        new_hash: str,
        total_games: int,
    ) -> None:
        fields = {
            "schema_version",
            "legacy_manifest_hash",
            "legacy_version",
            "new_manifest_hash",
            "new_version",
            "total_games",
        }
        if set(sidecar) != fields:
            raise ReplayCompatibilityError("Replay migration sidecar 字段无效")
        hashes = (
            sidecar.get("legacy_manifest_hash"),
            sidecar.get("new_manifest_hash"),
        )
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
            for value in hashes
        ):
            raise ReplayCompatibilityError("Replay migration sidecar hash 无效")
        if (
            type(sidecar.get("schema_version")) is not int
            or sidecar.get("schema_version") != 1
            or type(sidecar.get("legacy_version")) is not int
            or sidecar.get("legacy_version") != 1
            or type(sidecar.get("new_version")) is not int
            or sidecar.get("new_version") != SCHEMA_VERSION
            or type(sidecar.get("total_games")) is not int
            or sidecar.get("total_games") != total_games
            or sidecar.get("new_manifest_hash") != new_hash
            or (
                legacy_hash is not None
                and sidecar.get("legacy_manifest_hash") != legacy_hash
            )
        ):
            raise ReplayCompatibilityError("Replay migration sidecar 内容不匹配")

    def _write_manifest(self, manifest: dict[str, object]) -> None:
        self._write_bytes_atomic(self.manifest_path, self._manifest_payload(manifest))

    @staticmethod
    def _manifest_payload(manifest: dict[str, object]) -> bytes:
        return json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")

    def _write_bytes_atomic(self, destination: Path, payload: bytes) -> None:
        temporary = self._temporary_path(destination.parent, destination.name)
        try:
            with temporary.open("wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            self._fsync_directory(destination.parent)
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
