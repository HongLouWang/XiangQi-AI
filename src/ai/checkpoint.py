from __future__ import annotations

import json
import os
import random
import tempfile
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ai.config import TrainingConfig
from ai.encoding import ACTION_SIZE, INPUT_CHANNELS

CHECKPOINT_SCHEMA_VERSION = 1
ENCODING_VERSION = 1
ACTION_VERSION = 1


class CheckpointCompatibilityError(RuntimeError):
    """检查点存在，但不能由当前训练代码安全恢复。"""


@dataclass(frozen=True, slots=True)
class TrainingProgress:
    completed_games: int
    target_games: int
    training_steps: int

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 0 for value in asdict(self).values()):
            raise ValueError("训练进度必须是非负整数")
        if self.target_games < self.completed_games:
            raise ValueError("target_games 不能小于 completed_games")


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    progress: TrainingProgress
    config: TrainingConfig
    replay_manifest_hash: str
    replay_manifest_version: int
    generation: int


class CheckpointManager:
    """使用两个轮换槽保存可恢复的完整训练状态。"""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.pointer_path = self.path / "latest.json"
        self._slots = {
            "a": self.path / "checkpoint-a.pt",
            "b": self.path / "checkpoint-b.pt",
        }

    @property
    def latest_path(self) -> Path:
        pointer = self._read_pointer()
        return self._slots[pointer["slot"]]

    def has_checkpoint(self) -> bool:
        return any(path.is_file() for path in self._slots.values())

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        progress: TrainingProgress,
        config: TrainingConfig,
        *,
        replay_manifest_hash: str,
        replay_manifest_version: int,
        scheduler: Any | None = None,
        numpy_generator: np.random.Generator | None = None,
    ) -> Path:
        if not replay_manifest_hash:
            raise ValueError("replay_manifest_hash 不能为空")
        if type(replay_manifest_version) is not int or replay_manifest_version <= 0:
            raise ValueError("replay_manifest_version 必须是正整数")
        current = self._pointer_or_none()
        slot = "b" if current is not None and current["slot"] == "a" else "a"
        generation = 1 if current is None else int(current["generation"]) + 1
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "encoding_version": ENCODING_VERSION,
            "action_version": ACTION_VERSION,
            "input_channels": INPUT_CHANNELS,
            "action_size": ACTION_SIZE,
            "generation": generation,
            "config": self._serialize_config(config),
            "progress": asdict(progress),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": None if scheduler is None else scheduler.state_dict(),
            "replay_manifest_hash": replay_manifest_hash,
            "replay_manifest_version": replay_manifest_version,
            "python_rng_state": self._encode_python_rng(random.getstate()),
            "numpy_rng_state": self._encode_numpy_rng(np.random.get_state()),
            "numpy_generator_state": None
            if numpy_generator is None
            else self._encode_safe(deepcopy(numpy_generator.bit_generator.state)),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        }
        destination = self._slots[slot]
        temporary = self._temporary_path(destination.name)
        try:
            with temporary.open("wb") as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            self._fsync_directory()
        finally:
            temporary.unlink(missing_ok=True)
        self._write_pointer(slot, generation)
        return destination

    def load_latest(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        expected_config: TrainingConfig | None = None,
        scheduler: Any | None = None,
        map_location: str | torch.device = "cpu",
        restore_rng: bool = True,
        numpy_generator: np.random.Generator | None = None,
        expected_replay_manifest_hash: str | None = None,
        expected_replay_manifest_version: int | None = None,
        allow_replay_forward: bool = False,
        expected_replay_total_games: int | None = None,
        accepted_replay_predecessor_hash: str | None = None,
    ) -> LoadedCheckpoint:
        errors: list[str] = []
        compatibility_errors: list[CheckpointCompatibilityError] = []
        for slot in self._candidate_slots():
            try:
                payload = torch.load(
                    self._slots[slot], map_location=map_location, weights_only=True
                )
                loaded = self._validate_payload(payload)
                self._validate_model_state(model, payload["model_state"])
            except CheckpointCompatibilityError as error:
                compatibility_errors.append(error)
                errors.append(f"{slot}: {error}")
                continue
            except Exception as error:  # noqa: BLE001 - 损坏存档可抛出多种安全异常
                errors.append(f"{slot}: {error}")
                continue

            self._validate_caller_compatibility(
                loaded,
                expected_config,
                expected_replay_manifest_hash,
                expected_replay_manifest_version,
                allow_replay_forward,
                expected_replay_total_games,
                accepted_replay_predecessor_hash,
            )
            snapshots = self._capture_restore_snapshots(
                model, optimizer, scheduler, numpy_generator
            )
            try:
                model.load_state_dict(payload["model_state"])
                try:
                    optimizer.load_state_dict(payload["optimizer_state"])
                except (KeyError, TypeError, ValueError) as error:
                    raise CheckpointCompatibilityError(
                        "checkpoint optimizer_state 与当前优化器不兼容"
                    ) from error
                if scheduler is not None:
                    state = payload.get("scheduler_state")
                    if state is None:
                        raise CheckpointCompatibilityError(
                            "checkpoint 不含 scheduler_state"
                        )
                    scheduler.load_state_dict(state)
                if restore_rng:
                    self._restore_rng(payload, numpy_generator)
                return loaded
            except Exception as error:  # noqa: BLE001 - 损坏的 torch 存档可能抛出多种异常
                self._rollback_restore(
                    snapshots, model, optimizer, scheduler, numpy_generator
                )
                errors.append(f"{slot}: {error}")
        if compatibility_errors and len(compatibility_errors) == len(errors):
            raise compatibility_errors[-1]
        raise RuntimeError("没有可加载的 checkpoint；" + "; ".join(errors))

    def export_model(self, model: nn.Module, destination: Path | str) -> Path:
        output = Path(destination)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._temporary_path(output.name, directory=output.parent)
        try:
            with temporary.open("wb") as stream:
                torch.save(model.state_dict(), stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, output)
            self._fsync_directory(output.parent)
        finally:
            temporary.unlink(missing_ok=True)
        return output

    def _validate_payload(self, payload: object) -> LoadedCheckpoint:
        if not isinstance(payload, dict):
            raise TypeError("checkpoint payload 不是字典")
        required = (
            "config",
            "progress",
            "generation",
            "model_state",
            "optimizer_state",
            "scheduler_state",
            "replay_manifest_hash",
            "replay_manifest_version",
            "python_rng_state",
            "numpy_rng_state",
            "numpy_generator_state",
            "torch_rng_state",
            "cuda_rng_states",
        )
        for name in required:
            if name not in payload:
                raise CheckpointCompatibilityError(f"checkpoint 缺少必需字段 {name}")
        versions = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "encoding_version": ENCODING_VERSION,
            "action_version": ACTION_VERSION,
            "input_channels": INPUT_CHANNELS,
            "action_size": ACTION_SIZE,
        }
        for name, expected in versions.items():
            if payload.get(name) != expected:
                raise CheckpointCompatibilityError(
                    f"checkpoint {name} 不兼容：{payload.get(name)!r} != {expected}"
                )
        if not isinstance(payload["replay_manifest_hash"], str):
            raise CheckpointCompatibilityError("checkpoint replay_manifest_hash 无效")
        if (
            type(payload["replay_manifest_version"]) is not int
            or payload["replay_manifest_version"] <= 0
        ):
            raise CheckpointCompatibilityError(
                "checkpoint replay_manifest_version 无效"
            )
        if type(payload["generation"]) is not int or payload["generation"] <= 0:
            raise CheckpointCompatibilityError("checkpoint generation 无效")
        try:
            config_data = dict(payload["config"])
            config_data.setdefault("parallel_games", 16)
            config_data["run_dir"] = Path(config_data["run_dir"])
            config = TrainingConfig(**config_data)
            progress = TrainingProgress(**payload["progress"])
        except (KeyError, TypeError, ValueError) as error:
            raise CheckpointCompatibilityError("checkpoint 配置或进度无效") from error
        return LoadedCheckpoint(
            progress=progress,
            config=config,
            replay_manifest_hash=payload["replay_manifest_hash"],
            replay_manifest_version=payload["replay_manifest_version"],
            generation=int(payload["generation"]),
        )

    @staticmethod
    def _validate_caller_compatibility(
        loaded: LoadedCheckpoint,
        expected_config: TrainingConfig | None,
        expected_replay_manifest_hash: str | None,
        expected_replay_manifest_version: int | None,
        allow_replay_forward: bool,
        expected_replay_total_games: int | None,
        accepted_replay_predecessor_hash: str | None,
    ) -> None:
        if expected_config is not None:
            for name in ("channels", "residual_blocks"):
                actual = getattr(loaded.config, name)
                expected = getattr(expected_config, name)
                if actual != expected:
                    raise CheckpointCompatibilityError(
                        f"checkpoint {name} 不兼容：{actual} != {expected}"
                    )
        if expected_replay_manifest_hash is None:
            if (
                expected_replay_manifest_version is not None
                and loaded.replay_manifest_version != expected_replay_manifest_version
            ):
                raise CheckpointCompatibilityError(
                    "checkpoint 与当前 Replay manifest version 不一致"
                )
            return
        exact_current = loaded.replay_manifest_hash == expected_replay_manifest_hash
        exact_predecessor = (
            accepted_replay_predecessor_hash is not None
            and loaded.replay_manifest_hash == accepted_replay_predecessor_hash
        )
        if exact_current:
            if (
                expected_replay_manifest_version is not None
                and loaded.replay_manifest_version != expected_replay_manifest_version
            ):
                raise CheckpointCompatibilityError(
                    "checkpoint 与当前 Replay manifest version 不一致"
                )
            return
        if exact_predecessor:
            if (
                loaded.replay_manifest_version != 1
                or type(expected_replay_total_games) is not int
                or expected_replay_total_games != loaded.progress.completed_games
            ):
                raise CheckpointCompatibilityError(
                    "旧 Replay manifest 只允许等进度的一次性迁移恢复"
                )
            return
        if not allow_replay_forward:
            raise CheckpointCompatibilityError(
                "checkpoint 与当前 Replay manifest hash 不一致"
            )
        if type(expected_replay_total_games) is not int:
            raise ValueError("前进恢复必须提供 Replay total_games")
        if expected_replay_total_games < loaded.progress.completed_games:
            raise CheckpointCompatibilityError(
                "当前 Replay 进度落后于 checkpoint，不能安全恢复"
            )
        if expected_replay_total_games == loaded.progress.completed_games:
            raise CheckpointCompatibilityError(
                "Replay 总局数未前进但 manifest hash 不一致"
            )

    @staticmethod
    def _validate_model_state(model: nn.Module, saved_state: object) -> None:
        if not isinstance(saved_state, dict):
            raise CheckpointCompatibilityError("checkpoint model_state 无效")
        current = model.state_dict()
        if current.keys() != saved_state.keys():
            raise CheckpointCompatibilityError(
                "checkpoint model_state 参数名称与当前模型不兼容"
            )
        for name, tensor in current.items():
            saved = saved_state[name]
            if not isinstance(saved, torch.Tensor) or saved.shape != tensor.shape:
                raise CheckpointCompatibilityError(
                    f"checkpoint model_state 参数形状不兼容：{name}"
                )

    @staticmethod
    def _restore_rng(
        payload: dict[str, Any], numpy_generator: np.random.Generator | None
    ) -> None:
        python_state = CheckpointManager._decode_python_rng(payload["python_rng_state"])
        numpy_state = CheckpointManager._decode_numpy_rng(payload["numpy_rng_state"])
        generator_state = payload["numpy_generator_state"]
        decoded_generator = (
            None
            if generator_state is None
            else CheckpointManager._decode_safe(generator_state)
        )
        if numpy_generator is not None:
            if decoded_generator is None:
                raise CheckpointCompatibilityError(
                    "checkpoint 不含实际 NumPy Generator 状态"
                )
            try:
                test_generator = type(numpy_generator.bit_generator)()
                test_generator.state = decoded_generator
            except (TypeError, ValueError) as error:
                raise CheckpointCompatibilityError(
                    "NumPy Generator 类型与 checkpoint 不兼容"
                ) from error
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if numpy_generator is not None:
            numpy_generator.bit_generator.state = decoded_generator
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        cuda_states = payload.get("cuda_rng_states")
        if cuda_states is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])

    @staticmethod
    def _encode_python_rng(state: tuple[Any, ...]) -> dict[str, object]:
        version, values, gaussian = state
        return {
            "version": int(version),
            "values": torch.tensor(values, dtype=torch.int64),
            "gaussian": gaussian,
        }

    @staticmethod
    def _decode_python_rng(state: object) -> tuple[Any, ...]:
        if not isinstance(state, dict) or not isinstance(
            state.get("values"), torch.Tensor
        ):
            raise CheckpointCompatibilityError("checkpoint Python RNG 状态无效")
        try:
            return (
                int(state["version"]),
                tuple(int(value) for value in state["values"].tolist()),
                state["gaussian"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise CheckpointCompatibilityError(
                "checkpoint Python RNG 状态无效"
            ) from error

    @staticmethod
    def _encode_numpy_rng(state: tuple[Any, ...]) -> dict[str, object]:
        algorithm, keys, position, has_gaussian, cached_gaussian = state
        return {
            "algorithm": str(algorithm),
            "keys": torch.from_numpy(np.asarray(keys, dtype=np.int64)),
            "position": int(position),
            "has_gaussian": int(has_gaussian),
            "cached_gaussian": float(cached_gaussian),
        }

    @staticmethod
    def _decode_numpy_rng(state: object) -> tuple[Any, ...]:
        if not isinstance(state, dict) or not isinstance(
            state.get("keys"), torch.Tensor
        ):
            raise CheckpointCompatibilityError("checkpoint NumPy RNG 状态无效")
        try:
            return (
                str(state["algorithm"]),
                state["keys"].cpu().numpy().astype(np.uint32),
                int(state["position"]),
                int(state["has_gaussian"]),
                float(state["cached_gaussian"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise CheckpointCompatibilityError(
                "checkpoint NumPy RNG 状态无效"
            ) from error

    @staticmethod
    def _encode_safe(value: object) -> object:
        if isinstance(value, np.ndarray):
            return {"__kind__": "ndarray", "value": torch.from_numpy(value.copy())}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {
                str(key): CheckpointManager._encode_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, tuple):
            return {
                "__kind__": "tuple",
                "value": [CheckpointManager._encode_safe(item) for item in value],
            }
        if isinstance(value, list):
            return [CheckpointManager._encode_safe(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool, torch.Tensor)):
            return value
        raise TypeError(f"不能安全持久化的 RNG 状态类型：{type(value).__name__}")

    @staticmethod
    def _decode_safe(value: object) -> object:
        if isinstance(value, dict):
            if value.get("__kind__") == "ndarray":
                tensor = value.get("value")
                if not isinstance(tensor, torch.Tensor):
                    raise CheckpointCompatibilityError("安全数组编码无效")
                return tensor.cpu().numpy().copy()
            if value.get("__kind__") == "tuple":
                items = value.get("value")
                if not isinstance(items, list):
                    raise CheckpointCompatibilityError("安全元组编码无效")
                return tuple(CheckpointManager._decode_safe(item) for item in items)
            return {
                key: CheckpointManager._decode_safe(item) for key, item in value.items()
            }
        if isinstance(value, list):
            return [CheckpointManager._decode_safe(item) for item in value]
        return value

    @staticmethod
    def _capture_restore_snapshots(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None,
        numpy_generator: np.random.Generator | None,
    ) -> dict[str, object]:
        return {
            "model": deepcopy(model.state_dict()),
            "optimizer": deepcopy(optimizer.state_dict()),
            "scheduler": None
            if scheduler is None
            else deepcopy(scheduler.state_dict()),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "generator_rng": None
            if numpy_generator is None
            else deepcopy(numpy_generator.bit_generator.state),
            "torch_rng": torch.get_rng_state().clone(),
            "cuda_rng": [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None,
        }

    @staticmethod
    def _rollback_restore(
        snapshots: dict[str, object],
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None,
        numpy_generator: np.random.Generator | None,
    ) -> None:
        model.load_state_dict(snapshots["model"])
        optimizer.load_state_dict(snapshots["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(snapshots["scheduler"])
        random.setstate(snapshots["python_rng"])
        np.random.set_state(snapshots["numpy_rng"])
        if numpy_generator is not None:
            numpy_generator.bit_generator.state = snapshots["generator_rng"]
        torch.set_rng_state(snapshots["torch_rng"])
        if snapshots["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(snapshots["cuda_rng"])

    def _candidate_slots(self) -> tuple[str, ...]:
        pointer = self._pointer_or_none()
        if pointer is not None:
            first = str(pointer["slot"])
            other = "b" if first == "a" else "a"
            return tuple(slot for slot in (first, other) if self._slots[slot].is_file())
        candidates: list[tuple[int, str]] = []
        for slot, path in self._slots.items():
            if not path.is_file():
                continue
            try:
                payload = torch.load(path, map_location="cpu", weights_only=True)
                candidates.append((int(payload["generation"]), slot))
            except Exception:  # noqa: BLE001, S112 - 扫描时忽略任意损坏槽
                continue
        candidates.sort(reverse=True)
        return tuple(slot for _, slot in candidates)

    def _pointer_or_none(self) -> dict[str, object] | None:
        try:
            return self._read_pointer()
        except Exception:  # noqa: BLE001 - 指针指向的 torch 槽可能以任意方式损坏
            return None

    def _read_pointer(self) -> dict[str, object]:
        pointer = json.loads(self.pointer_path.read_text(encoding="utf-8"))
        if (
            pointer.get("slot") not in self._slots
            or type(pointer.get("generation")) is not int
        ):
            raise ValueError("latest.json 无效")
        slot = str(pointer["slot"])
        payload = torch.load(self._slots[slot], map_location="cpu", weights_only=True)
        if (
            not isinstance(payload, dict)
            or payload.get("generation") != pointer["generation"]
        ):
            raise ValueError("latest.json 与 checkpoint generation 不一致")
        return pointer

    def _write_pointer(self, slot: str, generation: int) -> None:
        temporary = self._temporary_path("latest.json")
        try:
            payload = json.dumps(
                {"slot": slot, "generation": generation}, sort_keys=True
            )
            with temporary.open("w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.pointer_path)
            self._fsync_directory()
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _serialize_config(config: TrainingConfig) -> dict[str, object]:
        result = asdict(config)
        result["run_dir"] = str(config.run_dir)
        return result

    def _temporary_path(self, prefix: str, directory: Path | None = None) -> Path:
        target = self.path if directory is None else directory
        descriptor, name = tempfile.mkstemp(
            prefix=f".{prefix}.", suffix=".tmp", dir=target
        )
        os.close(descriptor)
        return Path(name)

    def _fsync_directory(self, directory: Path | None = None) -> None:
        descriptor = os.open(self.path if directory is None else directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
