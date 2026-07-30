from dataclasses import dataclass
from enum import StrEnum


class Color(StrEnum):
    RED = "red"
    BLACK = "black"

    @property
    def opponent(self) -> "Color":
        return Color.BLACK if self is Color.RED else Color.RED


class PieceType(StrEnum):
    GENERAL = "general"
    ADVISOR = "advisor"
    ELEPHANT = "elephant"
    HORSE = "horse"
    ROOK = "rook"
    CANNON = "cannon"
    PAWN = "pawn"


@dataclass(frozen=True, slots=True)
class Coord:
    file: int
    rank: int

    def __post_init__(self) -> None:
        if not 0 <= self.file < 9 or not 0 <= self.rank < 10:
            raise ValueError(f"坐标越界: ({self.file}, {self.rank})")


@dataclass(frozen=True, slots=True)
class Piece:
    color: Color
    kind: PieceType


@dataclass(frozen=True, slots=True)
class Move:
    start: Coord
    end: Coord
    piece: Piece
    captured: Piece | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "start": [self.start.file, self.start.rank],
            "end": [self.end.file, self.end.rank],
            "piece": {"color": self.piece.color, "kind": self.piece.kind},
            "captured": (
                None
                if self.captured is None
                else {
                    "color": self.captured.color,
                    "kind": self.captured.kind,
                }
            ),
        }
