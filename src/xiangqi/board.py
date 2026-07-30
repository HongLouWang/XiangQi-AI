from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType

from xiangqi.domain import Color, Coord, Piece, PieceType

_FEN_TO_PIECE_TYPE = {
    "k": PieceType.GENERAL,
    "a": PieceType.ADVISOR,
    "e": PieceType.ELEPHANT,
    "h": PieceType.HORSE,
    "r": PieceType.ROOK,
    "c": PieceType.CANNON,
    "p": PieceType.PAWN,
}
_PIECE_TYPE_TO_FEN = {kind: symbol for symbol, kind in _FEN_TO_PIECE_TYPE.items()}


@dataclass(frozen=True, slots=True)
class Board:
    pieces: Mapping[Coord, Piece]

    def __post_init__(self) -> None:
        object.__setattr__(self, "pieces", MappingProxyType(dict(self.pieces)))

    @classmethod
    def empty(cls) -> Board:
        return cls({})

    @classmethod
    def standard(cls) -> Board:
        pieces: dict[Coord, Piece] = {}
        back_rank = (
            PieceType.ROOK,
            PieceType.HORSE,
            PieceType.ELEPHANT,
            PieceType.ADVISOR,
            PieceType.GENERAL,
            PieceType.ADVISOR,
            PieceType.ELEPHANT,
            PieceType.HORSE,
            PieceType.ROOK,
        )
        for color, rank in ((Color.BLACK, 0), (Color.RED, 9)):
            for file, kind in enumerate(back_rank):
                pieces[Coord(file, rank)] = Piece(color, kind)
        for color, rank in ((Color.BLACK, 2), (Color.RED, 7)):
            for file in (1, 7):
                pieces[Coord(file, rank)] = Piece(color, PieceType.CANNON)
        for color, rank in ((Color.BLACK, 3), (Color.RED, 6)):
            for file in range(0, 9, 2):
                pieces[Coord(file, rank)] = Piece(color, PieceType.PAWN)
        return cls(pieces)

    def at(self, coord: Coord) -> Piece | None:
        return self.pieces.get(coord)

    def place(self, coord: Coord, piece: Piece) -> Board:
        pieces = dict(self.pieces)
        pieces[coord] = piece
        return Board(pieces)

    def remove(self, coord: Coord) -> Board:
        pieces = dict(self.pieces)
        pieces.pop(coord, None)
        return Board(pieces)

    def move_unchecked(self, start: Coord, end: Coord) -> Board:
        piece = self.at(start)
        if piece is None:
            raise ValueError(f"起点没有棋子: ({start.file}, {start.rank})")
        pieces = dict(self.pieces)
        del pieces[start]
        pieces[end] = piece
        return Board(pieces)

    def to_fen(self) -> str:
        ranks: list[str] = []
        for rank in range(10):
            cells: list[str] = []
            empty_count = 0
            for file in range(9):
                piece = self.at(Coord(file, rank))
                if piece is None:
                    empty_count += 1
                    continue
                if empty_count:
                    cells.append(str(empty_count))
                    empty_count = 0
                symbol = _PIECE_TYPE_TO_FEN[piece.kind]
                cells.append(symbol.upper() if piece.color is Color.RED else symbol)
            if empty_count:
                cells.append(str(empty_count))
            ranks.append("".join(cells))
        return "/".join(ranks)

    @classmethod
    def from_fen(cls, fen: str) -> Board:
        rank_fields = fen.split("/")
        if len(rank_fields) != 10:
            raise ValueError("FEN 必须包含 10 行")

        pieces: dict[Coord, Piece] = {}
        for rank, field in enumerate(rank_fields):
            file = 0
            for symbol in field:
                if symbol.isdigit():
                    empty_count = int(symbol)
                    if not 1 <= empty_count <= 9:
                        raise ValueError(f"FEN 空位数无效: {symbol}")
                    file += empty_count
                    continue
                kind = _FEN_TO_PIECE_TYPE.get(symbol.lower())
                if kind is None:
                    raise ValueError(f"FEN 棋子符号无效: {symbol}")
                if file >= 9:
                    raise ValueError(f"FEN 第 {rank + 1} 行超过 9 列")
                color = Color.RED if symbol.isupper() else Color.BLACK
                pieces[Coord(file, rank)] = Piece(color, kind)
                file += 1
            if file != 9:
                raise ValueError(f"FEN 第 {rank + 1} 行不是 9 列")
        return cls(pieces)

    def position_key(self, side_to_move: Color) -> str:
        position = f"{self.to_fen()} {side_to_move.value}"
        return sha256(position.encode("ascii")).hexdigest()
