"""Table metaclass for dimension entities.

Every dimension entity is a @dataclass with metaclass=Table. The metaclass provides
generate(), write(), pick_existing(), and pick_existing_object() class methods.
Subclasses must implement _make_one(i) -> instance.
"""

from __future__ import annotations

import random
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Any, Callable

import pyarrow as pa
import pyarrow.parquet as pq


class Table(type):
    """Metaclass that adds generation and serialization to dataclasses."""

    def __new__(mcs, name: str, bases: tuple, namespace: dict) -> Table:
        cls = super().__new__(mcs, name, bases, namespace)
        cls.instances: list = []
        return cls

    def generate(cls, count: int) -> None:
        """Generate `count` instances using cls._make_one(i) for i in 1..count."""
        cls.instances = [cls._make_one(i) for i in range(1, count + 1)]

    def write(cls, path: Path) -> None:
        """Write all instances to a Parquet file at `path`."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = cls.as_arrays()
        table = pa.table(arrays)
        pq.write_table(table, path)

    def pick_existing(cls, field_name: str) -> Any:
        """Return a random value of `field_name` from existing instances."""
        return getattr(random.choice(cls.instances), field_name)

    def pick_existing_object(cls, filter_fn: Callable | None = None) -> Any:
        """Return a random instance, optionally filtered."""
        pool = cls.instances if filter_fn is None else [x for x in cls.instances if filter_fn(x)]
        return random.choice(pool)

    def as_arrays(cls) -> dict[str, list]:
        """Return instance data as a dict of lists (column-oriented)."""
        if not cls.instances:
            return {}
        field_names = [f.name for f in dc_fields(cls.instances[0]) if not f.name.startswith("_")]
        return {name: [getattr(inst, name) for inst in cls.instances] for name in field_names}
