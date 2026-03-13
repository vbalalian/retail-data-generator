from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq


def test_table_generate_creates_instances():
    from retail_gen.lib.table import Table

    @dataclass
    class Widget(metaclass=Table):
        id: int
        name: str

        @classmethod
        def _make_one(cls, i: int) -> "Widget":
            return cls(id=i, name=f"widget-{i}")

    Widget.generate(5)
    assert len(Widget.instances) == 5
    assert Widget.instances[0].id == 1
    assert Widget.instances[4].id == 5


def test_table_write_creates_parquet(tmp_path):
    from retail_gen.lib.table import Table

    @dataclass
    class Widget(metaclass=Table):
        id: int
        name: str

        @classmethod
        def _make_one(cls, i: int) -> "Widget":
            return cls(id=i, name=f"widget-{i}")

    Widget.generate(3)
    out = tmp_path / "widgets.parquet"
    Widget.write(out)
    table = pq.read_table(out)
    assert table.num_rows == 3
    assert set(table.column_names) == {"id", "name"}


def test_table_pick_existing_returns_valid_value():
    from retail_gen.lib.table import Table

    @dataclass
    class Widget(metaclass=Table):
        id: int
        name: str

        @classmethod
        def _make_one(cls, i: int) -> "Widget":
            return cls(id=i, name=f"widget-{i}")

    Widget.generate(10)
    val = Widget.pick_existing("id")
    assert 1 <= val <= 10


def test_table_pick_existing_object_with_filter():
    from retail_gen.lib.table import Table

    @dataclass
    class Widget(metaclass=Table):
        id: int
        name: str
        active: bool

        @classmethod
        def _make_one(cls, i: int) -> "Widget":
            return cls(id=i, name=f"widget-{i}", active=i % 2 == 0)

    Widget.generate(10)
    obj = Widget.pick_existing_object(lambda w: w.active)
    assert obj.active is True


def test_table_instances_isolated_between_classes():
    """Each Table subclass gets its own instances list."""
    from retail_gen.lib.table import Table

    @dataclass
    class Foo(metaclass=Table):
        id: int

        @classmethod
        def _make_one(cls, i: int) -> "Foo":
            return cls(id=i)

    @dataclass
    class Bar(metaclass=Table):
        id: int

        @classmethod
        def _make_one(cls, i: int) -> "Bar":
            return cls(id=i)

    Foo.generate(3)
    Bar.generate(5)
    assert len(Foo.instances) == 3
    assert len(Bar.instances) == 5


def test_table_as_arrays_returns_dict_of_lists():
    from retail_gen.lib.table import Table

    @dataclass
    class Widget(metaclass=Table):
        id: int
        name: str

        @classmethod
        def _make_one(cls, i: int) -> "Widget":
            return cls(id=i, name=f"widget-{i}")

    Widget.generate(3)
    arrays = Widget.as_arrays()
    assert arrays["id"] == [1, 2, 3]
    assert len(arrays["name"]) == 3
