"""CPU-only unit tests for ordered Sync unit copy/drop mapping."""

from fullsync_v0 import source_rank


def mapped(m, n):
    return [source_rank(m, n, index) for index in range(n)]


def test_copy_drop_examples():
    assert mapped(1, 1) == [0]
    assert mapped(1, 3) == [0, 0, 0]
    assert mapped(3, 1) == [2]
    assert mapped(3, 2) == [1, 2]
    assert mapped(2, 3) == [0, 0, 1]
    assert mapped(4, 4) == [0, 1, 2, 3]


def test_ordered_and_causal_right_boundary():
    for m in range(1, 9):
        for n in range(1, 9):
            indices = mapped(m, n)
            assert len(indices) == n
            assert indices == sorted(indices)
            assert 0 <= indices[0] <= indices[-1] < m
            assert indices[-1] == m - 1


if __name__ == "__main__":
    test_copy_drop_examples()
    test_ordered_and_causal_right_boundary()
    print("Full-Sync copy/drop mapping tests passed", flush=True)
