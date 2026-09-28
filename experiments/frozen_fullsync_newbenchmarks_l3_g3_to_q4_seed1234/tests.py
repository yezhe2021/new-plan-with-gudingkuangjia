from decimal import Decimal

from evaluate import canonical_number, gsm_options, source_rank


assert [source_rank(4, 2, i) for i in range(2)] == [1, 3]
assert [source_rank(2, 4, i) for i in range(4)] == [0, 0, 1, 1]
assert canonical_number(Decimal("18.00")) == "18"
options, gold = gsm_options("work\n#### 18", 1234)
assert len(options) == len(set(options)) == 4
assert options[gold] == "18"
print("CPU unit tests passed")
