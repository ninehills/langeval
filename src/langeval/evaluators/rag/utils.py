def str_inx(word_, string_):
    return [i for i in range(len(string_)) if string_[i] == word_]


def ab_max_inx(s_a, s_b):
    i, len_a, len_b = 0, len(s_a), len(s_b)
    while len_a > i and len_b > i and s_a[i] == s_b[i]:
        i += 1
    return i


def lcs(s_a, s_b):
    """计算两个字符串的所有不重复的公共字串"""
    res = []
    if s_a:
        a0_inx_in_b = str_inx(s_a[0], s_b)
        if a0_inx_in_b:
            b_end_inx, a_end_inx = -1, 0
            for inx in a0_inx_in_b:
                if b_end_inx > inx:
                    continue
                this_inx = ab_max_inx(s_a, s_b[inx:])
                a_end_inx = max(a_end_inx, this_inx)
                res.append(s_a[:this_inx])
                b_end_inx = this_inx + inx
            res += lcs(s_a[a_end_inx:], s_b)
        else:
            res += lcs(s_a[1:], s_b)
    return res

def overlap_coefficient_contain(s1: str, s2: str) -> float:
    """Compute overlap coefficient between two strings.

    Need find the longest common substring between the two strings.

    when s1 contains s2, overlap coefficient is 1.
    when s2 contains s1, overlap coefficient not 1.
    """
    s1_len = len(s1)
    s2_len = len(s2)
    if s1_len == 0 or s2_len == 0:
        return 0
    # Find the all common substrings between the two strings
    cs_list = lcs(s1, s2)
    # Calculate the weight length of all common substrings
    overlap_coefficient = 0
    for cs in cs_list:
        # 取平方可以增加长串的权重，降低短串的权重。
        # 同时过滤掉短字符
        if len(cs) >= min(5, s2_len):
            overlap_coefficient += (len(cs) / s2_len) ** 2
    return overlap_coefficient


if __name__ == "__main__":
    print(overlap_coefficient_contain("你好aaa", "你好"))
    print(overlap_coefficient_contain("你好", "你好aaaa"))
    print(overlap_coefficient_contain("你a好zzzzzzzzzzzzzzzzzzz", "你好xxxxxxxxxxxxxxxxxxxxxxx"))
