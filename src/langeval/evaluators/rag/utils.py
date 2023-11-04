def str_inx(word_, string_):
    return [i for i in range(len(string_)) if string_[i] == word_]


def ab_max_inx(s_a, s_b):
    i, len_a, len_b = 0, len(s_a), len(s_b)
    while len_a > i and len_b > i and s_a[i] == s_b[i]:
        i += 1
    return i


def lcs(s_a, s_b):
    """计算两个字符串的所有不重复的公共字串，但是要求字串长度大于1
    适合中文，因为中文是字符单位的，不适合英文(英文需要按照空格分词，以词为单位）
    """
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
    return [i for i in res if len(i) > 1]

def overlap_coefficient_contain(s1: str, s2: str) -> float:
    """Compute overlap coefficient between two strings.

    Need find the longest common substring between the two strings.

    when s1 contains s2, overlap coefficient is 1.
    when s2 contains s1, overlap coefficient not 1.
    """

    # Find the longest common substring and its length
    lcs_list = lcs(s1, s2)

    lcs_length = len("".join(lcs_list))

    # Calculate the minimum length between the two strings
    s1_len = len(s1)
    s2_len = len(s2)
    if s1_len == 0 or s2_len == 0:
        return 0

    # Calculate the overlap coefficient
    overlap_coefficient = lcs_length / s2_len

    return overlap_coefficient


if __name__ == "__main__":
    print(overlap_coefficient_contain("你好aaa", "你好"))
    print(overlap_coefficient_contain("你好", "你好aaaa"))
    print(overlap_coefficient_contain("你a好", "你好"))
