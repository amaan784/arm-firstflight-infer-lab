from firstflight.bench import memory


def test_parse_time_v_maxrss():
    stderr = "\tElapsed: 0:01\n\tMaximum resident set size (kbytes): 123456\n\tOther: 1\n"
    assert memory.parse_time_v_maxrss(stderr) == 123456 * 1024
    assert memory.parse_time_v_maxrss("no maxrss here") is None
    assert memory.parse_time_v_maxrss("") is None
